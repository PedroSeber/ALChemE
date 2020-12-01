import numpy as np
import matplotlib.pyplot as plt
import unyt
from collections import namedtuple, OrderedDict
import pdb

class HEN:
    """
    A class that holds streams and exchangers, used to solve HEN problems
    """
    def __init__(self, delta_t = 10, cold_cost = 7e-3, hot_cost = 11e-3, flow_unit = unyt.kg/unyt.s, temp_unit = unyt.degC, cp_unit = unyt.J/(unyt.delta_degC*unyt.kg)):
        self.delta_t = delta_t
        self.cold_cost = cold_cost
        self.hot_cost = hot_cost
        self.flow_unit = flow_unit
        self.temp_unit = temp_unit
        self.cp_unit = cp_unit
        self.first_utility = None
        self.streams = OrderedDict()
    
    def add_stream(self, t1, t2, cp, flow_rate = 1, stream_name = None, flow_unit = None, temp_unit = None, cp_unit = None):
        if flow_unit is None:
            flow_unit = self.flow_unit
        if temp_unit is None:
            temp_unit = self.temp_unit
        if cp_unit is None:
            cp_unit = self.cp_unit
        if stream_name is None:
            if t1 > t2: # Hot stream
                letter = 'H'
            else: # Cold stream
                letter = 'C'
            idx = 1
            while f'{letter}{idx}' in self.streams.keys():
                idx += 1
            stream_name = f'{letter}{idx}'

        self.streams[stream_name] = Stream(t1, t2, cp, flow_rate, flow_unit, temp_unit, cp_unit)

    def get_parameters(self):
        """
        This function obtains parameters (enthalpies, pinch temperature, heats above / below pinch) for the streams associated with this HEN object.
        """
        # Making unyt work since it doesn't allow multiplication of °C and °F
        if self.temp_unit == unyt.degC:
            local_temp_unit = unyt.delta_degC
        elif self.temp_unit == unyt.degF:
            local_temp_unit = unyt.delta_degF

        # Starting array from class data
        temperatures = np.empty( (len(self.streams), 2) )
        cp_vals = np.empty( (len(self.streams), 1) )

        for idx, values in enumerate(self.streams.items()): # values[0] has the stream names, values[1] has the properties
            temperatures[idx, 0], temperatures[idx, 1] = values[1].t1.value, values[1].t2.value
            cp_vals[idx, 0] = values[1].cp.value
        
        hot_streams = temperatures[:, 0] > temperatures[:, 1]
        plotted_ylines = np.concatenate((temperatures[hot_streams, :].flatten(), temperatures[~hot_streams, :].flatten() + self.delta_t))
        self._plotted_ylines = np.sort(np.unique(plotted_ylines))

        # Getting the heat and enthalpies at each interval
        tmp1 = np.atleast_2d(np.max(temperatures[hot_streams, :], axis = 1)).T >= np.atleast_2d(self._plotted_ylines[1:])
        tmp2 = np.atleast_2d(np.min(temperatures[hot_streams, :], axis = 1)).T <= np.atleast_2d(self._plotted_ylines[:-1])
        streams_in_interval1 = (tmp1 & tmp2).astype(np.int8) # Numpy treats this as boolean if I don't convert the type
        tmp1 = np.atleast_2d(np.max(temperatures[~hot_streams, :], axis = 1)).T >= np.atleast_2d(self._plotted_ylines[1:] - self.delta_t)
        tmp2 = np.atleast_2d(np.min(temperatures[~hot_streams, :], axis = 1)).T <= np.atleast_2d(self._plotted_ylines[:-1] - self.delta_t)
        streams_in_interval2 = (tmp1 & tmp2).astype(np.int8)
        delta_plotted_ylines = self._plotted_ylines[1:] - self._plotted_ylines[:-1]
        enthalpy_hot = np.sum(streams_in_interval1 * cp_vals[hot_streams] * delta_plotted_ylines, axis = 0) # sum(FCp_hot) * delta_t
        enthalpy_cold = np.sum(streams_in_interval2 * cp_vals[~hot_streams] * delta_plotted_ylines, axis = 0) # sum(FCp_cold) * delta_t
        q_interval = enthalpy_hot - enthalpy_cold # sum(FCp_hot - FCp_cold) * delta_t_interval
        
        
        q_interval = q_interval[::-1] # Flipping the heat array so it starts from the top
        q_sum = np.cumsum(q_interval)

        if np.min(q_sum) <= 0:
            first_utility = np.min(q_sum) # First utility is added to the minimum sum of heats, even if it isn't the first negative val
            self.first_utility_loc = np.where(q_sum == first_utility)[0][0] # np.where returns a tuple that contains an array containing the location
            self.first_utility = -first_utility * self.flow_unit*local_temp_unit*self.cp_unit # It's a heat going in, so we want it to be positive
            q_sum[self.first_utility_loc:] = q_sum[self.first_utility_loc:] + self.first_utility.value
            print('The first utility is %g %s, located after the interval %d\n' % (self.first_utility, self.first_utility.units, self.first_utility_loc+1))
        else: # No pinch point
            self.first_utility_loc = len(q_sum)
            print('Warning: there is no pinch point nor a first utility\n')
        
        self.last_utility = -q_sum[-1] * self.flow_unit*local_temp_unit*self.cp_unit
        self.enthalpy_hot = [0, enthalpy_hot] # The first value in enthalpy_hot is defined as 0
        # Shifting the cold enthalpy so that the first value starts at positive last_utility
        self.enthalpy_cold = np.insert(enthalpy_cold[:-1], 0, -self.last_utility)
        print('The last utility is %g %s\n' % (self.last_utility, self.last_utility.units))

        # Getting heats above / below pinch for each stream
        streams_in_interval = np.zeros((len(self.streams), len(delta_plotted_ylines)), dtype = np.int8)
        streams_in_interval[hot_streams, :] = streams_in_interval1
        streams_in_interval[~hot_streams, :] = -1*streams_in_interval2
        q_above = np.sum(streams_in_interval[:, -1-self.first_utility_loc:] * cp_vals * delta_plotted_ylines[-1-self.first_utility_loc:], axis = 1)
        q_below = np.sum(streams_in_interval[:, :-1-self.first_utility_loc] * cp_vals * delta_plotted_ylines[:-1-self.first_utility_loc], axis = 1)
        for idx, elem in enumerate(self.streams):
            self.streams[elem].q_above = q_above[idx] * self.first_utility.units
            self.streams[elem].q_below = q_below[idx] * self.first_utility.units

    def make_tid(self): # Add a show_middle_temps, show_q parameter for customization
        """
        This function plots a temperature-interval diagram using the streams and exchangers currently associated with this HEN object.
        self.get_parameters() must be called before this function.
        """
        # Changing standard plotting options
        plt.rcParams['axes.titlesize'] = 5
        plt.rcParams['axes.labelsize'] = 5
        plt.rcParams['font.size'] = 3

        # Starting array from class data
        temperatures = np.empty( (len(self.streams), 2) )
        cp_vals = np.empty( (len(self.streams), 1) )
        x_tick_labels = np.empty(len(temperatures), dtype = 'object') # The names of the streams

        for idx, values in enumerate(self.streams.items()): # values[0] has the stream names, values[1] has the properties
            temperatures[idx, 0], temperatures[idx, 1] = values[1].t1.value, values[1].t2.value
            cp_vals[idx, 0] = values[1].cp.value
            x_tick_labels[idx] = values[0]


        # Plotting the temperature graphs
        fig1, ax1 = plt.subplots(dpi = 350)
        ax1.set_title('Temperature Interval Diagram')
        axis_delta = max(self.delta_t, 20) # Shifts the y-axis by at least 20 degrees
        ax1.set_xlim(-0.5, len(temperatures)-0.5)
        ax1.set_ylim(np.min(temperatures) - axis_delta, np.max(temperatures) + axis_delta)
        ax1.set_ylabel(f"Hot temperatures ({self.temp_unit})")
        ax1.set_yticks(np.linspace(np.min(temperatures) - axis_delta, np.max(temperatures) + axis_delta, 11))
        q_above_text_loc = ax1.get_ylim()[1] - 0.01*(ax1.get_ylim()[1] - ax1.get_ylim()[0])
        
        
        # Manipulating the temperatures so that the hot and cold values are on the same y-position, even though they're shifted by delta_t
        plotted_ylines = np.empty(temperatures.size, dtype = 'object')
        for idx in range(len(temperatures)):
            if temperatures[idx, 0] > temperatures[idx, 1]:
                my_color = 'r'
                tplot1 = temperatures[idx, 0]
                tplot2 = temperatures[idx, 1]
            else:
                my_color = 'b'
                tplot1 = temperatures[idx, 0] + self.delta_t
                tplot2 = temperatures[idx, 1] + self.delta_t

            ax1.vlines(idx, tplot1, tplot2, color = my_color, linewidth = 0.25) # Vertical line for each stream
            q_above_text = r'$Q_{Top}$ = %g %s' % (self.streams[x_tick_labels[idx]].q_above, self.streams[x_tick_labels[idx]].q_above.units)
            ax1.text(idx, q_above_text_loc, q_above_text, ha = 'center', va = 'top') # Heat above the pinch point
            print(ax1.get_ylim())
        
        # Horizontal lines for each temperature
        for elem in self._plotted_ylines:
            ax1.axhline(elem, color = 'k', linewidth = 0.25)
            my_label = str(elem) + str(self.temp_unit) + ' Hot side, ' + str(elem - self.delta_t) + str(self.temp_unit) + ' Cold side'
            ax1.text(np.mean(ax1.get_xlim()), elem, my_label, ha = 'center', va = 'bottom')
        
        # Labeling the x-axis with the stream names
        ax1.set_xticks(range(len(temperatures)))
        ax1.set_xticklabels(x_tick_labels)

        # Adding the pinch point
        ax1.axhline(self._plotted_ylines[-2-self.first_utility_loc], color = 'k', linewidth = 0.5) # Arrays start at 0 but end at -1, so we need an extra -1 in this line and the next
        ax1.text(np.mean(ax1.get_xlim()), self._plotted_ylines[-2-self.first_utility_loc] - 1, 'Pinch Point', ha = 'center', va = 'top')
        plt.show(block = False)

        """
        ax = gca;
        ti = ax.TightInset;
        ax.Position = [ti(1), ti(2), 1 - ti(1) - ti(3), 1 - ti(2) - ti(4)]; % Removing whitespace from the graph
        
        
        % Adding heats and Cp values to the graph
        q_above = sum(streams_in_interval(:, end-first_utility_loc+1:end) .* cp_vals .* delta_plotted_ylines(end-first_utility_loc+1:end), 2);
        q_below = sum(streams_in_interval(:, 1:end-first_utility_loc) .* cp_vals .* delta_plotted_ylines(1:end-first_utility_loc), 2);
        for idx = 1:length(temperatures)
            text(idx-0.08, ax.YLim(1)+(ax.YLim(2)-ax.YLim(1))/190*5, sprintf("C_P = %g", cp_vals(idx))) % C_P values near the bottom
            text(idx-0.10, ax.YLim(1)+(ax.YLim(2)-ax.YLim(1))/190*10, sprintf("Q_B_o_t = %g", q_below(idx)))
            text(idx-0.10, ax.YLim(2)-5, sprintf("Q_T_o_p = %g", q_above(idx)))
        end
        
        % Plotting the temperature-enthalpy diagram
        fig2 = figure('units', 'normalized', 'outerposition', [0 0.03 0.925 0.925], 'DefaultAxesFontSize', 20);
        hold on
        title('Composite Curves')
        xlabel('Enthalpy')
        ylabel(strcat("Temperature (", temp_unit, " )"))
        % There may not be hot or cold streams at a given interval. This slicing fixes that, at least partially
        % There will still be issues if all streams on one side "skip" at least one interval
        plot(cumsum(enthalpy_hot), plotted_ylines(end-length(enthalpy_hot)+1:end), '-or')
        plot(cumsum(enthalpy_cold), plotted_ylines(1:length(enthalpy_cold)) - delta_t, '-ob')
        """

class Stream():
    def __init__(self, t1, t2, cp, flow_rate, flow_unit, temp_unit, cp_unit):
        self.t1 = t1 * temp_unit
        self.t2 = t2 * temp_unit
        self.cp = cp * cp_unit
        self.flow_rate = flow_rate * flow_unit
        self.q_above = None
        self.q_below = None