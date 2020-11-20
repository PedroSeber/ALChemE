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

        stream_init = namedtuple('Stream', ['t1', 't2', 'cp', 'flow_rate'])
        self.streams[stream_name] = stream_init(t1*temp_unit, t2*temp_unit, cp*cp_unit, flow_rate*flow_unit)

    def make_graph(self):
        """
        This function receives two matrices representing the temperatures and C_P (heat capacity) values for a system of streams and
        returns a Temperature Interval Diagram, a Composite Curve for both hot and cold streams, and the minimum utilities needed
        
        Parameters
        --------------------
        temperatures: Nx2 array-like
            Stream temperatures, where each row represents a stream, the first column represents the initial temperature, and the second
            column represents the final temperature.
            For hot streams, the first value is higher. For cold streams, the second value is higher
        cp_vals: Nx1 array-like
            The heat capacity of each stream, in the order the streams were passed
        delta_t: double, optional, default = 10
            The minimum temperature difference between the hot and cold streams
        temp_unit: string, optional, default = " °C"
            A string that represents the units of your temperature scale, used for graph labels
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
        
        
        # Manipulating the temperatures so that the hot and cold values are on the same y-position, even though they're shifted by delta_t
        plotted_ylines = np.empty(temperatures.size, dtype = 'object')
        hot_streams = np.empty(len(temperatures), dtype = bool)
        for idx in range(len(temperatures)):
            if temperatures[idx, 0] > temperatures[idx, 1]:
                my_color = 'r'
                my_label1 = (str(temperatures[idx, 0]) + str(self.temp_unit) + " Hot side, " + str(temperatures[idx, 0]-self.delta_t) + str(self.temp_unit)
                            + " Cold side")
                my_label2 = (str(temperatures[idx, 1]) + str(self.temp_unit) + " Hot side, " + str(temperatures[idx, 1]-self.delta_t) + str(self.temp_unit)
                            + " Cold side")
                tplot1 = temperatures[idx, 0]
                tplot2 = temperatures[idx, 1]
                hot_streams[idx] = True
            else:
                my_color = 'b'
                my_label1 = (str(temperatures[idx, 0]+self.delta_t) + str(self.temp_unit) + " Hot side, " + str(temperatures[idx, 0]) + str(self.temp_unit)
                            + " Cold side")
                my_label2 = (str(temperatures[idx, 1]+self.delta_t) + str(self.temp_unit) + " Hot side, " + str(temperatures[idx, 1]) + str(self.temp_unit)
                            + " Cold side")
                tplot1 = temperatures[idx, 0] + self.delta_t
                tplot2 = temperatures[idx, 1] + self.delta_t

            ax1.vlines(idx, tplot1, tplot2, color = my_color, linewidth = 0.25) # Vertical line for each stream
            # Horizontal lines for each temperature
            if tplot1 not in plotted_ylines: # Ensures we don't plot two of the same line, screwing-up the labels
                ax1.axhline(tplot1, color = 'k', linewidth = 0.25)
                ax1.text(np.mean(ax1.get_xlim()), tplot1, my_label1, ha = 'center', va = 'bottom')
                plotted_ylines[idx] = tplot1
            if tplot2 not in plotted_ylines:
                ax1.axhline(tplot2, color = 'k', linewidth = 0.25)
                #ax1.hlines(tplot2, ax1.get_xlim()[0], ax1.get_xlim()[1], color = 'k', linewidth = 0.5)
                ax1.text(np.mean(ax1.get_xlim()), tplot2, my_label2, ha = 'center', va = 'bottom')
                plotted_ylines[-1-idx] = tplot2; # Was -1-idx+1
        ax1.set_xticks(range(len(temperatures)))
        ax1.set_xticklabels(x_tick_labels)
        plt.show(block = False)
        plotted_ylines = np.sort(plotted_ylines[plotted_ylines != None])

        # Getting the heat and enthalpies at each interval
        tmp1 = np.atleast_2d(np.max(temperatures[:sum(hot_streams), :], axis = 1)).T >= np.atleast_2d(plotted_ylines[1:])
        tmp2 = np.atleast_2d(np.min(temperatures[:sum(hot_streams), :], axis = 1)).T <= np.atleast_2d(plotted_ylines[:-1])
        streams_in_interval1 = (tmp1 & tmp2).astype(np.int8) # Numpy treats this as boolean if I don't convert the type
        tmp1 = np.atleast_2d(np.max(temperatures[sum(hot_streams):, :], axis = 1)).T >= np.atleast_2d(plotted_ylines[1:] - self.delta_t)
        tmp2 = np.atleast_2d(np.min(temperatures[sum(hot_streams):, :], axis = 1)).T <= np.atleast_2d(plotted_ylines[:-1] - self.delta_t)
        streams_in_interval2 = (tmp1 & tmp2).astype(np.int8)
        streams_in_interval = np.concatenate((streams_in_interval1, -streams_in_interval2)) # Hot streams, then -cold streams
        delta_plotted_ylines = plotted_ylines[1:] - plotted_ylines[:-1]
        q_interval = np.sum(streams_in_interval * cp_vals * delta_plotted_ylines, axis = 0) # sum(FCp_hot - FCp_cold) * delta_t_interval
        enthalpy_hot = np.sum(streams_in_interval1 * cp_vals[:sum(hot_streams)] * delta_plotted_ylines, axis = 0) # sum(FCp_hot) * delta_t
        enthalpy_cold = np.sum(streams_in_interval2 * cp_vals[sum(hot_streams):] * delta_plotted_ylines, axis = 0) # sum(FCp_cold) * delta_t
        
        q_interval = q_interval[::-1] # Flipping the heat array so it starts from the top
        q_sum = np.cumsum(q_interval)

        if np.min(q_sum) <= 0:
            first_utility = np.min(q_sum) # First utility is added to the minimum sum of heats, even if it isn't the first negative val
            first_utility_loc = np.where(q_sum == first_utility)[0][0] # np.where returns a tuple that contains an array containing the location
            first_utility = -first_utility # It's a heat going in, so we want it to be positive
            q_sum[first_utility_loc:] = q_sum[first_utility_loc:] + first_utility
            ax1.axhline(plotted_ylines[-2-first_utility_loc], color = 'k', linewidth = 0.5) # Arrays start at 0 but end at -1, so we need an extra -1 in this line and the next
            ax1.text(np.mean(ax1.get_xlim()), plotted_ylines[-2-first_utility_loc], 'Pinch Point', ha = 'center', va = 'top')
            print('The first utility is %g, located after the interval %d\n' % (first_utility, first_utility_loc+1))
            pass
        else: # No pinch point
            first_utility_loc = len(q_sum)
            print('Warning: there is no pinch point nor a first utility\n')
        
        last_utility = -q_sum[-1]
        enthalpy_hot = [0, enthalpy_hot] # The first value in enthalpy_hot is defined as 0
        # Shifting the cold enthalpy so that the first value starts at positive last_utility
        pdb.set_trace()
        enthalpy_cold = np.insert(enthalpy_cold[:-1], 0, -last_utility)
        print('The last utility is %g\n' % last_utility)
        """
        ax = gca;
        ti = ax.TightInset;
        ax.Position = [ti(1), ti(2), 1 - ti(1) - ti(3), 1 - ti(2) - ti(4)]; % Removing whitespace from the graph
        
        

        if min(q_sum) <= 0
            [first_utility, first_utility_loc] = min(q_sum); % First utility is added to the minimum sum of heats, even if it isn't the first negative val
            first_utility = -first_utility; % It's a heat going in, so we want it to be positive
            q_sum(first_utility_loc:end) = q_sum(first_utility_loc:end) + first_utility;
            yline(plotted_ylines(end-first_utility_loc), 'k', 'Pinch Point', 'FontWeight', 'bold', 'LabelHorizontalAlignment', 'center', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.0);
            fprintf('The first utility is %g, located after the interval %d\n', first_utility, first_utility_loc)
        else % No pinch point
            first_utility_loc = length(q_sum);
            fprintf('Warning: there is no pinch point nor a first utility\n')
        end
        last_utility = -q_sum(end);
        enthalpy_hot = [0, enthalpy_hot]; % The first value in enthalpy_hot is defined as 0
        % Shifting the cold enthalpy so that the first value starts at positive last_utility
        enthalpy_cold = [-last_utility, enthalpy_cold(1:end-1)];
        fprintf('The last utility is %g\n', last_utility)
        
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
        
        % TODO: adding heat exchangers automatically
        fig3 = figure('Position', [10 10 1296 1080], 'DefaultAxesFontSize', 20);
        copyobj(ax, fig3) % Copying figure 1
        close % Temporary, since we don't have code here yet
        """