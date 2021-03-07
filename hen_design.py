##############################################################################
# IMPORT CALLS
##############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unyt
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import pdb
import os
import pickle
import warnings
from gekko import GEKKO

class HEN:
    """
    A class that holds streams and exchangers, used to solve HEN problems
    """
    def __init__(self, delta_t = 10, flow_unit = unyt.kg/unyt.s, temp_unit = unyt.degC, cp_unit = unyt.J/(unyt.delta_degC*unyt.kg)):
        self.delta_t = delta_t * temp_unit
        self.flow_unit = flow_unit
        self.temp_unit = temp_unit
        self.cp_unit = cp_unit
        self.streams = pd.Series()
        self.inactive_hot_streams = 0
        self.inactive_cold_streams = 0
        self.hot_utilities = pd.Series()
        self.cold_utilities = pd.Series()
        self.exchangers = pd.Series()

        # Making unyt work since it doesn't like multiplying with °C and °F
        if self.temp_unit == unyt.degC:
            self.delta_temp_unit = unyt.delta_degC
        elif self.temp_unit == unyt.degF:
            self.delta_temp_unit = unyt.delta_degF
        else:
            self.delta_temp_unit = temp_unit

        self.heat_unit = self.flow_unit * self.cp_unit * self.delta_temp_unit
    
    def add_stream(self, t1, t2, cp = None, flow_rate = 1, heat = None, stream_name = None, temp_unit = None, cp_unit = None, flow_unit = None, heat_unit = None, HENOS_oe_tree = None):
        if cp is None and heat is None:
            raise ValueError('One of cp or heat must be passed')
        elif cp is not None and heat is not None:
            warnings.warn('You have passed both a cp and a heat. The heat input will be ignored')

        # Converting to default units (as declared via the HEN class)        
        if temp_unit is None:
            t1 *= self.temp_unit
            t2 *= self.temp_unit
        else:
            t1 *= temp_unit
            t1 = t1.to(self.temp_unit)
            t2 *= temp_unit
            t2 = t2.to(self.temp_unit)
        
        if flow_unit is None:
            flow_rate *= self.flow_unit
        else:
            flow_rate *= flow_unit
            flow_rate = flow_rate.to(self.flow_unit)
        
        if cp: # cp was passed; ignoring any value passed in the heat parameter
            if cp_unit is None:
                cp *= self.cp_unit
            else:
                cp *= cp_unit
                cp = cp.to(self.cp_unit)
        else: # Heat was passed
            if heat_unit is None:
                heat *= self.heat_unit
            else:
                heat *= heat_unit
                heat = heat.to(self.heat_unit)
            cp = heat / (np.abs(t2.value - t1.value) * self.delta_temp_unit * self.flow_unit)
        
        if stream_name is None:
            if t1 > t2: # Hot stream
                letter = 'H'
            else: # Cold stream
                letter = 'C'
            idx = 1
            while f'{letter}{idx}' in self.streams.keys():
                idx += 1
            stream_name = f'{letter}{idx}'

        # Generating the stream object and adding it to the HEN object
        temp = pd.Series(Stream(t1, t2, cp, flow_rate), [stream_name])
        self.streams = pd.concat([self.streams, temp])
        self.upper_limit = None

        if HENOS_oe_tree is not None:
            temp_diff = t2 - t1
            temp_diff = temp_diff.tolist() * self.delta_temp_unit
            oeDataVector = [stream_name, t1, t2, cp*flow_rate, cp*flow_rate*temp_diff]
            print(oeDataVector)
            HENOS_oe_tree.receive_new_stream(oeDataVector)

    def activate_stream(self, streams_to_change):
        if isinstance(streams_to_change, str): # Only one stream name was passed
            if not self.streams[streams_to_change].active:
                self.streams[streams_to_change].active = True
                if self.streams[streams_to_change].t1 > self.streams[streams_to_change].t2:
                    self.inactive_hot_streams -= 1
                else:
                    self.inactive_cold_streams -= 1
            else:
                raise ValueError(f'Stream {streams_to_change} is already inactive')
        elif isinstance(streams_to_change, (list, tuple, set)): # A container of stream names was passed
            for elem in streams_to_change:
                if not self.streams[elem].active:
                    self.streams[elem].active = True
                    if self.streams[elem].t1 > self.streams[elem].t2:
                        self.inactive_hot_streams -= 1
                    else:
                        self.inactive_cold_streams -= 1
                else:
                    warnings.warn(f'Stream {elem} is already inactive. Ignoring this input and continuing')
        else:
            raise TypeError('The streams_to_change parameter should be a string or list/tuple/set of strings')
    
    def inactivate_stream(self, streams_to_change):
        if isinstance(streams_to_change, str): # Only one stream name was passed
            if self.streams[streams_to_change].active:
                self.streams[streams_to_change].active = False
                if self.streams[streams_to_change].t1 > self.streams[streams_to_change].t2:
                    self.inactive_hot_streams += 1
                else:
                    self.inactive_cold_streams += 1
            else:
                raise ValueError(f'Stream {streams_to_change} is already active')
        elif isinstance(streams_to_change, (list, tuple, set)): # A container of stream names was passed
            for elem in streams_to_change:
                if self.streams[elem].active:
                    self.streams[elem].active = False
                    if self.streams[elem].t1 > self.streams[elem].t2:
                        self.inactive_hot_streams += 1
                    else:
                        self.inactive_cold_streams += 1
                else:
                    warnings.warn(f'Stream {elem} is already active. Ignoring this input and continuing')
        else:
            raise TypeError('The streams_to_change parameter should be a string or list/tuple/set of strings')
    
    def add_utility(self, utility_type, temperature, cost = 0, utility_name = None, temp_unit = None, cost_unit = None):
        if utility_type.casefold() in {'hot', 'hot utility', 'h', 'hu'}:
            utility_type = 'hot'
        elif utility_type.casefold() in {'cold', 'cold utility', 'c', 'cu'}:
            utility_type = 'cold'
        else:
            raise ValueError('The utility_type parameter should be either "hot" or "cold"')

        # Converting to default units (as declared via the HEN class)        
        if temp_unit is None:
            temperature *= self.temp_unit
        else:
            temperature *= temp_unit
            temperature = temperature.to(self.temp_unit)
        
        if cost_unit is None: # None that cost unit is in currency / energy (or currency / power), so that's why we divide it by self.heat unit
            cost /= self.heat_unit
        else:
            cost *= cost_unit
            cost = cost.to(1/self.heat_unit)
        
        if utility_name is None:
            if utility_type == 'hot': # Hot utility
                letter = 'HU'
            else: # Cold utility
                letter = 'CU'
            idx = 1
            while f'{letter}{idx}' in self.streams.keys():
                idx += 1
            utility_name = f'{letter}{idx}'
        
        # Generating the utility object and adding it to the HEN object
        temp = pd.Series(Utility(utility_type, temperature, cost), [utility_name])
        if utility_type == 'hot': # Hot utility
            self.hot_utilities = pd.concat([self.hot_utilities, temp])
        else: # Cold utility
            self.cold_utilities = pd.concat([self.cold_utilities, temp])
    
    def delete(self, obj_to_del):
        if obj_to_del in self.hot_utilities: # More will be added once exchangers to utilities get implemented
            del self.hot_utilities[obj_to_del]
        elif obj_to_del in self.cold_utilities:
            del self.cold_utilities[obj_to_del]
        elif obj_to_del in self.streams:
            for exchanger in self.streams[obj_to_del].connected_exchangers[::-1]: # All exchangers connected to a deleted stream should be removed
                self.delete(exchanger)
            del self.streams[obj_to_del]
        elif obj_to_del in self.exchangers:
            s1 = self.exchangers[obj_to_del].stream1 # Names of the streams connected by this exchanger
            s2 = self.exchangers[obj_to_del].stream2
            # Restoring the Q remaining and current temperature values of each stream
            if self.exchangers[obj_to_del].pinch == 'above':
                self.streams[s1].q_above_remaining += self.exchangers[obj_to_del].heat
                self.streams[s1].current_t_above += self.exchangers[obj_to_del].heat / (self.streams[s1].cp * self.streams[s1].flow_rate)
                self.streams[s2].q_above_remaining += self.exchangers[obj_to_del].heat
                self.streams[s2].current_t_above -= self.exchangers[obj_to_del].heat / (self.streams[s2].cp * self.streams[s2].flow_rate)
            else:
                self.streams[s1].q_below_remaining += self.exchangers[obj_to_del].heat
                self.streams[s1].current_t_below += self.exchangers[obj_to_del].heat / (self.streams[s1].cp * self.streams[s1].flow_rate)
                self.streams[s2].q_below_remaining += self.exchangers[obj_to_del].heat
                self.streams[s2].current_t_below -= self.exchangers[obj_to_del].heat / (self.streams[s2].cp * self.streams[s2].flow_rate)
            # Removing the exchanger from each stream's connected_exchangers list
            self.streams[s1].connected_exchangers.remove(obj_to_del)
            self.streams[s2].connected_exchangers.remove(obj_to_del)
            del self.exchangers[obj_to_del]
        else:
            raise ValueError(f'{obj_to_del} not found in the utilities, streams, or exchangers')

    def get_parameters(self):
        """
        This function obtains parameters (enthalpies, pinch temperature, heats above / below pinch) for the streams associated with this HEN object.
        """

        # Starting array from class data
        temperatures = np.empty( (len(self.streams) - self.inactive_hot_streams - self.inactive_cold_streams, 2) )
        cp_vals = np.empty( (len(self.streams) - self.inactive_hot_streams - self.inactive_cold_streams, 1) )

        for idx, values in enumerate(self.streams.items()): # values[0] has the stream names, values[1] has the properties
            if values[1].active: # Checks whether stream is active
                temperatures[idx, 0], temperatures[idx, 1] = values[1].t1, values[1].t2
                cp_vals[idx, 0] = values[1].cp * values[1].flow_rate
            else:
                idx -= 1
        
        self.hot_streams = temperatures[:, 0] > temperatures[:, 1]
        plotted_ylines = np.concatenate((temperatures[self.hot_streams, :].flatten(), temperatures[~self.hot_streams, :].flatten() + self.delta_t.value))
        self._plotted_ylines = np.sort(np.unique(plotted_ylines))

        # Getting the heat and enthalpies at each interval
        tmp1 = np.atleast_2d(np.max(temperatures[self.hot_streams, :], axis = 1)).T >= np.atleast_2d(self._plotted_ylines[1:])
        tmp2 = np.atleast_2d(np.min(temperatures[self.hot_streams, :], axis = 1)).T <= np.atleast_2d(self._plotted_ylines[:-1])
        streams_in_interval1 = (tmp1 & tmp2).astype(np.int8) # Numpy treats this as boolean if I don't convert the type
        tmp1 = np.atleast_2d(np.max(temperatures[~self.hot_streams, :], axis = 1)).T >= np.atleast_2d(self._plotted_ylines[1:] - self.delta_t.value)
        tmp2 = np.atleast_2d(np.min(temperatures[~self.hot_streams, :], axis = 1)).T <= np.atleast_2d(self._plotted_ylines[:-1] - self.delta_t.value)
        streams_in_interval2 = (tmp1 & tmp2).astype(np.int8)
        delta_plotted_ylines = self._plotted_ylines[1:] - self._plotted_ylines[:-1]
        enthalpy_hot = np.sum(streams_in_interval1 * cp_vals[self.hot_streams] * delta_plotted_ylines, axis = 0) # sum(FCp_hot) * ΔT
        enthalpy_cold = np.sum(streams_in_interval2 * cp_vals[~self.hot_streams] * delta_plotted_ylines, axis = 0) # sum(FCp_cold) * ΔT
        q_interval = enthalpy_hot - enthalpy_cold # sum(FCp_hot - FCp_cold) * ΔT_interval
        
        
        q_interval = q_interval[::-1] # Flipping the heat array so it starts from the top
        q_sum = np.cumsum(q_interval)

        if np.min(q_sum) <= 0:
            first_utility = np.min(q_sum) # First utility is added to the minimum sum of heats, even if it isn't the first negative val
            self.first_utility_loc = np.where(q_sum == first_utility)[0][0] # np.where returns a tuple that contains an array containing the location
            self.first_utility = -first_utility * self.flow_unit*self.delta_temp_unit*self.cp_unit # It's a heat going in, so we want it to be positive
            q_sum[self.first_utility_loc:] = q_sum[self.first_utility_loc:] + self.first_utility.value
            print('The first utility is %g %s, located after interval %d\n' % (self.first_utility, self.first_utility.units, self.first_utility_loc+1))
        else: # No pinch point
            self.first_utility = 0 * self.flow_unit*self.delta_temp_unit*self.cp_unit
            self.first_utility_loc = 0
            print('Warning: there is no pinch point nor a first utility\n')
        
        self.last_utility = q_sum[-1] * self.flow_unit*self.delta_temp_unit*self.cp_unit
        self.enthalpy_hot = np.insert(enthalpy_hot, 0, 0) # The first value in enthalpy_hot is defined as 0
        # Shifting the cold enthalpy so that the first value starts at positive last_utility
        self.enthalpy_cold = np.insert(enthalpy_cold, 0, self.last_utility)
        print('The last utility is %g %s\n' % (self.last_utility, self.last_utility.units))

        # Getting heats above / below pinch for each stream
        streams_in_interval = np.zeros((len(self.streams) - self.inactive_hot_streams - self.inactive_cold_streams, len(delta_plotted_ylines)), dtype = np.int8)
        streams_in_interval[self.hot_streams, :] = streams_in_interval1
        streams_in_interval[~self.hot_streams, :] = streams_in_interval2
        self._interval_heats = streams_in_interval * cp_vals * delta_plotted_ylines
        if self.first_utility_loc:
            q_above = np.sum(self._interval_heats[:, -1-self.first_utility_loc:], axis = 1)
            q_below = np.sum(self._interval_heats[:, :-1-self.first_utility_loc], axis = 1)
        else:
            q_below = np.sum(self._interval_heats, axis = 1)
            q_above = np.zeros_like(q_below)
        for idx, elem in enumerate(self.streams):
            if elem.active:
                elem.q_above = q_above[idx] * self.first_utility.units
                elem.q_above_remaining = q_above[idx] * self.first_utility.units
                elem.q_below = q_below[idx] * self.first_utility.units
                elem.q_below_remaining = q_below[idx] * self.first_utility.units
                if elem.current_t_above is None:
                    if self.first_utility_loc > 0: # Arrays begin at 0 but end at -1, so a pinch point at the highest interval causes issues
                        elem.current_t_above = self._plotted_ylines[-self.first_utility_loc - 2] * self.temp_unit - self.delta_t # Shifting the cold temperature by delta T
                    else:
                        elem.current_t_above = self._plotted_ylines[-1] * self.temp_unit - self.delta_t # Shifting the cold temperature by delta T
                elif elem.current_t_below is None:
                    if self.first_utility_loc > 0:
                        elem.current_t_below = self._plotted_ylines[-self.first_utility_loc - 2] * self.temp_unit
                    else:
                        elem.current_t_below = self._plotted_ylines[-1] * self.temp_unit
            else:
                idx -= 1
        
        # Heat limits used in the backend version of place_exchangers()
        self.upper_limit = np.zeros((np.sum(self.hot_streams) + len(self.hot_utilities), np.sum(~self.hot_streams) + len(self.cold_utilities)), dtype = np.object)
        self.lower_limit = np.zeros_like(self.upper_limit)
        self.forbidden = np.zeros_like(self.upper_limit, dtype = np.bool)
        self.required = np.zeros_like(self.forbidden)

    def make_tid(self, show_temperatures = True, show_properties = True, tab_control = None):
        """
        This function plots a temperature-interval diagram using the streams and exchangers currently associated with this HEN object.
        self.get_parameters() must be called before this function.
        """
        # Changing standard plotting options
        plt.rcParams['axes.titlesize'] = 5
        plt.rcParams['axes.labelsize'] = 5
        plt.rcParams['font.size'] = 3

        # Starting array from class data
        temperatures = np.empty( (len(self.streams) - self.inactive_hot_streams - self.inactive_cold_streams, 2) )
        cp_vals = np.empty( (len(self.streams) - self.inactive_hot_streams - self.inactive_cold_streams, 1) )
        x_tick_labels = np.empty(len(temperatures), dtype = 'object') # The names of the streams

        # We want the hot streams to come first on this plot
        hot_idx = 0
        cold_idx = -len(temperatures) + np.sum(self.hot_streams)
        for values in self.streams.items(): # values[0] has the stream names, values[1] has the properties
            if values[1].active: # Checks whether stream is active
                if values[1].t1 > values[1].t2: # Hot stream
                    temperatures[hot_idx, 0] = self._plotted_ylines.searchsorted(values[1].t1) # Conversion from temperature to an index; used to plot equidistant lines
                    temperatures[hot_idx, 1] = self._plotted_ylines.searchsorted(values[1].t2)
                    cp_vals[hot_idx, 0] = values[1].cp.value
                    x_tick_labels[hot_idx] = values[0]
                    hot_idx += 1
                else: # Cold stream
                    temperatures[cold_idx, 0] = self._plotted_ylines.searchsorted(values[1].t1 + self.delta_t) # Need to add ΔT 
                    temperatures[cold_idx, 1] = self._plotted_ylines.searchsorted(values[1].t2 + self.delta_t)
                    cp_vals[cold_idx, 0] = values[1].cp.value
                    x_tick_labels[cold_idx] = values[0]
                    cold_idx += 1

        # Plotting the temperature graphs
        fig1, ax1 = plt.subplots(dpi = 350)
        ax1.set_title('Temperature Interval Diagram')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-0.5 - 0.5*(len(self._plotted_ylines)//10), len(self._plotted_ylines) - 0.5 + 0.5*(len(self._plotted_ylines)//10)) #len()//10 is a correction for HEN with many temperature intervals
        ax1.set_yticks([])
        q_above_text_loc = ax1.get_ylim()[1] - 0.01*(ax1.get_ylim()[1] - ax1.get_ylim()[0])
        q_below_text_loc = ax1.get_ylim()[0] + 0.04*(ax1.get_ylim()[1] - ax1.get_ylim()[0])
        cp_text_loc = ax1.get_ylim()[0] + 0.01*(ax1.get_ylim()[1] - ax1.get_ylim()[0])
        # Setting the hot streams area and the cold streams area. Streams should be equally spaced
        # Hot streams go from 0.01 to 0.49; cold streams go from 0.51 to 0.99
        hot_distance = 0.48 / (sum(self.hot_streams) + 1)
        hot_idx = 1
        cold_distance = 0.48 / (sum(~self.hot_streams) + 1)
        cold_idx = 1
        x_tick_loc = [0] * len(temperatures)
        
        # Manipulating the temperatures so that the hot and cold values are on the same y-position, even though they're shifted by delta_t
        for idx in range(len(temperatures)):
            if temperatures[idx, 0] > temperatures[idx, 1]:
                my_color = 'r'
                my_marker = 'v'
                horizontal_loc = 0.01 + hot_idx * hot_distance
                hot_idx += 1
            else:
                my_color = 'b'
                my_marker = '^'
                horizontal_loc = 0.51 + cold_idx * cold_distance
                cold_idx += 1

            x_tick_loc[idx] = horizontal_loc
            ax1.vlines(horizontal_loc, temperatures[idx, 0], temperatures[idx, 1], color = my_color, linewidth = 0.25) # Vertical line for each stream
            ax1.plot(horizontal_loc, temperatures[idx, 1], color = my_color, marker = my_marker, markersize = 1) # Marker at the end of each vertical line
            if show_properties:
                q_above_text = r'$Q_{Top}$: %g %s' % (self.streams[x_tick_labels[idx]].q_above, self.heat_unit)
                ax1.text(horizontal_loc, q_above_text_loc, q_above_text, ha = 'center', va = 'top') # Heat above the pinch point
                q_below_text = r'$Q_{Bot}$: %g %s' % (self.streams[x_tick_labels[idx]].q_below, self.heat_unit)
                ax1.text(horizontal_loc, q_below_text_loc, q_below_text, ha = 'center', va = 'bottom') # Heat below the pinch point
                cp_unit_plot = str(self.cp_unit * self.flow_unit).replace('delta_deg', '°', 1)
                cp_text = r'$Fc_p$: %g %s' % (cp_vals[idx], cp_unit_plot)
                ax1.text(horizontal_loc, cp_text_loc, cp_text, ha = 'center', va = 'bottom') # Heat below the pinch point
        
        # Horizontal lines for each temperature
        for idx, elem in enumerate(self._plotted_ylines):
            ax1.axhline(idx, color = 'k', linewidth = 0.25)
            if show_temperatures:
                my_label1 = str(elem) + str(self.temp_unit) + '               ' # Extra spaces are used to pseudo-center the text
                my_label2 = '               ' + str(elem - self.delta_t.value) + str(self.temp_unit)
                ax1.text(np.mean(ax1.get_xlim()), idx, my_label1, ha = 'center', va = 'bottom', c = 'red')
                ax1.text(np.mean(ax1.get_xlim()), idx, my_label2, ha = 'center', va = 'bottom', c = 'blue')
        
        # Labeling the x-axis with the stream names
        ax1.set_xticks(x_tick_loc)
        ax1.set_xticklabels(x_tick_labels)

        # Adding the pinch point
        if self.first_utility_loc:
            ax1.axhline(len(self._plotted_ylines)-self.first_utility_loc-2, color = 'k', linewidth = 0.5) # Xth interval but (X-1)th line, that's why we need a -2
            if show_temperatures:
                ax1.text(np.mean(ax1.get_xlim()), len(self._plotted_ylines)-self.first_utility_loc-2 - 0.01, 'Pinch Point', ha = 'center', va = 'top')
        else:
            ax1.axhline(len(self._plotted_ylines) - 1, color = 'k', linewidth = 0.5)
            if show_temperatures:
                ax1.text(np.mean(ax1.get_xlim()), len(self._plotted_ylines) - 1 - 0.01, 'Pinch Point', ha = 'center', va = 'top')
        
        plt.show(block = False)
        if tab_control: # Embed into GUI
            generate_GUI_plot(fig1, tab_control, 'Temperature Interval Diagram')

    def make_cc(self, tab_control = None):
        plt.rcParams['axes.titlesize'] = 5
        plt.rcParams['axes.labelsize'] = 5
        plt.rcParams['font.size'] = 3

        fig2, ax2 = plt.subplots(dpi = 350)
        ax2.set_title('Composite Curve')
        ax2.set_ylabel(f'Temperature ({self.temp_unit})')
        ax2.set_xlabel(f'Enthalpy ({self.first_utility.units})')
        # Note: There may be issues if all streams on one side fully skip one or more intervals. Not sure how to test this properly.
        hot_index = np.concatenate(([True], np.sum(self._interval_heats[self.hot_streams], axis = 0, dtype = np.bool))) # First value in the hot scale is defined as 0, so it's always True
        cold_index = np.concatenate(([True], np.sum(self._interval_heats[~self.hot_streams], axis = 0, dtype = np.bool))) # First value in the cold scale is defined as the cold utility, so it's always True
        ax2.plot(np.cumsum(self.enthalpy_hot[hot_index]), self._plotted_ylines[hot_index], '-or', linewidth = 0.25, ms = 1.5)
        ax2.plot(np.cumsum(self.enthalpy_cold[cold_index]), self._plotted_ylines[cold_index] - self.delta_t.value, '-ob', linewidth = 0.25, ms = 1.5)
        ax2.set_ylim(ax2.get_ylim()[0], ax2.get_ylim()[1] * 1.05) # Making the y-axis a little longer to avoid CC's overlapping with text

        # Text showing the utilities and overlap
        top_text_loc = ax2.get_ylim()[1] - 0.03*(ax2.get_ylim()[1] - ax2.get_ylim()[0])
        cold_text_loc = ax2.get_xlim()[0] + 0.03*(ax2.get_xlim()[1] - ax2.get_xlim()[0])
        cold_text = 'Minimum Cold utility:\n%4g %s' % (self.last_utility, self.last_utility.units)
        ax2.text(cold_text_loc, top_text_loc, cold_text, ha = 'left', va = 'top')
        hot_text_loc = ax2.get_xlim()[1] - 0.03*(ax2.get_xlim()[1] - ax2.get_xlim()[0])
        hot_text = 'Minimum Hot utility:\n%4g %s' % (self.first_utility, self.first_utility.units)
        ax2.text(hot_text_loc, top_text_loc, hot_text, ha = 'right', va = 'top')
        overlap_text_loc = np.mean(ax2.get_xlim())
        overlap = np.minimum(np.cumsum(self.enthalpy_hot)[-1], np.cumsum(self.enthalpy_cold)[-1]) - np.maximum(self.enthalpy_hot[0], self.enthalpy_cold[0])
        overlap_text = 'Maximum Heat recovery:\n%4g %s' % (overlap, self.first_utility.units)
        ax2.text(overlap_text_loc, top_text_loc, overlap_text, ha = 'center', va = 'top')

        plt.show(block = False)
        if tab_control: # Embed into GUI
            generate_GUI_plot(fig2, tab_control, 'Composite Curve')

        """ TODO: remove whitespace around the graphs
        ax = gca;
        ti = ax.TightInset;
        ax.Position = [ti(1), ti(2), 1 - ti(1) - ti(3), 1 - ti(2) - ti(4)]; % Removing whitespace from the graph
        """

    def place_exchangers(self, pinch, upper = None, lower = None, forbidden = None, required = None):
        """
        Notes to self (WIP):
        Equations come from C.A. Floudas, "Nonlinear and Mixed-Integer Optimization", p. 283
        self._interval_heats has each stream as its rows and each interval as its columns, such that the topmost interval is the rightmost column
        """

        # Restricting the number of intervals based on the pinch point
        if pinch.casefold() in {'above', 'top', 'up'}:
            if self.first_utility_loc == 0:
                raise ValueError('This HEN doesn\'t have anything above the pinch')
            else:
                num_of_intervals = self.first_utility_loc + 1#self._interval_heats[:, -self.first_utility_loc-1:].shape[-1]
            pinch = 'above'
        elif pinch.casefold() in {'below', 'bottom', 'bot', 'down'}:
            if self.first_utility_loc == 0: # No pinch point --> take all intervals
                num_of_intervals = self._interval_heats.shape[-1]
            else:
                num_of_intervals = self._interval_heats[:, :-self.first_utility_loc-1].shape[-1]
            pinch = 'below'

        # Starting GEKKO
        m = GEKKO(remote = False)

        # Forbidden and required matches
        if forbidden is None:
            forbidden = np.zeros((np.sum(self.hot_streams) + len(self.hot_utilities), np.sum(~self.hot_streams) + len(self.cold_utilities)), dtype = np.bool)
        elif forbidden.shape != (np.sum(self.hot_streams) + len(self.hot_utilities), np.sum(~self.hot_streams) + len(self.cold_utilities)):
            raise ValueError('Forbidden must be a %dx%d matrix' % (np.sum(self.hot_streams) + len(self.hot_utilities), np.sum(~self.hot_streams) + len(self.cold_utilities)))
        if required is None:
            required = np.zeros_like(forbidden)
        elif required.shape != forbidden.shape:
            raise ValueError('Required must be a %dx%d matrix' % (forbidden.shape[0], forbidden.shape[1]))

        # Setting the upper heat exchanged limit for each pair of streams
        if upper is None: # Automatically set the upper limits
            upper = np.zeros_like(forbidden, dtype = np.float64)
            upper = self._get_maximum_heats(upper, pinch, num_of_intervals)
        elif isinstance(upper, (int, float)): # A single value was passed, representing a maximum threshold
            temp_upper = upper
            upper = np.zeros_like(forbidden, dtype = np.float64)
            upper = self._get_maximum_heats(upper, pinch, num_of_intervals)
            upper[upper > temp_upper] = temp_upper # Setting the given upper limit only for streams that naturally had a higher limit
        elif upper.shape != forbidden.shape: # An array-like was passed, but it has the wrong shape
            raise ValueError('Upper must be a %dx%d matrix' % (forbidden.shape[0], forbidden.shape[1]))
        else: # An array-like was passed
            for rowidx in range(upper.shape[0]):
                for colidx in range(upper.shape[1]):
                    upper[rowidx, colidx] = m.Const(upper[rowidx, colidx], f'Qlim_upper_{rowidx}{colidx}')
        # Setting the lower heat exchanged limit for each pair of streams
        if lower is None:
            lower = np.zeros_like(forbidden, dtype = np.float64)
        elif isinstance(lower, (int, float)): # A single value was passed, representing a minimum threshold
            if np.sum(lower > upper):
                raise ValueError(f'The lower threshold you passed is greater than the maximum heat of {np.sum(lower > upper)} streams')
            lower = np.ones_like(forbidden, dtype = np.float64) * lower
        elif upper.shape != forbidden.shape: # An array-like was passed, but it has the wrong shape
            raise ValueError('Lower must be a %dx%d matrix' % (forbidden.shape[0], forbidden.shape[1]))
        else: # An array-like was passed
            for rowidx in range(lower.shape[0]):
                for colidx in range(lower.shape[1]):
                    lower[rowidx, colidx] = m.Const(lower[rowidx, colidx], f'Qlim_lower_{rowidx}{colidx}')

        # First N rows of residuals are the N hot utilities
        # The extra interval represents the heats coming in from "above the highest interval" (always 0)
        residuals = np.zeros((np.sum(self.hot_streams) + len(self.hot_utilities), num_of_intervals + 1), dtype = np.object)
        for rowidx in range(residuals.shape[0]):
            # All R_0 and R_K must equal 0
            residuals[rowidx, 0] = m.Const(0, f'R_{rowidx}0') 
            residuals[rowidx, residuals.shape[1]-1] = m.Const(0, f'R_{rowidx}{residuals.shape[1]-1}')
            for colidx in range(1, residuals.shape[1] - 1):
                residuals[rowidx, colidx] = m.Var(0, lb = 0, name = f'R_{rowidx}{colidx}')
        residuals = np.fliplr(residuals) # R_X0 is above the highest interval, R_X1 is the highest interval, and so on

        # Q_exchanger is how much heat each exchanger will transfer
        # First N rows of Q_exchanger are the N hot utilities; first M columns are the M cold utilities
        Q_exchanger = np.zeros((np.sum(self.hot_streams) + len(self.hot_utilities), np.sum(~self.hot_streams) + len(self.cold_utilities), num_of_intervals ), dtype = np.object) # Hot streams, cold streams, and intervals
        for rowidx in range(Q_exchanger.shape[0]):
            for colidx in range(Q_exchanger.shape[1]):
                for intervalidx in range(Q_exchanger.shape[2]):
                    if rowidx < len(self.hot_utilities) and colidx < len(self.cold_utilities): # Matches between 2 utilities shouldn't exist
                        Q_exchanger[rowidx, colidx, intervalidx] = m.Const(0, f'Q_{rowidx}{colidx}{intervalidx+1}')
                    else: # Match involves at least one stream
                        Q_exchanger[rowidx, colidx, intervalidx] = m.Var(0, lb = 0, name = f'Q_{rowidx}{colidx}{intervalidx+1}')
        Q_exchanger = Q_exchanger[:, :, ::-1] #Q_XX1 is the highest interval, Q_XX2 is the 2nd highest, and so on

        Q_exc_tot = np.zeros((Q_exchanger.shape[:2]), dtype = np.object)
        for rowidx in range(Q_exchanger.shape[0]):
            for colidx in range(Q_exchanger.shape[1]):
                Q_exc_tot[rowidx, colidx] = m.Intermediate(m.sum(Q_exchanger[rowidx, colidx, :]), f'Q_tot_{rowidx}{colidx}')

        matches = np.zeros_like(Q_exc_tot, dtype = np.object) # Whether there is a heat exchanger between two streams
        for rowidx in range(matches.shape[0]):
            for colidx in range(matches.shape[1]):
                if rowidx < len(self.hot_utilities) and pinch == 'below': # No hot utilities are used below pinch
                    matches[rowidx, colidx] = m.Const(0, f'Y_{rowidx}{colidx}')
                elif colidx < len(self.cold_utilities) and pinch == 'above': # No cold utilities are used above pinch
                    matches[rowidx, colidx] = m.Const(0, f'Y_{rowidx}{colidx}')
                elif rowidx < len(self.hot_utilities) and colidx < len(self.cold_utilities): # No matches between utilities
                    matches[rowidx, colidx] = m.Const(0, f'Y_{rowidx}{colidx}')
                elif forbidden[rowidx, colidx]:
                    matches[rowidx, colidx] = m.Const(0, f'Y_{rowidx}{colidx}')
                elif required[rowidx, colidx]:
                    matches[rowidx, colidx] = m.Const(1, f'Y_{rowidx}{colidx}')
                else:
                    matches[rowidx, colidx] = m.Var(0, lb = 0, ub = 1, integer = True, name = f'Y_{rowidx}{colidx}')
        
        """
        Matricial equations, as described in C.A. Floudas, "Nonlinear and Mixed-Integer Optimization", p. 283
        These don't really work, but I'm keeping them for historical reasons
        eqn1_left = residuals[self.hot_utilities:, :-1] - residuals[self.hot_utilities:, 1:] + np.sum(Q_exchanger[self.hot_utilities:, :, :], axis = 1)*np.array(self._interval_heats[self.hot_streams], dtype = bool)
        eqn1_right = self._interval_heats[self.hot_streams]
        eqn2_left = residuals[:self.hot_utilities, :-1] - residuals[:self.hot_utilities, 1:] + np.sum(Q_exchanger[:self.hot_utilities, self.cold_utilities:, :], axis = 1)*np.ones((self.hot_utilities, self._interval_heats.shape[-1]), dtype = bool)
        eqn2_right = np.ones((self.hot_utilities, self._interval_heats.shape[-1]))*self.first_utility.value
        eqn3_left = np.sum(Q_exchanger[:, self.cold_utilities:, :], axis = 0)*np.array(self._interval_heats[~self.hot_streams] , dtype = bool)
        eqn3_right = self._interval_heats[~self.hot_streams]
        eqn4_left = np.sum(Q_exchanger[self.hot_utilities:, :self.cold_utilities, :], axis = 0)*np.ones((self.cold_utilities, self._interval_heats.shape[-1]), dtype = bool)
        eqn4_right = np.ones((self.cold_utilities, self._interval_heats.shape[-1]))*self.last_utility.value
        """

        # Eqn 1
        if pinch == 'above': # Used in Eqns 1 and 3. Needed because self._interval_heats has all intervals, not just the ones above pinch
            interval_correction = -num_of_intervals
        else:
            interval_correction = 0
        
        for stidx, rowidx in enumerate(range(len(self.hot_utilities), Q_exchanger.shape[0])): # stidx bc self.hot_streams has only streams, but no utilities
            for colidx in range(Q_exchanger.shape[2]):
                m.Equation(residuals[rowidx, colidx] - residuals[rowidx, colidx+1] +
                    m.sum(Q_exchanger[rowidx, :, colidx]) == self._interval_heats[self.hot_streams][stidx, interval_correction+colidx])

        # Eqn 2
        if pinch == 'above':
            for rowidx in range(len(self.hot_utilities)):
                m.Equation(m.sum(Q_exchanger[rowidx, len(self.cold_utilities):, :]) == self.first_utility.value)

        # Eqn 3
        for stidx, rowidx in enumerate(range(len(self.cold_utilities), Q_exchanger.shape[1])):
            for colidx in range(Q_exchanger.shape[2]):
                m.Equation(m.sum(Q_exchanger[:, rowidx, colidx]) == self._interval_heats[~self.hot_streams][stidx, interval_correction+colidx])
        
        # Eqn 4
        if pinch == 'below':
            for rowidx in range(len(self.cold_utilities)):
                m.Equation(m.sum(Q_exchanger[len(self.hot_utilities):, rowidx, :]) == self.last_utility.value)

        for rowidx in range(Q_exc_tot.shape[0]):
            for colidx in range(Q_exc_tot.shape[1]):
                m.Equation(lower[rowidx, colidx]*matches[rowidx, colidx] <= Q_exc_tot[rowidx, colidx])
                m.Equation(upper[rowidx, colidx]*matches[rowidx, colidx] >= Q_exc_tot[rowidx, colidx])
        m.Minimize(m.sum(matches))
        m.options.IMODE = 3 # Steady-state optimization
        m.options.solver = 1 # APOPT solver
        m.solver_options = ['minlp_maximum_iterations 500', \
                            # minlp iterations with integer solution
                            'minlp_max_iter_with_int_sol 100', \
                            # treat minlp as nlp
                            'minlp_as_nlp 0', \
                            # nlp sub-problem max iterations
                            'nlp_maximum_iterations 100', \
                            # 1 = depth first, 2 = breadth first
                            'minlp_branch_method 1', \
                            # maximum deviation from whole number
                            'minlp_integer_tol 0.02', \
                            # covergence tolerance
                            'minlp_gap_tol 0.01']
        m.solve()

        # Saving the results to variables (for ease of access)
        results = m.load_results()
        Y_results = np.zeros_like(matches, dtype = np.int8)
        Q_tot_results = np.zeros_like(Q_exc_tot, dtype = np.float)
        rowidx, colidx = 0, 0
        for elem in results: # Adding to both matrices in one loop
            if elem.startswith('q_tot'): # Adding Q_tot first since it comes first
                Q_tot_results[rowidx, colidx] = results[elem][0]
                colidx += 1
                if colidx == Q_tot_results.shape[1]:
                    colidx = 0
                    rowidx += 1
        Q_tot_results[Q_tot_results < 1e-9] = 0 # Rounding to prevent extremely small heats from being counted. Cutoff may need extra tuning
        Y_results[Q_tot_results > 0] = 1
        
        # Generating names to be used in a Pandas DataFrame with the results
        row_names = self.hot_utilities.index.append(self.streams.index[self.hot_streams])
        col_names = self.cold_utilities.index.append(self.streams.index[~self.hot_streams])
        """
        TODO: change how indices are generated, as this method will fail if there are inactive streams
        Perhaps change how self.hot_streams is defined to count all streams, then add another boolean array self.active and do self.hot_streams & self.active
        """
        #for idx, elem in enumerate(self.streams):
        #    if elem.active and self.hot_streams[idx]:
        #        row_names.append(self.streams.index[idx])
        #    elif elem.active and not self.hot_streams[idx]:
        #        col_names.append(self.streams.index[idx])
        #    else: # Inactive stream
        #        idx -= 1
        Y_results = pd.DataFrame(Y_results, row_names, col_names)
        Q_tot_results = pd.DataFrame(Q_tot_results, row_names, col_names)
        return Y_results, Q_tot_results


    def add_exchanger(self, stream1, stream2, heat = 'auto', ref_stream = 1, exchanger_delta_t = None, pinch = 'above', exchanger_name = None, U = 100, U_unit = unyt.J/(unyt.s*unyt.m**2*unyt.delta_degC), 
        exchanger_type = 'Fixed Head', cost_a = 0, cost_b = 0, pressure = 0, pressure_unit = unyt.Pa):

        # General data validation
        if exchanger_type.casefold() in {'fixed head', 'fixed', 'fixed-head'}:
            exchanger_type = 'Fixed Head'
        elif exchanger_type.casefold() in {'floating head', 'floating', 'floating-head'}:
            exchanger_type = 'Floating Head'
        elif exchanger_type.casefold() in {'u tube', 'u', 'u-tube'}:
            exchanger_type = 'U-Tube'
        elif exchanger_type.casefold() in {'kettle vaporizer', 'kettle', 'kettle-vaporizer'}:
            exchanger_type = 'Kettle Vaporizer'
        else:
            raise ValueError(f'{exchanger_type} is an invalid type')
        
        if cost_a < 0:
            raise ValueError('The cost_a parameter must be >= 0')
        elif cost_b < 0:
            raise ValueError('The cost_b parameter must be >= 0')
        
        if exchanger_name is None:
            idx = 1
            while f'E{idx}' in self.exchangers.keys():
                idx +=1
            exchanger_name = f'E{idx}'

        # Exchanger calculations
        if pinch.casefold() == 'above' or pinch.casefold() == 'top' or pinch.casefold() == 'up':
            if self.streams[stream1].t1 < self.streams[stream1].t2: # We want Stream1 to be the hot stream, but it's currently the cold stream
                stream1, stream2 = stream2, stream1
                if ref_stream == 1: # Inverting the ref_stream parameter since we inverted the streams
                    ref_stream = 2
                else:
                    ref_stream = 1
            
            if exchanger_delta_t is not None: # Operating the exchanger using a stream ΔT value
                if ref_stream == 1: # Temperature values must be referring to only one of the streams - the first stream in this case
                    heat = self.streams[stream1].cp * self.streams[stream1].flow_rate * (np.abs(exchanger_delta_t)*self.delta_temp_unit)
                else:
                    heat = self.streams[stream2].cp * self.streams[stream2].flow_rate * (np.abs(exchanger_delta_t)*self.delta_temp_unit)
            elif type(heat) is str and heat.casefold() == 'auto':
                if self.streams[stream1].cp * self.streams[stream1].flow_rate >= self.streams[stream2].cp * self.streams[stream2].flow_rate:
                    heat = np.minimum(self.streams[stream1].q_above_remaining, self.streams[stream2].q_above_remaining) # Maximum heat exchanged is the minimum total heat between streams
                else: # Can't use all the heat as the hot stream has a lower F*c_P, leading to a ΔT conflict
                    # Finding a temporary temperature where the heats exchanged are equal, but still respecting ΔT
                    T_F = (self.streams[stream1].cp * self.streams[stream1].flow_rate * self.streams[stream1].current_t_above.value * self.delta_temp_unit +
                    self.streams[stream2].cp * self.streams[stream2].flow_rate * (self.delta_t.value + self.streams[stream2].current_t_above.value) * self.delta_temp_unit )
                    T_F /= self.streams[stream1].cp * self.streams[stream1].flow_rate + self.streams[stream2].cp * self.streams[stream2].flow_rate
                    heat_temp = self.streams[stream1].cp * self.streams[stream1].flow_rate * (self.streams[stream1].current_t_above.value*self.delta_temp_unit - T_F)
                    # The heat_temp value can be greater than the minimum total heat of one of the streams, so we must take this into account
                    heat = np.minimum(self.streams[stream1].q_above_remaining, self.streams[stream2].q_above_remaining)
                    heat = np.minimum(heat, heat_temp)
            else: # Heat passed was a number, and no temperatures were passed
                heat = heat * self.streams[stream1].q_above.units

            s1_q_above = self.streams[stream1].q_above_remaining - heat
            s1_t_above = self.streams[stream1].current_t_above - heat/(self.streams[stream1].cp * self.streams[stream1].flow_rate)
            s2_q_above = self.streams[stream2].q_above_remaining - heat
            s2_t_above = self.streams[stream2].current_t_above + heat/(self.streams[stream2].cp * self.streams[stream2].flow_rate)
            
            # Data validation
            delta_T1 = self.streams[stream1].current_t_above - s2_t_above
            delta_T2 = s1_t_above - self.streams[stream2].current_t_above
            if self.streams[stream1].current_t_above < s2_t_above:
                raise ValueError(f'Match is thermodynamically impossible, as the hot stream is entering with a temperature of {self.streams[stream1].current_t_above:.4g}, '
                    f'while the cold stream is leaving with a temperature of {s2_t_above:.4g}')
            elif s1_t_above < self.streams[stream2].current_t_above:
                raise ValueError(f'Match is thermodynamically impossible, as the hot stream is leaving with a temperature of {s1_t_above:.4g}, '
                    f'while the cold stream is entering with a temperature of {self.streams[stream2].current_t_above:.4g}')
            elif delta_T1 < self.delta_t:
                warnings.warn(f"Warning: match violates minimum ΔT, which equals {self.delta_t:.4g}\nThis match's ΔT is {delta_T1:.4g}")
            elif delta_T2 < self.delta_t:
                warnings.warn(f"Warning: match violates minimum ΔT, which equals {self.delta_t:.4g}\nThis match's ΔT is {delta_T2:.4g}")
            
            # Recording the data
            self.streams[stream1].q_above_remaining = s1_q_above
            self.streams[stream1].current_t_above = s1_t_above
            self.streams[stream2].q_above_remaining = s2_q_above
            self.streams[stream2].current_t_above = s2_t_above

        elif pinch.casefold() == 'below' or pinch.casefold() == 'bottom' or pinch.casefold() == 'bot' or pinch.casefold == 'down':
            if self.streams[stream1].t1 < self.streams[stream1].t2: # We want Stream1 to be the hot stream, but it's currently the cold stream
                stream1, stream2 = stream2, stream1
                if ref_stream == 1: # Inverting the ref_stream parameter since we inverted the streams
                    ref_stream = 2
                else:
                    ref_stream = 1

            if exchanger_delta_t is not None: # Operating the exchanger using temperatures
                if ref_stream == 1: # Temperature values must be referring to only one of the streams - the first stream in this case
                    heat = self.streams[stream1].cp * self.streams[stream1].flow_rate * (np.abs(exchanger_delta_t)*self.delta_temp_unit)
                else:
                    heat = self.streams[stream2].cp * self.streams[stream2].flow_rate * (np.abs(exchanger_delta_t)*self.delta_temp_unit)
            elif type(heat) is str and heat.casefold() == 'auto':
                if self.streams[stream1].cp * self.streams[stream1].flow_rate <= self.streams[stream2].cp * self.streams[stream2].flow_rate:
                    heat = np.minimum(self.streams[stream1].q_below_remaining, self.streams[stream2].q_below_remaining) # Maximum heat exchanged is the minimum total heat between streams
                else: # Can't use all the heat as the hot stream has a lower F*c_P, leading to a ΔT conflict
                    # Finding a temporary temperature where the heats exchanged are equal, but still respecting ΔT
                    T_F = (self.streams[stream1].cp * self.streams[stream1].flow_rate * self.streams[stream1].current_t_below.value * self.delta_temp_unit +
                    self.streams[stream2].cp * self.streams[stream2].flow_rate * (self.delta_t.value + self.streams[stream2].current_t_below.value) * self.delta_temp_unit )
                    T_F /= self.streams[stream1].cp * self.streams[stream1].flow_rate + self.streams[stream2].cp * self.streams[stream2].flow_rate
                    heat_temp = self.streams[stream1].cp * self.streams[stream1].flow_rate * (self.streams[stream1].current_t_below.value*self.delta_temp_unit - T_F)
                    # The heat_temp value can be greater than the minimum total heat of one of the streams, so we must take this into account
                    heat = np.minimum(self.streams[stream1].q_below_remaining, self.streams[stream2].q_below_remaining)
                    heat = np.minimum(heat, heat_temp)
            else: # Heat passed was a number, and no temperatures were passed
                heat = heat * self.streams[stream1].q_above.units
            
            s1_q_below = self.streams[stream1].q_below_remaining - heat
            s1_t_below = self.streams[stream1].current_t_below - heat/(self.streams[stream1].cp * self.streams[stream1].flow_rate)
            s2_q_below = self.streams[stream2].q_below_remaining - heat
            s2_t_below = self.streams[stream2].current_t_below + heat/(self.streams[stream2].cp * self.streams[stream2].flow_rate)
            
            # Data validation
            delta_T1 = self.streams[stream1].current_t_below - s2_t_below
            delta_T2 = s1_t_below - self.streams[stream2].current_t_below
            if self.streams[stream1].current_t_below < s2_t_below:
                raise ValueError(f'Match is thermodynamically impossible, as the hot stream is entering with a temperature of {self.streams[stream1].current_t_below:.4g}, '
                    f'while the cold stream is leaving with a temperature of {s2_t_below:.4g}')
            elif s1_t_below < self.streams[stream2].current_t_below:
                raise ValueError(f'Match is thermodynamically impossible, as the hot stream is leaving with a temperature of {s1_t_below:.4g}, '
                    f'while the cold stream is entering with a temperature of {self.streams[stream2].current_t_below:.4g}')
            elif delta_T1 < self.delta_t:
                    print(f"Warning: match violates minimum ΔT, which equals {self.delta_t:.4g}\nThis match's ΔT is {delta_T1:.4g}")
            elif delta_T2 < self.delta_t:
                    print(f"Warning: match violates minimum ΔT, which equals {self.delta_t:.4g}\nThis match's ΔT is {delta_T2:.4g}")
            
            # Recording the data
            self.streams[stream1].q_below_remaining = s1_q_below
            self.streams[stream1].current_t_below = s1_t_below
            self.streams[stream2].q_below_remaining = s2_q_below
            self.streams[stream2].current_t_below = s2_t_below
        
        # Creating the exchanger object
        delta_T_lm = (delta_T1.value - delta_T2.value) / (np.log(delta_T1.value/delta_T2.value)) * self.delta_temp_unit
        U = U * U_unit
        pressure = pressure * pressure_unit
        temp = pd.Series(HeatExchanger(stream1, stream2, heat, pinch, U, delta_T_lm, exchanger_type, cost_a, cost_b, pressure, exchanger_name), [exchanger_name])
        self.exchangers = pd.concat([self.exchangers, temp])
        self.streams[stream1].connected_exchangers.append(exchanger_name) # Used when the stream is deleted
        self.streams[stream2].connected_exchangers.append(exchanger_name)
        
    def save(self, name, overwrite = False):
        if "." in name:
            file_name = name
        else:
            file_name = name + ".p"

        if not overwrite:
            while os.path.exists(file_name):
                word = file_name.split('.')
                file_name = word[0] + "DUPLICATE." +  word[1]
            print("The File Name you chose already exists in this directory. Saving as " + file_name + " instead")

        pickle.dump(self, open( file_name, "wb" ))
        
    @classmethod
    def load(cls,file = None):
        if file == None:
            files = os.listdir()
            file_list = []
            for myfile in files:
                if myfile.endswith('.p'):
                    file_list.append(myfile)
            if len(file_list) != 1:
                raise ValueError('You must supply a file name (with extension) to HEN.load()'+
            '\n Alternatively, ensure there\'s only one .p file in the working directory')
            else:
                file = file_list[0]
        return pickle.load(open(file, 'rb'))
    
    def _get_maximum_heats(self, upper, pinch, num_of_intervals):
        """
        Auxiliary function to calculate the maximum heat transferable between two streams.
        Shouldn't be called by the user; rather, it is automatically called by place_exchangers().
        """
        for rowidx in range(upper.shape[0]):
            for colidx in range(upper.shape[1]):
                if rowidx < len(self.hot_utilities) and pinch == 'below': # No hot utilities are used below pinch
                    upper[rowidx, colidx] = 0
                elif colidx < len(self.cold_utilities) and pinch == 'above': # No cold utilities are used above pinch
                    upper[rowidx, colidx] = 0
                elif rowidx < len(self.hot_utilities) and colidx < len(self.cold_utilities): # No matches between utilities
                    upper[rowidx, colidx] = 0
                elif rowidx < len(self.hot_utilities): # Match between hot utility and cold stream
                    temp_idx = np.sum(self.hot_streams) + colidx - len(self.cold_utilities)
                    if pinch == 'above':
                        upper[rowidx, colidx] = np.min((self.first_utility, np.sum(self._interval_heats[temp_idx, -num_of_intervals-1:]) ))
                    else:
                        upper[rowidx, colidx] = np.min((self.first_utility, np.sum(self._interval_heats[temp_idx, :num_of_intervals]) ))
                elif colidx < len(self.cold_utilities): # Match between hot stream and cold utility
                    temp_idx = rowidx - len(self.hot_utilities)
                    if pinch == 'above':
                        upper[rowidx, colidx] = np.min((np.sum(self._interval_heats[temp_idx, -num_of_intervals-1:]), self.last_utility))
                    else:
                        upper[rowidx, colidx] = np.min((np.sum(self._interval_heats[temp_idx, :num_of_intervals]), self.last_utility))
                else: # Match between two streams
                    temp_idx1 = rowidx - len(self.hot_utilities)
                    temp_idx2 = np.sum(self.hot_streams) + colidx - len(self.cold_utilities)
                    if pinch == 'above':
                        lowest_cold = (self._interval_heats[temp_idx2, -num_of_intervals:] != 0).argmax()
                        upper[rowidx, colidx] = np.min((np.sum(self._interval_heats[temp_idx1, -num_of_intervals+lowest_cold:]), np.sum(self._interval_heats[temp_idx2, -num_of_intervals:]) ))
                    else:
                        lowest_cold = (self._interval_heats[temp_idx2, :num_of_intervals] != 0).argmax()
                        upper[rowidx, colidx] = np.min((np.sum(self._interval_heats[temp_idx1, lowest_cold:num_of_intervals]), np.sum(self._interval_heats[temp_idx2, :num_of_intervals]) ))
        return upper

class Stream():
    def __init__(self, t1, t2, cp, flow_rate):
        self.t1 = t1
        self.t2 = t2
        self.cp = cp
        self.flow_rate = flow_rate
        self.q_above = None # Will be updated once pinch point is found
        self.q_below = None
        self.active = True
        self.connected_exchangers = [] # When stream is deleted, exchangers should also be deleted

        if self.t1 > self.t2: # Hot stream
            self.current_t_above = self.t1
            self.current_t_below = None # Will be updated once pinch point is found
        else: # Cold stream
            self.current_t_above = None
            self.current_t_below = self.t1
    
    def __repr__(self):
        if self.t1 > self.t2:
            stream_type = 'Hot'
        else:
            stream_type = 'Cold'
        text =(f'{stream_type} stream with T_in = {self.t1} and T_out = {self.t2}'
             f'c_p = {self.cp} and flow rate = {self.flow_rate}\n')
        if self.q_above is not None:
            text += f'Above pinch: {self.q_above} total, {self.q_above_remaining:.6g} remaining, T = {self.current_t_above:.4g}\n'
            text += f'Below pinch: {self.q_below} total, {self.q_below_remaining:.6g} remaining, T = {self.current_t_below:.4g}\n'
        return text

class Utility():
    def __init__(self, utility_type, temperature, cost):
        self.utility_type = utility_type
        self.temperature = temperature
        self.cost = cost
        # More things can be added if the get_parameters() function is updated to allow multiple utilities
    
    def __repr__(self):
        text = (f'A {self.utility_type} utility at a temperature of {self.temperature}\n'
            f'Has a cost of {self.cost}')
        return text

class HeatExchanger():
    def __init__(self, stream1, stream2, heat, pinch, U, delta_T_lm, exchanger_type, cost_a, cost_b, pressure, name):
        self.stream1 = stream1
        self.stream2 = stream2
        self.heat = heat
        self.pinch = pinch
        self.U = U
        self.delta_T_lm = delta_T_lm
        self.area = self.heat / (self.U * self.delta_T_lm)
        self.exchanger_type = exchanger_type
        self.pressure = pressure
        self.name = name # Used in the self.delete() function

        # Calculating costs
        Ac = self.area.to('ft**2').value
        pressure_c = self.pressure.to('psi').value
        if exchanger_type == 'Floating Head':
            self.cost_base = np.exp(12.0310) * Ac**(-0.8709) * np.exp(0.09005*np.log(Ac)**2)
        elif exchanger_type == 'Fixed Head':
            self.cost_base = np.exp(11.4185) * Ac**(-0.9228) * np.exp(0.09861*np.log(Ac)**2)
        elif exchanger_type == 'U-Tube':
            self.cost_base = np.exp(11.5510) * Ac**(-0.9186) * np.exp(0.09790*np.log(Ac)**2)
        elif exchanger_type == 'Kettle Vaporizer':
            self.cost_base = np.exp(12.3310) * Ac**(-0.8709) * np.exp(0.09005*np.log(Ac)**2)
        Fm = cost_a + (Ac/100)**cost_b
        if pressure_c > 100:
            Fp = 0.9803 + 0.018*pressure_c/100 + 0.0017*(pressure_c/100)**2
        else:
            Fp = 1
        self.cost_fob = self.cost_base * Fm * Fp

    def __repr__(self):
        text = (f'A {self.exchanger_type} heat exchanger exchanging {self.heat:.6g} between {self.stream1} and {self.stream2} {self.pinch} the pinch\n'
            f'Has a U = {self.U:.4g}, area = {self.area:.4g}, and ΔT_lm = {self.delta_T_lm:.4g}\n'
            f'Has a base cost of ${self.cost_base:,.2f} and a free on board cost of ${self.cost_fob:,.2f}\n')
        return text


# UTILITY FUNCTIONS
def generate_GUI_plot(plot, tabControl, tab_name):
    """
    A function which generates a relevant model plot onto the GUI
    """
    new_tab = ttk.Frame(tabControl)
    tabControl.add(new_tab, text=tab_name)
    tabControl.pack(expand=1, fill='both')
    new_canvas = FigureCanvasTkAgg(plot, master=new_tab)
    new_canvas.draw()

    new_canvas.get_tk_widget().pack()
