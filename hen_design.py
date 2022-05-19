##############################################################################
# IMPORT CALLS
##############################################################################
# Used everywhere
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unyt
# Used in the GUI
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import warnings
# Used in the HEN.save() and HEN.load() functions
import os
import pickle
# Used in the HEN.solve_HEN() functions
from gekko import GEKKO
from time import time
from itertools import product, combinations
from joblib import Parallel, delayed
import pdb

class HEN:
    """
    A class that holds streams and exchangers, used to solve HEN problems
    """
    def __init__(self, delta_t = 10, flow_unit = unyt.kg/unyt.s, temp_unit = unyt.degC, cp_unit = unyt.J/(unyt.delta_degC*unyt.kg), GUI_terminal = None):
        self.delta_t = delta_t * temp_unit
        self.flow_unit = flow_unit
        self.temp_unit = temp_unit
        self.cp_unit = cp_unit
        self.streams = pd.Series()
        self.hot_utilities = pd.Series()
        self.cold_utilities = pd.Series()
        self.exchangers = pd.Series()
        self.active_streams = np.array([], dtype = np.bool)
        self.GUI_terminal = GUI_terminal
        
        # Making unyt work since it doesn't like multiplying with °C and °F
        if self.temp_unit == unyt.degC:
            self.delta_temp_unit = unyt.delta_degC
        elif self.temp_unit == unyt.degF:
            self.delta_temp_unit = unyt.delta_degF
        else:
            self.delta_temp_unit = temp_unit

        self.heat_unit = self.flow_unit * self.cp_unit * self.delta_temp_unit
    
    def add_stream(self, t1, t2, cp = None, flow_rate = 1, heat = None, stream_name = None, temp_unit = None, cp_unit = None, flow_unit = None, heat_unit = None, GUI_oe_tree = None):
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
        self.active_streams = np.append(self.active_streams, True)

        if GUI_oe_tree is not None:
            temp_diff = t2 - t1
            temp_diff = temp_diff.tolist() * self.delta_temp_unit
            oeDataVector = [stream_name, t1, t2, cp*flow_rate, abs(cp*flow_rate*temp_diff)]
            print(oeDataVector)
            GUI_oe_tree.receive_new_stream(oeDataVector)

    def activate_stream(self, streams_to_change):
        if isinstance(streams_to_change, str): # Only one stream name was passed
            if not self.streams[streams_to_change].active:
                self.streams[streams_to_change].active = True
                loc = self.streams.index.get_loc(streams_to_change)
                self.active_streams[loc] = True
            else:
                raise ValueError(f'Stream {streams_to_change} is already inactive')
        elif isinstance(streams_to_change, (list, tuple, set)): # A container of stream names was passed
            for elem in streams_to_change:
                if not self.streams[elem].active:
                    self.streams[elem].active = True
                    loc = self.streams.index.get_loc(elem)
                    self.active_streams[loc] = True
                else:
                    warnings.warn(f'Stream {elem} is already inactive. Ignoring this input and continuing')
        else:
            raise TypeError('The streams_to_change parameter should be a string or list/tuple/set of strings')
    
    def deactivate_stream(self, streams_to_change):
        if isinstance(streams_to_change, str): # Only one stream name was passed
            if self.streams[streams_to_change].active:
                self.streams[streams_to_change].active = False
                loc = self.streams.index.get_loc(streams_to_change)
                self.active_streams[loc] = False
            else:
                raise ValueError(f'Stream {streams_to_change} is already active')
        elif isinstance(streams_to_change, (list, tuple, set)): # A container of stream names was passed
            for elem in streams_to_change:
                if self.streams[elem].active:
                    self.streams[elem].active = False
                    loc = self.streams.index.get_loc(elem)
                    self.active_streams[loc] = False
                else:
                    warnings.warn(f'Stream {elem} is already active. Ignoring this input and continuing')
        else:
            raise TypeError('The streams_to_change parameter should be a string or list/tuple/set of strings')
    
    def add_utility(self, utility_type, temperature, cost = 0, utility_name = None, temp_unit = None, cost_unit = None, GUI_oe_tree = None):
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
            idx = 1
            if utility_type == 'hot': # Hot utility
                letter = 'HU'
                while f'{letter}{idx}' in self.hot_utilities.keys():
                    idx += 1
            else: # Cold utility
                letter = 'CU'
                while f'{letter}{idx}' in self.cold_utilities.keys():
                    idx += 1
            utility_name = f'{letter}{idx}'
            
        
        # Generating the utility object and adding it to the HEN object
        temp = pd.Series(Utility(utility_type, temperature, cost), [utility_name])
        if utility_type == 'hot': # Hot utility
            self.hot_utilities = pd.concat([self.hot_utilities, temp])
        else: # Cold utility
            self.cold_utilities = pd.concat([self.cold_utilities, temp])
            
        if GUI_oe_tree is not None:
            oeDataVector = [utility_name, utility_type, temperature, cost, '', 'Active']
            GUI_oe_tree.receive_new_utility(oeDataVector)            
    
    def delete(self, obj_to_del):
        if obj_to_del in self.hot_utilities: # More will be added once exchangers to utilities get implemented
            del self.hot_utilities[obj_to_del]
        elif obj_to_del in self.cold_utilities:
            del self.cold_utilities[obj_to_del]
        elif obj_to_del in self.streams:
            for exchanger in self.streams[obj_to_del].connected_exchangers[::-1]: # All exchangers connected to a deleted stream should be removed
                self.delete(exchanger)
            loc = self.streams.index.get_loc(obj_to_del)
            self.active_streams = np.delete(self.active_streams, loc)
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
        temperatures = np.empty( (len(self.streams), 2) )
        cp_vals = np.empty( (len(self.streams), 1) )

        for idx, values in enumerate(self.streams.items()): # values[0] has the stream names, values[1] has the properties
            temperatures[idx, 0], temperatures[idx, 1] = values[1].t1, values[1].t2
            cp_vals[idx, 0] = values[1].cp * values[1].flow_rate
        
        self.hot_streams = temperatures[:, 0] > temperatures[:, 1]
        plotted_ylines = np.concatenate((temperatures[self.hot_streams&self.active_streams, :].flatten(), temperatures[~self.hot_streams&self.active_streams, :].flatten() + self.delta_t.value))
        self._plotted_ylines = np.sort(np.unique(plotted_ylines))

        # Getting the heat and enthalpies at each interval
        tmp1 = np.atleast_2d(np.max(temperatures[self.hot_streams&self.active_streams, :], axis = 1)).T >= np.atleast_2d(self._plotted_ylines[1:])
        tmp2 = np.atleast_2d(np.min(temperatures[self.hot_streams&self.active_streams, :], axis = 1)).T <= np.atleast_2d(self._plotted_ylines[:-1])
        streams_in_interval1 = (tmp1 & tmp2).astype(np.int8) # Numpy treats this as boolean if I don't convert the type
        tmp1 = np.atleast_2d(np.max(temperatures[~self.hot_streams&self.active_streams, :], axis = 1)).T >= np.atleast_2d(self._plotted_ylines[1:] - self.delta_t.value)
        tmp2 = np.atleast_2d(np.min(temperatures[~self.hot_streams&self.active_streams, :], axis = 1)).T <= np.atleast_2d(self._plotted_ylines[:-1] - self.delta_t.value)
        streams_in_interval2 = (tmp1 & tmp2).astype(np.int8)
        delta_plotted_ylines = self._plotted_ylines[1:] - self._plotted_ylines[:-1]
        enthalpy_hot = np.sum(streams_in_interval1 * cp_vals[self.hot_streams&self.active_streams] * delta_plotted_ylines, axis = 0) # sum(FCp_hot) * ΔT
        enthalpy_cold = np.sum(streams_in_interval2 * cp_vals[~self.hot_streams&self.active_streams] * delta_plotted_ylines, axis = 0) # sum(FCp_cold) * ΔT
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
            if self.GUI_terminal is not None:
                self.GUI_terminal.print2screen('Warning: there is no pinch point nor a first utility\n', False)
        
        self.last_utility = q_sum[-1] * self.flow_unit*self.delta_temp_unit*self.cp_unit
        self.enthalpy_hot = np.insert(enthalpy_hot, 0, 0) # The first value in enthalpy_hot is defined as 0
        # Shifting the cold enthalpy so that the first value starts at positive last_utility
        self.enthalpy_cold = np.insert(enthalpy_cold, 0, self.last_utility)
        print('The last utility is %g %s\n' % (self.last_utility, self.last_utility.units))

        # Getting heats above / below pinch for each stream
        streams_in_interval = np.zeros((len(self.streams), len(delta_plotted_ylines)), dtype = np.int8)
        streams_in_interval[self.hot_streams&self.active_streams, :] = streams_in_interval1
        streams_in_interval[~self.hot_streams&self.active_streams, :] = streams_in_interval2
        self._interval_heats = streams_in_interval * cp_vals * delta_plotted_ylines
        if self.first_utility_loc:
            q_above = np.sum(self._interval_heats[:, -1-self.first_utility_loc:], axis = 1)
            q_below = np.sum(self._interval_heats[:, :-1-self.first_utility_loc], axis = 1)
        else:
            q_below = np.sum(self._interval_heats, axis = 1)
            q_above = np.zeros_like(q_below)
        for idx, elem in enumerate(self.streams):
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
        #self._interval_heats = self._interval_heats[self.active_streams, :] # Removing inactive streams; convenient for place_exchangers()
        
        # Heat limits used in the frontend version of place_exchangers()
        self.upper_limit = np.ones((np.sum(self.hot_streams&self.active_streams) + len(self.hot_utilities), np.sum(~self.hot_streams&self.active_streams) + len(self.cold_utilities)), dtype = np.object) * -1
        self.lower_limit = np.zeros_like(self.upper_limit)
        self.required = np.zeros_like(self.upper_limit, dtype = np.bool)

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
        temperatures = np.empty( (len(self.streams) - np.sum(~self.active_streams), 2) )
        cp_vals = np.empty( (len(self.streams) - np.sum(~self.active_streams), 1) )
        x_tick_labels = np.empty(len(temperatures), dtype = 'object') # The names of the streams

        # We want the hot streams to come first on this plot
        hot_idx = 0
        cold_idx = -len(temperatures) + np.sum(self.hot_streams&self.active_streams)
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
        ax1.set_ylim(-0.5 - 0.5*(len(self._plotted_ylines)//10), len(self._plotted_ylines) - 0.5 + 0.5*(len(self._plotted_ylines)//10)) # len()//10 is a correction for HEN with many temperature intervals
        ax1.set_yticks([])
        q_above_text_loc = ax1.get_ylim()[1] - 0.01*(ax1.get_ylim()[1] - ax1.get_ylim()[0])
        q_below_text_loc = ax1.get_ylim()[0] + 0.04*(ax1.get_ylim()[1] - ax1.get_ylim()[0])
        cp_text_loc = ax1.get_ylim()[0] + 0.01*(ax1.get_ylim()[1] - ax1.get_ylim()[0])
        # Setting the hot streams area and the cold streams area. Streams should be equally spaced
        # Hot streams go from 0.01 to 0.49; cold streams go from 0.51 to 0.99
        hot_distance = 0.48 / (sum(self.hot_streams&self.active_streams) + 1)
        hot_idx = 1
        cold_distance = 0.48 / (sum(~self.hot_streams&self.active_streams) + 1)
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
            ax1.arrow(horizontal_loc, temperatures[idx, 0], 0, temperatures[idx, 1]-temperatures[idx, 0], color = my_color, length_includes_head = True,
                      linewidth = 0.005, head_width = 0.0064, head_length = (self._plotted_ylines[-1]-self._plotted_ylines[0])/1500)
            # Q and Fc_p values for each stream
            if show_properties:
                q_above_text = '$Q_{Top}$: %g' % (self.streams[x_tick_labels[idx]].q_above)
                ax1.text(horizontal_loc, q_above_text_loc, q_above_text, ha = 'center', va = 'top') # Heat above the pinch point
                q_below_text = '$Q_{Bot}$: %g' % (self.streams[x_tick_labels[idx]].q_below)
                ax1.text(horizontal_loc, q_below_text_loc, q_below_text, ha = 'center', va = 'bottom') # Heat below the pinch point
                cp_unit_plot = str(self.cp_unit * self.flow_unit).replace('delta_deg', '°', 1)
                cp_text = f'$Fc_p$: {cp_vals[idx][0]:g}'
                ax1.text(horizontal_loc, cp_text_loc, cp_text, ha = 'center', va = 'bottom') # F_cp value
        
        # Text with the units of Q and Fc_p in the center of the plot
        if show_properties:
                ax1.text(np.mean(ax1.get_xlim()), q_above_text_loc, f'Q unit: {self.heat_unit}', ha = 'center', va = 'top')
                ax1.text(np.mean(ax1.get_xlim()), q_below_text_loc, f'Q unit: {self.heat_unit}', ha = 'center', va = 'bottom')
                ax1.text(np.mean(ax1.get_xlim()), cp_text_loc, f'$Fc_p$ unit: {cp_unit_plot}', ha = 'center', va = 'bottom')
        
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
        
        # Using tight_layout() with frontend causes the title / axes ticks to be cropped, and w_pad / h_pad do not seem to help
        if tab_control is None:
            plt.tight_layout()
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
        hot_index = np.concatenate(([True], np.sum(self._interval_heats[self.hot_streams&self.active_streams], axis = 0, dtype = np.bool))) # First value in the hot scale is defined as 0, so it's always True
        cold_index = np.concatenate(([True], np.sum(self._interval_heats[~self.hot_streams&self.active_streams], axis = 0, dtype = np.bool))) # First value in the cold scale is defined as the cold utility, so it's always True
        # Hot line
        ax2.plot(np.cumsum(self.enthalpy_hot[hot_index][1:]), self._plotted_ylines[hot_index][1:], '-or', linewidth = 0.25, ms = 1.5)
        # Cold line
        ax2.plot(np.cumsum(self.enthalpy_cold[cold_index][:-1]), self._plotted_ylines[cold_index][:-1] - self.delta_t.value, '-ob', linewidth = 0.25, ms = 1.5)
        ax2.set_ylim(ax2.get_ylim()[0], ax2.get_ylim()[1] * 1.05) # Making the y-axis a little longer to avoid CC's overlapping with text

        # Arrow markers at the end of each line, rotated to follow the line
        ylim = ax2.get_ylim()
        xlim = ax2.get_xlim()
        # Hot arrow
        dx = self.enthalpy_hot[hot_index][0]-self.enthalpy_hot[hot_index][1]
        dy = self._plotted_ylines[hot_index][0]-self._plotted_ylines[hot_index][1]
        ax2.arrow(self.enthalpy_hot[hot_index][1], self._plotted_ylines[hot_index][1], dx, dy, color = 'r', length_includes_head = True, 
            linewidth = 0.25, head_width = (ylim[1]-ylim[0])*0.02, head_length = (xlim[1]-xlim[0])*0.01)
        # Cold arrow
        dx = np.cumsum(self.enthalpy_cold[cold_index])[-1] - np.cumsum(self.enthalpy_cold[cold_index])[-2]
        dy = self._plotted_ylines[cold_index][-1]-self._plotted_ylines[cold_index][-2]
        ax2.arrow(np.cumsum(self.enthalpy_cold[cold_index])[-2], self._plotted_ylines[cold_index][-2]-self.delta_t.value, dx, dy, color = 'b', length_includes_head = True,
            linewidth = 0.25, head_width = (ylim[1]-ylim[0])*0.02, head_length = (xlim[1]-xlim[0])*0.01)

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

        # Using tight_layout() with frontend causes the title / axes ticks to be cropped, and w_pad / h_pad do not seem to help
        if tab_control is None:
            plt.tight_layout()
        plt.show(block = False)
        if tab_control: # Embed into GUI
            generate_GUI_plot(fig2, tab_control, 'Composite Curve')

    def _place_exchangers(self, pinch, num_of_intervals, upper, lower, required, U = 100, U_unit = unyt.J/(unyt.s*unyt.m**2*unyt.delta_degC), exchanger_type = 'Fixed Head', called_by_GMS = False):
        """
        Equations come from C.A. Floudas, "Nonlinear and Mixed-Integer Optimization", p. 283, and were streamlined by me.
        self._interval_heats has each stream as its rows and each interval as its columns, such that the topmost interval is the rightmost column
        """

        # Starting GEKKO
        m = GEKKO(remote = False)

        # Q_exchanger is how much heat each exchanger will transfer
        # First N rows of Q_exchanger are the N hot utilities; first M columns are the M cold utilities
        Q_exchanger = np.zeros((np.sum(self.hot_streams&self.active_streams) + len(self.hot_utilities), 
                                np.sum(~self.hot_streams&self.active_streams) + len(self.cold_utilities)), dtype = np.object) # Hot streams and cold streams
        hot_stidx = 0 # Used to iterate over streams via self._interval_heats
        for rowidx in range(Q_exchanger.shape[0]):
            cold_stidx = 0
            for colidx in range(Q_exchanger.shape[1]):
                # No hot utilities are used below pinch
                if rowidx < len(self.hot_utilities) and (pinch == 'below' or self.first_utility == 0):
                    Q_exchanger[rowidx, colidx] = m.Const(0, f'Q_{rowidx}{colidx}')
                # No cold utilities are used above pinch
                elif colidx < len(self.cold_utilities) and (pinch == 'above' or self.last_utility == 0): 
                    Q_exchanger[rowidx, colidx] = m.Const(0, f'Q_{rowidx}{colidx}')
                # No matches between utilities
                elif rowidx < len(self.hot_utilities) and colidx < len(self.cold_utilities):
                    Q_exchanger[rowidx, colidx] = m.Const(0, f'Q_{rowidx}{colidx}')
                # Hot stream not present above pinch (cold was checked previously)
                elif rowidx >= len(self.hot_utilities) and pinch == 'above' and np.sum(self._interval_heats[self.hot_streams&self.active_streams][hot_stidx, -num_of_intervals:]) == 0:
                    Q_exchanger[rowidx, colidx] = m.Const(0, f'Q_{rowidx}{colidx}')
                # Hot stream not present below pinch (cold was checked previously)
                elif pinch == 'below' and np.sum(self._interval_heats[self.hot_streams&self.active_streams][hot_stidx, :num_of_intervals]) == 0:
                    Q_exchanger[rowidx, colidx] = m.Const(0, f'Q_{rowidx}{colidx}')
                elif upper[rowidx, colidx] == 0:
                    Q_exchanger[rowidx, colidx] = m.Const(0, f'Q_{rowidx}{colidx}')
                else: # Valid match
                    Q_exchanger[rowidx, colidx] = m.Var(0, lb = 0, name = f'Q_{rowidx}{colidx}')
                if colidx >= len(self.cold_utilities): # Increase cold stream counter iff colidx represents a stream and not a utility
                    cold_stidx += 1
            if rowidx >= len(self.hot_utilities): # Increase hot stream counter iff colidx represents a stream and not a utility
                hot_stidx += 1

        matches = np.zeros_like(Q_exchanger, dtype = np.object) # Whether there is a heat exchanger between two streams
        hot_stidx = 0 # Used to iterate over streams via self._interval_heats
        for rowidx in range(matches.shape[0]):
            cold_stidx = 0
            for colidx in range(matches.shape[1]):
                # No hot utilities are used below pinch
                if rowidx < len(self.hot_utilities) and (pinch == 'below' or self.first_utility == 0):
                    matches[rowidx, colidx] = m.Const(0, f'Y_{rowidx}{colidx}')
                    if pinch == 'above' and not called_by_GMS:
                        self.always_forbidden_above[rowidx, colidx] = True
                    elif pinch == 'below' and not called_by_GMS:
                        self.always_forbidden_below[rowidx, colidx] = True
                # No cold utilities are used above pinch
                elif colidx < len(self.cold_utilities) and pinch == 'above':
                    matches[rowidx, colidx] = m.Const(0, f'Y_{rowidx}{colidx}')
                    if pinch == 'above' and not called_by_GMS:
                        self.always_forbidden_above[rowidx, colidx] = True
                    elif pinch == 'below' and not called_by_GMS:
                        self.always_forbidden_below[rowidx, colidx] = True
                # No matches between utilities
                elif rowidx < len(self.hot_utilities) and colidx < len(self.cold_utilities):
                    matches[rowidx, colidx] = m.Const(0, f'Y_{rowidx}{colidx}')
                    if pinch == 'above' and not called_by_GMS:
                        self.always_forbidden_above[rowidx, colidx] = True
                    elif pinch == 'below' and not called_by_GMS:
                        self.always_forbidden_below[rowidx, colidx] = True
                # Hot utility and cold stream, but cold stream not present above pinch
                elif rowidx < len(self.hot_utilities) and (
                np.sum(self._interval_heats[~self.hot_streams&self.active_streams][cold_stidx, -num_of_intervals:]) == 0):
                    matches[rowidx, colidx] = m.Const(0, f'Y_{rowidx}{colidx}')
                    if pinch == 'above' and not called_by_GMS:
                        self.always_forbidden_above[rowidx, colidx] = True
                    elif pinch == 'below' and not called_by_GMS:
                        self.always_forbidden_below[rowidx, colidx] = True
                # Cold utility and hot stream, but hot stream not present below pinch
                elif colidx < len(self.cold_utilities) and (
                np.sum(self._interval_heats[self.hot_streams&self.active_streams][hot_stidx, :num_of_intervals]) == 0):
                    matches[rowidx, colidx] = m.Const(0, f'Y_{rowidx}{colidx}')
                    if pinch == 'above' and not called_by_GMS:
                        self.always_forbidden_above[rowidx, colidx] = True
                    elif pinch == 'below' and not called_by_GMS:
                        self.always_forbidden_below[rowidx, colidx] = True
                # Match between streams, but at least one isn't present above pinch
                elif rowidx >= len(self.hot_utilities) and pinch == 'above' and (
                np.sum(self._interval_heats[self.hot_streams&self.active_streams][hot_stidx, -num_of_intervals:]) == 0 or
                np.sum(self._interval_heats[~self.hot_streams&self.active_streams][cold_stidx, -num_of_intervals:]) == 0):
                    matches[rowidx, colidx] = m.Const(0, f'Y_{rowidx}{colidx}')
                    if pinch == 'above' and not called_by_GMS:
                        self.always_forbidden_above[rowidx, colidx] = True
                    elif pinch == 'below' and not called_by_GMS:
                        self.always_forbidden_below[rowidx, colidx] = True
                # Match between streams, but at least one isn't present below pinch
                elif colidx >= len(self.cold_utilities) and pinch == 'below' and (
                np.sum(self._interval_heats[self.hot_streams&self.active_streams][hot_stidx, :num_of_intervals]) == 0 or
                np.sum(self._interval_heats[~self.hot_streams&self.active_streams][cold_stidx, :num_of_intervals]) == 0):
                    matches[rowidx, colidx] = m.Const(0, f'Y_{rowidx}{colidx}')
                    if pinch == 'above' and not called_by_GMS:
                        self.always_forbidden_above[rowidx, colidx] = True
                    elif pinch == 'below' and not called_by_GMS:
                        self.always_forbidden_below[rowidx, colidx] = True
                elif upper[rowidx, colidx] == 0:
                    matches[rowidx, colidx] = m.Const(0, f'Y_{rowidx}{colidx}')
                elif required[rowidx, colidx]:
                    matches[rowidx, colidx] = m.Const(1, f'Y_{rowidx}{colidx}')
                    m.Equation(lower[rowidx, colidx]*matches[rowidx, colidx] <= Q_exchanger[rowidx, colidx])
                    m.Equation(upper[rowidx, colidx]*matches[rowidx, colidx] >= Q_exchanger[rowidx, colidx])
                else:
                    matches[rowidx, colidx] = m.Var(0, lb = 0, ub = 1, integer = True, name = f'Y_{rowidx}{colidx}')
                    m.Equation(lower[rowidx, colidx]*matches[rowidx, colidx] <= Q_exchanger[rowidx, colidx])
                    m.Equation(upper[rowidx, colidx]*matches[rowidx, colidx] >= Q_exchanger[rowidx, colidx])
                if colidx >= len(self.cold_utilities): # Increase cold stream counter iff colidx represents a stream and not a utility
                    cold_stidx += 1
            if rowidx >= len(self.hot_utilities): # Increase hot stream counter iff colidx represents a stream and not a utility
                hot_stidx += 1

        
        # Eqn 1
        for stidx, rowidx in enumerate(range(len(self.hot_utilities), Q_exchanger.shape[0])): # stidx bc self.hot_streams has only streams, but no utilities
            # Create an equation iff stream is present in subnetwork
            if pinch == 'above' and np.sum(self._interval_heats[self.hot_streams&self.active_streams][stidx, -num_of_intervals:]):
                m.Equation(m.sum(Q_exchanger[rowidx, :]) == m.sum(self._interval_heats[self.hot_streams&self.active_streams][stidx, -num_of_intervals:]) )
            elif pinch == 'below' and np.sum(self._interval_heats[self.hot_streams&self.active_streams][stidx, :num_of_intervals]):
                m.Equation(m.sum(Q_exchanger[rowidx, :]) == m.sum(self._interval_heats[self.hot_streams&self.active_streams][stidx, :num_of_intervals]) )

        # Eqn 2
        if pinch == 'above' and self.first_utility > 0:
            for rowidx in range(len(self.hot_utilities)):
                m.Equation(m.sum(Q_exchanger[rowidx, len(self.cold_utilities):]) == self.first_utility.value)

        # Eqn 3
        for stidx, colidx in enumerate(range(len(self.cold_utilities), Q_exchanger.shape[1])):
            # Create an equation iff stream is present in subnetwork
            if pinch == 'above' and np.sum(self._interval_heats[~self.hot_streams&self.active_streams][stidx, -num_of_intervals:]):
                m.Equation(m.sum(Q_exchanger[:, colidx]) == m.sum(self._interval_heats[~self.hot_streams&self.active_streams][stidx, -num_of_intervals:]) )
            elif pinch == 'below' and np.sum(self._interval_heats[~self.hot_streams&self.active_streams][stidx, :num_of_intervals]):
                m.Equation(m.sum(Q_exchanger[:, colidx]) == m.sum(self._interval_heats[~self.hot_streams&self.active_streams][stidx, :num_of_intervals]) )
        
        # Eqn 4
        if pinch == 'below' and self.last_utility > 0:
            for colidx in range(len(self.cold_utilities)):
                m.Equation(m.sum(Q_exchanger[len(self.hot_utilities):, colidx]) == self.last_utility.value)
              
        m.Minimize(m.sum(matches))
        m.options.IMODE = 3 # Steady-state optimization
        m.options.solver = 1 # APOPT solver
        m.options.csv_write = 2
        m.options.web = 0
        m.solver_options = ['minlp_maximum_iterations 750', \
                            # minlp iterations with integer solution
                            'minlp_max_iter_with_int_sol 750', \
                            # treat minlp as nlp
                            'minlp_as_nlp 0', \
                            # nlp sub-problem max iterations
                            'nlp_maximum_iterations 100', \
                            # 1 = depth first, 2 = breadth first
                            'minlp_branch_method 1', \
                            # maximum deviation from whole number
                            'minlp_integer_tol 0.0001', \
                            # covergence tolerance
                            'minlp_gap_tol 0.001']
        m.solve(disp = False)

        # Saving the results to variables (for ease of access)
        results = m.load_results()
        Q_tot_results = np.zeros_like(Q_exchanger, dtype = np.float64)
        costs = np.zeros_like(Q_tot_results)
        # Generating names to be used in a Pandas DataFrame with the results
        row_names = self.hot_utilities.index.append(self.streams.index[self.hot_streams&self.active_streams])
        col_names = self.cold_utilities.index.append(self.streams.index[~self.hot_streams&self.active_streams])
        Q_tot_results = pd.DataFrame(Q_tot_results, row_names, col_names)
        costs = pd.DataFrame(costs, row_names, col_names)
        U = U * U_unit
        # Populating the DataFrames with the results
        for rowidx in range(Q_exchanger.shape[0]):
            for colidx in range(Q_exchanger.shape[1]):
                # No matches between utilities --> Y, heats, and costs are always 0
                if rowidx < len(self.hot_utilities) and colidx < len(self.cold_utilities):
                    continue
            # Elements with nonzero values are stored as intermediates within Q_exc_tot
            # 2e-5 is rounding to prevent extremely small heats from being counted. Cutoff may need extra tuning
                elif 'LOWER' in dir(Q_exchanger[rowidx, colidx]) and Q_exchanger[rowidx, colidx][0] > 2e-5:
                    Q_tot_results.iat[rowidx, colidx] = Q_exchanger[rowidx, colidx][0]

                    # Obtaining ΔT values for ΔT_lm
                    # Hot utility and cold stream
                    if rowidx < len(self.hot_utilities):
                        delta_T1 = self.hot_utilities[row_names[rowidx]].temperature - self.streams[col_names[colidx]].t2
                        delta_T2 = self.hot_utilities[row_names[rowidx]].temperature - self.streams[col_names[colidx]].t1
                    # Hot stream and cold utility
                    elif colidx < len(self.cold_utilities):
                        delta_T1 = self.streams[row_names[rowidx]].t1 - self.cold_utilities[col_names[colidx]].temperature
                        delta_T2 = self.streams[row_names[rowidx]].t2 - self.cold_utilities[col_names[colidx]].temperature
                    # Hot stream and cold stream
                    else:
                        # Cold stream ends at a point higher than the hot stream begins, thus only part of the cold stream can be used
                        if self.streams[col_names[colidx]].t2 >= self.streams[row_names[rowidx]].t1 - self.delta_t:
                            delta_T1 = self.delta_t
                        else:
                            delta_T1 = self.streams[row_names[rowidx]].t1 - self.streams[col_names[colidx]].t2
                        # Cold stream begins at a point higher than the hot stream ends, thus only part of the hot stream can be used
                        if self.streams[col_names[colidx]].t1 >= self.streams[row_names[rowidx]].t2 - self.delta_t:
                            delta_T2 = self.delta_t
                        else:
                            delta_T2 = self.streams[row_names[rowidx]].t2 - self.streams[col_names[colidx]].t1

                    if delta_T1 == delta_T2:
                        delta_T_lm = delta_T1.value * self.delta_temp_unit
                    else:
                        delta_T_lm = (delta_T1.value - delta_T2.value) / (np.log(delta_T1.value/delta_T2.value)) * self.delta_temp_unit
                    area = Q_exchanger[rowidx, colidx][0] * self.heat_unit / (U * delta_T_lm)
                    Ac = area.to('ft**2').value

                    # All costs for Ac < 150 came from linear regressions for the appropriate cost eqn with 200 <= Ac <= 1000
                    if exchanger_type == 'Floating Head':
                        if Ac > 150:
                            costs.iat[rowidx, colidx] = np.exp(12.0310) * Ac**(-0.8709) * np.exp(0.09005*np.log(Ac)**2)
                        else:
                            costs.iat[rowidx, colidx] = 11.76656116273358*Ac + 18378.519876797985
                    elif exchanger_type == 'Fixed Head':
                        if Ac > 150:
                            costs.iat[rowidx, colidx] = np.exp(11.4185) * Ac**(-0.9228) * np.exp(0.09861*np.log(Ac)**2)
                        else:
                            costs.iat[rowidx, colidx] = 7.875782676947923*Ac + 9308.899770148431
                    elif exchanger_type == 'U-Tube':
                        if Ac > 150:
                            costs.iat[rowidx, colidx] = np.exp(11.5510) * Ac**(-0.9186) * np.exp(0.09790*np.log(Ac)**2)
                        else:
                            costs.iat[rowidx, colidx] = 8.838982418325454*Ac + 10684.794389373843
                    elif exchanger_type == 'Kettle Vaporizer':
                        if Ac > 150:
                            costs.iat[rowidx, colidx] = np.exp(12.3310) * Ac**(-0.8709) * np.exp(0.09005*np.log(Ac)**2)
                        else:
                            costs.iat[rowidx, colidx] = 15.883196220397641*Ac + 24808.40692590638
                    
        Q_tot_results = np.round(Q_tot_results, 5) # Ignoring very small decimals that cause integers to be non-integers
        costs = np.round(costs, 2) # Money needs only 2 decimals
        results = pd.concat((Q_tot_results, costs), keys = ['Q', 'cost'])
        m.cleanup() # Deletes temp files generated by GEKKO
        return results
    
    def solve_HEN(self, pinch, depth = 0, upper = None, lower = None, required = None, U = 100, U_unit = unyt.J/(unyt.s*unyt.m**2*unyt.delta_degC), exchanger_type = 'Fixed Head'):
        """
        The main function used by ALChemE to automatically place exchangers
        """
        # Required matches
        if required is None:
            required = np.zeros((np.sum(self.hot_streams&self.active_streams) + len(self.hot_utilities), np.sum(~self.hot_streams&self.active_streams) + len(self.cold_utilities)), dtype = np.bool)
        elif required.shape != (np.sum(self.hot_streams&self.active_streams) + len(self.hot_utilities), np.sum(~self.hot_streams&self.active_streams) + len(self.cold_utilities)):
            raise ValueError('Required must be a %dx%d matrix' % (np.sum(self.hot_streams&self.active_streams) + len(self.hot_utilities), np.sum(~self.hot_streams&self.active_streams) + len(self.cold_utilities)))
        
        # Restricting the number of intervals based on the pinch point
        if pinch.casefold() in {'above', 'top', 'up'}:
            if self.first_utility_loc == 0:
                if self.GUI_terminal is not None:
                    self.GUI_terminal.print2screen('ERROR: This HEN doesn\'t have anything above the pinch', True)
                raise ValueError('This HEN doesn\'t have anything above the pinch')
            else:
                num_of_intervals = self.first_utility_loc + 1
            self.always_forbidden_above = np.zeros_like(required) # Used in the get_more_sols area of this function
            pinch = 'above'
        elif pinch.casefold() in {'below', 'bottom', 'bot', 'down'}:
            if self.first_utility_loc == 0: # No pinch point --> take all intervals
                num_of_intervals = self._interval_heats[self.active_streams].shape[-1]
            else:
                num_of_intervals = self._interval_heats[self.active_streams, :-self.first_utility_loc-1].shape[-1]
            self.always_forbidden_below = np.zeros_like(required) # Used in the get_more_sols area of this function
            pinch = 'below'
        
        # Setting the upper heat exchanged limit for each pair of streams
        if upper is None: # Automatically set the upper limits
            upper = np.zeros_like(required, dtype = np.float64)
            upper = self._get_maximum_heats(pinch)
        elif isinstance(upper, (int, float)): # A single value was passed, representing a maximum threshold
            temp_upper = upper
            upper = np.zeros_like(required, dtype = np.float64)
            upper = self._get_maximum_heats(pinch)
            upper[upper > temp_upper] = temp_upper # Setting the given upper limit only for streams that naturally had a higher limit
        elif upper.shape != required.shape: # An array-like was passed, but it has the wrong shape
            raise ValueError('Upper must be a %dx%d matrix' % (required.shape[0], required.shape[1]))
        else:
            temp_upper = self._get_maximum_heats(pinch)
            upper[(upper == -1) | (upper > temp_upper)] = temp_upper[(upper == -1) | (upper > temp_upper)]
        # Setting the lower heat exchanged limit for each pair of streams
        if lower is None:
            lower = np.zeros_like(required, dtype = np.float64)
        elif isinstance(lower, (int, float)): # A single value was passed, representing a minimum threshold
            if np.sum(lower > upper):
                if self.GUI_terminal is not None:
                    self.GUI_terminal.print2screen(f'The lower threshold you passed is greater than the maximum heat of {np.sum(lower > upper)} streams\n', True)
                raise ValueError(f'The lower threshold you passed is greater than the maximum heat of {np.sum(lower > upper)} streams')
            lower = np.ones_like(required, dtype = np.float64) * lower
        elif upper.shape != required.shape: # An array-like was passed, but it has the wrong shape
            raise ValueError('Lower must be a %dx%d matrix' % (required.shape[0], required.shape[1]))
        
        # Updating the HEN_object variables (used in the get_more_sols area of this function)
        self.upper_limit = upper
        self.lower_limit = lower
        self.required = required

        t1 = time()
        results = self._place_exchangers(pinch, num_of_intervals, upper, lower, required, U, U_unit, exchanger_type)
        t2 = time()
        print(f'The first solution took {t2-t1:.2f} seconds')
        if self.GUI_terminal is not None:
            self.GUI_terminal.print2screen(f'The first solution took {t2-t1:.2f} seconds\n', False)
        # Storing the results in the HEN object
        if pinch == 'above':
            self.results_above = [results]
        elif pinch == 'below':
            self.results_below = [results]
        
        if depth == 0: # No further iterations to get more solutions
            return results
        
        ###### get_more_solutions area ######
        # Setting up the individual locations
        final_combinations = []
        self._failed_depth_one = set()
        # Removing elements that are always forbidden, forbidden by the user, or required by the user, so we don't waste time combining them
        for elem in product(range(self.required.shape[0]), range(self.required.shape[1]) ):
            if pinch == 'above' and not (self.always_forbidden_above[elem] or self.required[elem] or self.upper_limit[elem] == 0):
                final_combinations.append(elem)
            elif pinch == 'below' and not (self.always_forbidden_below[elem] or self.required[elem] or self.upper_limit[elem] == 0):
                final_combinations.append(elem)
        
        ##### Case depth == 1 #####
        print('Current depth = 1')
        total_iterations = len(final_combinations)
        Parallel(n_jobs = -1, require = 'sharedmem')(delayed(self._get_more_sols_depth_one)(
            pinch, upper, lower, required, U, U_unit, exchanger_type, num_of_intervals, total_iterations, iter_count, elem) for iter_count, elem in enumerate(final_combinations))
        
        # Setup for depth > 1
        new_final_combinations = []
        for elem in final_combinations:
            if elem not in self._failed_depth_one:
                new_final_combinations.append(elem)

        ##### Cases depth > 1 #####
        for cur_depth in range(2, depth+1):
            print(f'\nCurrent depth = {cur_depth}')
            depth_combinations = list(combinations(new_final_combinations, cur_depth))
            total_iterations = len(depth_combinations)
            Parallel(n_jobs = -1, require = 'sharedmem')(delayed(self._get_more_sols)(
                pinch, upper, lower, required, U, U_unit, exchanger_type, num_of_intervals, total_iterations, iter_count, elem) for iter_count, elem in enumerate(depth_combinations))
            
    def _get_more_sols_depth_one(self, pinch, upper, lower, required, U, U_unit, exchanger_type, num_of_intervals, total_iterations, iter_count, elem):
        """
        Auxiliary function of solve_HEN().
        Is called when cur_depth == 1 (even if depth > 1), and shouldn't be called by the user.
        Needed due to joblib's parallelization.
        """
        # Setting up local variables to prevent changes
        local_upper = upper.copy()
        local_required = required.copy()


        if pinch == 'above':
            # Original solution had a match here, so we'll try forbidding it
            if self.results_above[0].loc['Q'].iat[elem] > 0:
                local_upper[elem] = 0
            # Original solution didn't have a match here, so we'll try requiring it
            elif self.results_above[0].loc['Q'].iat[elem] == 0:
                local_required[elem] = True
                
            print(f'Iteration {iter_count + 1} out of {total_iterations}')
            try:
                unique_sol = True
                results = self._place_exchangers(pinch, num_of_intervals, local_upper, lower, local_required, U, U_unit, exchanger_type, called_by_GMS = True)
                for prev_sol in self.results_above:
                    if np.allclose(prev_sol.loc['Q'], results.loc['Q'], 0, 1e-6): # Using a 1e-6 absolute tolerance to compare heats
                        unique_sol = False
                        break
                if unique_sol:
                    self.results_above.append(results)
                    print(f'Found a unique solution during iteration {iter_count + 1}. Solution has a cost of ${results.loc["cost"].sum().sum():,.2f}')
            except Exception:
                self._failed_depth_one.add(elem)

        elif pinch == 'below':
            # Original solution had a match here, so we'll try forbidding it
            if self.results_below[0].loc['Q'].iat[elem] > 0:
                local_upper[elem] = 0
            # Original solution didn't have a match here, so we'll try requiring it
            elif self.results_below[0].loc['Q'].iat[elem] == 0:
                local_required[elem] = True
            
            print(f'Iteration {iter_count + 1} out of {total_iterations}')
            try:
                unique_sol = True
                results = self._place_exchangers(pinch, num_of_intervals, local_upper, lower, local_required, U, U_unit, exchanger_type, called_by_GMS = True)
                for prev_sol in self.results_below:
                    if np.allclose(prev_sol.loc['Q'], results.loc['Q'], 0, 1e-6): # Using a 1e-6 absolute tolerance to compare heats
                        unique_sol = False
                        break
                if unique_sol:
                    self.results_below.append(results)
                    print(f'Found a unique solution during iteration {iter_count + 1}. Solution has a cost of ${results.loc["cost"].sum().sum():,.2f}')
            except Exception:
                self._failed_depth_one.add(elem)
    
    def _get_more_sols(self, pinch, upper, lower, required, U, U_unit, exchanger_type, num_of_intervals, total_iterations, iter_count, elem):
        """
        Auxiliary function of solve_HEN().
        Is called when cur_depth > 1, and shouldn't be called by the user.
        Needed due to joblib's parallelization.
        """
        # Setting up local variables to prevent changes
        local_upper = upper.copy()
        local_required = required.copy()

        if pinch == 'above':
            # Elem is a container of indices, so we'll iterate for each index
            for depth_elem in elem:
                # Original solution had a match here, so we'll try forbidding it
                if self.results_above[0].loc['Q'].iat[depth_elem] > 0:
                    local_upper[depth_elem] = 0
                # Original solution didn't have a match here, so we'll try requiring it
                elif self.results_above[0].loc['Q'].iat[depth_elem] == 0:
                    local_required[depth_elem] = True
                
            print(f'Iteration {iter_count + 1} out of {total_iterations}')
            try:
                unique_sol = True
                results = self._place_exchangers(pinch, num_of_intervals, local_upper, lower, local_required, U, U_unit, exchanger_type, called_by_GMS = True)
                for prev_sol in self.results_above:
                    if np.allclose(prev_sol.loc['Q'], results.loc['Q'], 0, 1e-6): # Using a 1e-6 absolute tolerance to compare heats
                        unique_sol = False
                        break
                if unique_sol:
                    self.results_above.append(results)
                    print(f'Found a unique solution during iteration {iter_count + 1}. Solution has a cost of ${results.loc["cost"].sum().sum():,.2f}')
            except Exception:
                pass
        
        elif pinch == 'below':
            # Elem is a container of indices, so we'll iterate for each index
            for depth_elem in elem:
                # Original solution had a match here, so we'll try forbidding it
                if self.results_below[0].loc['Q'].iat[depth_elem] > 0:
                    local_upper[depth_elem] = 0
                # Original solution didn't have a match here, so we'll try requiring it
                elif self.results_below[0].loc['Q'].iat[depth_elem] == 0:
                    local_required[depth_elem] = True
                
            print(f'Iteration {iter_count + 1} out of {total_iterations}')
            try:
                unique_sol = True
                results = self._place_exchangers(pinch, num_of_intervals, local_upper, lower, local_required, U, U_unit, exchanger_type, called_by_GMS = True)
                for prev_sol in self.results_below:
                    if np.allclose(prev_sol.loc['Q'], results.loc['Q'], 0, 1e-6): # Using a 1e-6 absolute tolerance to compare heats
                        unique_sol = False
                        continue
                if unique_sol:
                    self.results_below.append(results)
                    print(f'Found a unique solution during iteration {iter_count + 1}. Solution has a cost of ${results.loc["cost"].sum().sum():,.2f}')
            except Exception:
                pass

    def add_exchanger(self, stream1, stream2, heat = 'auto', ref_stream = 1, exchanger_delta_t = None, pinch = 'above', exchanger_name = None, U = 100, U_unit = unyt.J/(unyt.s*unyt.m**2*unyt.delta_degC), 
        exchanger_type = 'Fixed Head', cost_a = 0, cost_b = 0, pressure = 0, pressure_unit = unyt.Pa, GUI_oe_tree = None):

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
                if self.GUI_terminal is not None:
                    self.GUI_terminal.print2screen(f'ERROR: Match is thermodynamically impossible, as the hot stream is entering with a temperature of {self.streams[stream1].current_t_above:.4g}, '
                    f'while the cold stream is leaving with a temperature of {s2_t_above:.4g}\n', True)
                raise ValueError(f'Match is thermodynamically impossible, as the hot stream is entering with a temperature of {self.streams[stream1].current_t_above:.4g}, '
                    f'while the cold stream is leaving with a temperature of {s2_t_above:.4g}')
            elif s1_t_above < self.streams[stream2].current_t_above:
                if self.GUI_terminal is not None:
                    self.GUI_terminal.print2screen(f'ERROR: Match is thermodynamically impossible, as the hot stream is leaving with a temperature of {s1_t_above:.4g}, '
                    f'while the cold stream is entering with a temperature of {self.streams[stream2].current_t_above:.4g}\n', True)
                raise ValueError(f'Match is thermodynamically impossible, as the hot stream is leaving with a temperature of {s1_t_above:.4g}, '
                    f'while the cold stream is entering with a temperature of {self.streams[stream2].current_t_above:.4g}')
            elif delta_T1 < self.delta_t:
                if self.GUI_terminal is not None:
                    self.GUI_terminal.print2screen(f'WARNING: match violates minimum ΔT, which equals {self.delta_t:.4g}\nThis match\'s ΔT is {delta_T1:.4g}\n', False)
                warnings.warn(f"Warning: match violates minimum ΔT, which equals {self.delta_t:.4g}\nThis match's ΔT is {delta_T1:.4g}")
            elif delta_T2 < self.delta_t:
                if self.GUI_terminal is not None:
                    self.GUI_terminal.print2screen(f"WARNING: match violates minimum ΔT, which equals {self.delta_t:.4g}\nThis match's ΔT is {delta_T2:.4g}\n", False)
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
                if self.GUI_terminal is not None:
                    self.GUI_terminal.print2screen(f'Match is thermodynamically impossible, as the hot stream is entering with a temperature of {self.streams[stream1].current_t_below:.4g}, '
                    f'while the cold stream is leaving with a temperature of {s2_t_below:.4g}\n', True)
                raise ValueError(f'Match is thermodynamically impossible, as the hot stream is entering with a temperature of {self.streams[stream1].current_t_below:.4g}, '
                    f'while the cold stream is leaving with a temperature of {s2_t_below:.4g}')
            elif s1_t_below < self.streams[stream2].current_t_below:
                if self.GUI_terminal is not None:
                    self.GUI_terminal.print2screen(f'Match is thermodynamically impossible, as the hot stream is leaving with a temperature of {s1_t_below:.4g}, '
                    f'while the cold stream is entering with a temperature of {self.streams[stream2].current_t_below:.4g}\n', True)
                raise ValueError(f'Match is thermodynamically impossible, as the hot stream is leaving with a temperature of {s1_t_below:.4g}, '
                    f'while the cold stream is entering with a temperature of {self.streams[stream2].current_t_below:.4g}')
            elif delta_T1 < self.delta_t:
                if self.GUI_terminal is not None:
                    self.GUI_terminal.print2screen(f"WARNING: match violates minimum ΔT, which equals {self.delta_t:.4g}\nThis match's ΔT is {delta_T1:.4g}\n", False)
                print(f"Warning: match violates minimum ΔT, which equals {self.delta_t:.4g}\nThis match's ΔT is {delta_T1:.4g}")
            elif delta_T2 < self.delta_t:
                if self.GUI_terminal is not None:
                    self.GUI_terminal.print2screen(f"WARNING: match violates minimum ΔT, which equals {self.delta_t:.4g}\nThis match's ΔT is {delta_T2:.4g}\n", False)
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
        
        if GUI_oe_tree is not None:
            oeDataVector = [exchanger_name, stream1, stream2, heat, self.exchangers[exchanger_name].cost_fob, 'Active']
            print(oeDataVector)
            GUI_oe_tree.receive_new_exchanger(oeDataVector)
        
    def save(self, name, overwrite = False):
        # Ensuring the saved file has an extension
        if '.' not in name:
            name += '.p'

        # Manipulating the file name if overwrite is False
        if not overwrite and os.path.exists(name):
            while os.path.exists(name):
                word = name.split('.')
                name = word[0] + '_DUPLICATE.' +  word[1]
            print(f'The file name you chose already exists in this directory. Saving as {name} instead')

        with open(name, 'wb') as f:
            pickle.dump(self, f)
        
    @classmethod
    def load(cls, name = None):
        # Automatically finding a .p file. Works if there is only one .p file in the working directory
        if name is None:
            files = os.listdir()
            file_list = []
            for myfile in name:
                if myfile.endswith('.p'):
                    file_list.append(myfile)
            if len(file_list) != 1:
                raise ValueError('You must supply a file name (with extension) to HEN.load()\n'+
                                 'Alternatively, ensure there\'s only one .p file in the working directory')
            else:
                name = file_list[0]
        
        with open(name, 'rb') as f:
            return pickle.load(f)
    
    def _get_maximum_heats(self, pinch):
        """
        Auxiliary function to calculate the maximum heat transferable between two streams.
        Shouldn't be called by the user; rather, it is automatically called by solve_HEN().
        """
        # num_of_intervals setup
        if pinch.casefold() == 'above':
            if self.first_utility_loc == 0:
                if self.GUI_terminal is not None:
                    self.GUI_terminal.print2screen('ERROR: This HEN doesn\'t have anything above the pinch', True)
                raise ValueError('This HEN doesn\'t have anything above the pinch')
            else:
                num_of_intervals = self.first_utility_loc + 1
        elif pinch.casefold() == 'below':
            if self.first_utility_loc == 0: # No pinch point --> take all intervals
                num_of_intervals = self._interval_heats[self.active_streams].shape[-1]
            else:
                num_of_intervals = self._interval_heats[self.active_streams, :-self.first_utility_loc-1].shape[-1]
        
        upper = np.zeros((np.sum(self.hot_streams&self.active_streams) + len(self.hot_utilities), np.sum(~self.hot_streams&self.active_streams) + len(self.cold_utilities)), dtype = np.object)
        
        # Getting the maximum heats
        for rowidx in range(upper.shape[0]):
            for colidx in range(upper.shape[1]):
                if rowidx < len(self.hot_utilities) and pinch == 'below': # No hot utilities are used below pinch
                    upper[rowidx, colidx] = 0
                elif colidx < len(self.cold_utilities) and pinch == 'above': # No cold utilities are used above pinch
                    upper[rowidx, colidx] = 0
                elif rowidx < len(self.hot_utilities) and colidx < len(self.cold_utilities): # No matches between utilities
                    upper[rowidx, colidx] = 0
                elif rowidx < len(self.hot_utilities): # Match between hot utility and cold stream
                    temp_idx = colidx - len(self.cold_utilities)
                    if pinch == 'above':
                        upper[rowidx, colidx] = np.min((self.first_utility, np.sum(self._interval_heats[~self.hot_streams&self.active_streams][temp_idx, -num_of_intervals:]) ))
                    else:
                        upper[rowidx, colidx] = np.min((self.first_utility, np.sum(self._interval_heats[~self.hot_streams&self.active_streams][temp_idx, :num_of_intervals]) ))
                elif colidx < len(self.cold_utilities): # Match between hot stream and cold utility
                    temp_idx = rowidx - len(self.hot_utilities)
                    if pinch == 'above':
                        upper[rowidx, colidx] = np.min((np.sum(self._interval_heats[self.hot_streams&self.active_streams][temp_idx, -num_of_intervals:]), self.last_utility))
                    else:
                        upper[rowidx, colidx] = np.min((np.sum(self._interval_heats[self.hot_streams&self.active_streams][temp_idx, :num_of_intervals]), self.last_utility))
                else: # Match between two streams
                    temp_idx1 = rowidx - len(self.hot_utilities)
                    temp_idx2 = colidx - len(self.cold_utilities)
                    if pinch == 'above':
                        lowest_cold = (self._interval_heats[~self.hot_streams&self.active_streams][temp_idx2, -num_of_intervals:] != 0).argmax()
                        upper[rowidx, colidx] = np.min((np.sum(self._interval_heats[self.hot_streams&self.active_streams][temp_idx1, -num_of_intervals+lowest_cold:]), 
                                                        np.sum(self._interval_heats[~self.hot_streams&self.active_streams][temp_idx2, -num_of_intervals:]) ))
                    else:
                        lowest_cold = (self._interval_heats[~self.hot_streams&self.active_streams][temp_idx2, :num_of_intervals] != 0).argmax()
                        upper[rowidx, colidx] = np.min((np.sum(self._interval_heats[self.hot_streams&self.active_streams][temp_idx1, lowest_cold:num_of_intervals]), 
                                                        np.sum(self._interval_heats[~self.hot_streams&self.active_streams][temp_idx2, :num_of_intervals]) ))
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
            self.stream_type = 'Hot'
        else: # Cold stream
            self.current_t_above = None
            self.current_t_below = self.t1
            self.stream_type = 'Cold'
    
    def __repr__(self):
        if self.t1 > self.t2:
            stream_type = 'Hot'
        else:
            stream_type = 'Cold'
        text =(f'{stream_type} stream with T_in = {self.t1} and T_out = {self.t2}\n'
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
