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

class WReN:
    """
    A class that holds processes, used to solve WReN problems
    """
    def __init__(self, conc_unit = unyt.mg/unyt.kg, flow_unit = unyt.kg/unyt.s):
        self.conc_unit = conc_unit
        self.flow_unit = flow_unit
        self.processes = pd.Series()
        self.active_processes = np.array([], dtype = np.bool)
    
    def add_process(self, sink_conc, source_conc, sink_flow, source_flow = None, process_name = None, conc_unit = None, flow_unit = None, conc_names = None, GUI_oe_tree = None):
        # Converting to default units (as declared via the WReN class)        
        if conc_unit is None:
            if isinstance(sink_conc, (list, tuple, pd.Series, np.ndarray)):
                for idx in range(len(sink_conc)):
                    sink_conc[idx] *= self.conc_unit
                    source_conc[idx] *= self.conc_unit
            else:
                sink_conc *= self.conc_unit
                source_conc *= self.conc_unit
        else:
            if isinstance(sink_conc, (list, tuple, pd.Series, np.ndarray)):
                for idx in range(len(sink_conc)):
                    sink_conc[idx] *= conc_unit
                    sink_conc[idx] = sink_conc[idx].to(self.conc_unit)
                    source_conc[idx] *= conc_unit
                    source_conc[idx] = source_conc[idx].to(self.conc_unit)
            else:
                sink_conc *= conc_unit
                sink_conc = sink_conc.to(self.conc_unit)
                source_conc *= conc_unit
                source_conc = source_conc.to(self.conc_unit)
        
        if source_flow is None:
            source_flow = sink_flow

        if flow_unit is None:
            sink_flow *= self.flow_unit
            source_flow *= self.flow_unit
        else:
            sink_flow *= flow_unit
            sink_flow = sink_flow.to(self.flow_unit)
            source_flow *= flow_unit
            source_flow = source_flow.to(self.flow_unit)
        
        if process_name is None:
            idx = 1
            while f'P{idx}' in self.processes.keys():
                idx += 1
            process_name = f'P{idx}'
        
        # Transforming concentrations from a container to a Pandas DataFrame
        # A single contaminant is of the unyt.array.unyt_quantity type
        if isinstance(sink_conc, (list, tuple, np.ndarray)) and not isinstance(sink_conc, unyt.array.unyt_quantity):
            sink_conc = pd.Series(sink_conc, index = conc_names)
            source_conc = pd.Series(source_conc, index = conc_names)

        # Generating the process object and adding it to the WReN object
        temp = pd.Series(Process(sink_conc, source_conc, sink_flow, source_flow), [process_name])
        self.processes = pd.concat([self.processes, temp])
        self.active_processes = np.append(self.active_processes, True)

        if GUI_oe_tree is not None:
            temp_diff = t2 - t1
            temp_diff = temp_diff.tolist() * self.delta_temp_unit
            oeDataVector = [stream_name, t1, t2, cp*flow_rate, cp*flow_rate*temp_diff]
            print(oeDataVector)
            GUI_oe_tree.receive_new_stream(oeDataVector)

    def activate_process(self, processes_to_change):
        if isinstance(processes_to_change, str): # Only one process name was passed
            if not self.processes[processes_to_change].active:
                self.processes[processes_to_change].active = True
                loc = self.processes.index.get_loc(processes_to_change)
                self.active_processes[loc] = True
            else:
                raise ValueError(f'Process {processes_to_change} is already inactive')
        elif isinstance(processes_to_change, (list, tuple, set)): # A container of process names was passed
            for elem in processes_to_change:
                if not self.processes[elem].active:
                    self.processes[elem].active = True
                    loc = self.processes.index.get_loc(elem)
                    self.active_processes[loc] = True
                else:
                    warnings.warn(f'Process {elem} is already inactive. Ignoring this input and continuing')
        else:
            raise TypeError('The processes_to_change parameter should be a string or list/tuple/set of strings')
    
    def deactivate_process(self, processes_to_change):
        if isinstance(processes_to_change, str): # Only one process name was passed
            if self.processes[processes_to_change].active:
                self.processes[processes_to_change].active = False
                loc = self.processes.index.get_loc(processes_to_change)
                self.active_process[loc] = False
            else:
                raise ValueError(f'Process {processes_to_change} is already active')
        elif isinstance(processes_to_change, (list, tuple, set)): # A container of process names was passed
            for elem in processes_to_change:
                if self.processes[elem].active:
                    self.processes[elem].active = False
                    loc = self.processes.index.get_loc(elem)
                    self.active_processes[loc] = False
                else:
                    warnings.warn(f'Process {elem} is already active. Ignoring this input and continuing')
        else:
            raise TypeError('The processes_to_change parameter should be a string or list/tuple/set of strings')          
    
    def delete(self, obj_to_del):
        if obj_to_del in self.processes:
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
        self.upper_limit = np.zeros((np.sum(self.hot_streams&self.active_streams) + len(self.hot_utilities), np.sum(~self.hot_streams&self.active_streams) + len(self.cold_utilities)), dtype = np.object)
        self.lower_limit = np.zeros_like(self.upper_limit)
        self.forbidden = np.zeros_like(self.upper_limit, dtype = np.bool)
        self.required = np.zeros_like(self.forbidden)

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

        plt.show(block = False)
        if tab_control: # Embed into GUI
            generate_GUI_plot(fig2, tab_control, 'Composite Curve')

        """ TODO: remove whitespace around the graphs
        ax = gca;
        ti = ax.TightInset;
        ax.Position = [ti(1), ti(2), 1 - ti(1) - ti(3), 1 - ti(2) - ti(4)]; % Removing whitespace from the graph
        """

    def place_exchangers(self, pinch, upper = None, lower = None, forbidden = None, required = None, U = 100, U_unit = unyt.J/(unyt.s*unyt.m**2*unyt.delta_degC), exchanger_type = 'Fixed Head', called_by_GMS = False):
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
                num_of_intervals = self.first_utility_loc + 1
            pinch = 'above'
        elif pinch.casefold() in {'below', 'bottom', 'bot', 'down'}:
            if self.first_utility_loc == 0: # No pinch point --> take all intervals
                num_of_intervals = self._interval_heats[self.active_streams].shape[-1]
            else:
                num_of_intervals = self._interval_heats[self.active_streams, :-self.first_utility_loc-1].shape[-1]
            pinch = 'below'

        # Starting GEKKO
        m = GEKKO(remote = False)

        # Forbidden and required matches
        if forbidden is None:
            forbidden = np.zeros((np.sum(self.hot_streams&self.active_streams) + len(self.hot_utilities), np.sum(~self.hot_streams&self.active_streams) + len(self.cold_utilities)), dtype = np.bool)
        elif forbidden.shape != (np.sum(self.hot_streams&self.active_streams) + len(self.hot_utilities), np.sum(~self.hot_streams&self.active_streams) + len(self.cold_utilities)):
            raise ValueError('Forbidden must be a %dx%d matrix' % (np.sum(self.hot_streams&self.active_streams) + len(self.hot_utilities), np.sum(~self.hot_streams&self.active_streams) + len(self.cold_utilities)))
        if required is None:
            required = np.zeros_like(forbidden)
        elif required.shape != forbidden.shape:
            raise ValueError('Required must be a %dx%d matrix' % (forbidden.shape[0], forbidden.shape[1]))

        # Auto-forbidden matches (for the get_more_solutions() function)
        if pinch == 'above' and not called_by_GMS:
            self.always_forbidden_above = np.zeros_like(forbidden)
        elif pinch == 'below' and not called_by_GMS:
            self.always_forbidden_below = np.zeros_like(forbidden)
        
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
        # Setting the lower heat exchanged limit for each pair of streams
        if lower is None:
            lower = np.zeros_like(forbidden, dtype = np.float64)
        elif isinstance(lower, (int, float)): # A single value was passed, representing a minimum threshold
            if np.sum(lower > upper):
                raise ValueError(f'The lower threshold you passed is greater than the maximum heat of {np.sum(lower > upper)} streams')
            lower = np.ones_like(forbidden, dtype = np.float64) * lower
        elif upper.shape != forbidden.shape: # An array-like was passed, but it has the wrong shape
            raise ValueError('Lower must be a %dx%d matrix' % (forbidden.shape[0], forbidden.shape[1]))
        
        # Updating the HEN_object variables (used in get_more_sols() )
        if not called_by_GMS:
            self.upper_limit = upper
            self.lower_limit = lower
            self.forbidden = forbidden
            self.required = required

        # First N rows of residuals are the N hot utilities
        # The extra interval represents the heats coming in from "above the highest interval" (always 0)
        residuals = np.zeros((np.sum(self.hot_streams&self.active_streams) + len(self.hot_utilities), num_of_intervals + 1), dtype = np.object)
        hot_stidx = 0 # Used to iterate over streams via self._interval_heats
        for rowidx in range(residuals.shape[0]):
            # All R_0 and R_K must equal 0
            residuals[rowidx, 0] = m.Const(0, f'R_{rowidx}0') 
            residuals[rowidx, residuals.shape[1]-1] = m.Const(0, f'R_{rowidx}{residuals.shape[1]-1}')
            for intervalidx in range(1, residuals.shape[1] - 1):
                # No hot utilities are used below pinch
                if rowidx < len(self.hot_utilities) and (pinch == 'below' or self.first_utility == 0):
                    residuals[rowidx, intervalidx] = m.Const(0, f'R_{rowidx}{intervalidx}')
                # Hot stream not present above pinch
                elif rowidx >= len(self.hot_utilities) and pinch == 'above' and np.sum(self._interval_heats[self.hot_streams&self.active_streams][hot_stidx, -num_of_intervals:]) == 0:
                    residuals[rowidx, intervalidx] = m.Const(0, f'R_{rowidx}{intervalidx}')
                # Hot stream not present below pinch
                elif pinch == 'below' and np.sum(self._interval_heats[self.hot_streams&self.active_streams][hot_stidx, :num_of_intervals]) == 0:
                    residuals[rowidx, intervalidx] = m.Const(0, f'R_{rowidx}{intervalidx}')
                else:
                    residuals[rowidx, intervalidx] = m.Var(0, lb = 0, name = f'R_{rowidx}{intervalidx}')
            if rowidx >= len(self.hot_utilities): # Increase hot stream counter iff colidx represents a stream and not a utility
                hot_stidx += 1
        residuals = np.fliplr(residuals) # R_X0 is above the highest interval, R_X1 is the highest interval, and so on

        # Q_exchanger is how much heat each exchanger will transfer
        # First N rows of Q_exchanger are the N hot utilities; first M columns are the M cold utilities
        Q_exchanger = np.zeros((np.sum(self.hot_streams&self.active_streams) + len(self.hot_utilities), 
            np.sum(~self.hot_streams&self.active_streams) + len(self.cold_utilities), num_of_intervals ), dtype = np.object) # Hot streams, cold streams, and intervals
        Q_exc_tot = np.zeros((Q_exchanger.shape[:2]), dtype = np.object)
        hot_stidx = 0 # Used to iterate over streams via self._interval_heats
        for rowidx in range(Q_exchanger.shape[0]):
            cold_stidx = 0
            for colidx in range(Q_exchanger.shape[1]):
                has_variable = 0 # Used to generate Q_exchanger_tot intermediates
                for intervalidx in range(Q_exchanger.shape[2]):
                    # No hot utilities are used below pinch
                    if rowidx < len(self.hot_utilities) and (pinch == 'below' or self.first_utility == 0):
                        Q_exchanger[rowidx, colidx, intervalidx] = m.Const(0, f'Q_{rowidx}{colidx}{intervalidx+1}')
                    # No cold utilities are used above pinch
                    elif colidx < len(self.cold_utilities) and (pinch == 'above' or self.last_utility == 0): 
                        Q_exchanger[rowidx, colidx, intervalidx] = m.Const(0, f'Q_{rowidx}{colidx}{intervalidx+1}')
                    # No matches between utilities
                    elif rowidx < len(self.hot_utilities) and colidx < len(self.cold_utilities):
                        Q_exchanger[rowidx, colidx, intervalidx] = m.Const(0, f'Q_{rowidx}{colidx}{intervalidx+1}')
                    # Cold stream not present in this interval above pinch. Intervals are counted in reverse bc Q_exchanger is reversed later
                    elif pinch == 'above' and self._interval_heats[~self.hot_streams&self.active_streams][cold_stidx, -intervalidx-1] == 0: #self._interval_heats[~self.hot_streams&self.active_streams][cold_stidx, -num_of_intervals:][-intervalidx-1]
                        Q_exchanger[rowidx, colidx, intervalidx] = m.Const(0, f'Q_{rowidx}{colidx}{intervalidx+1}')
                    # Cold stream not present in this interval below pinch. Intervals are counted in reverse bc Q_exchanger is reversed later
                    elif colidx >= len(self.cold_utilities) and pinch == 'below' and self._interval_heats[~self.hot_streams&self.active_streams][cold_stidx, :num_of_intervals][-intervalidx-1] == 0:
                        Q_exchanger[rowidx, colidx, intervalidx] = m.Const(0, f'Q_{rowidx}{colidx}{intervalidx+1}')
                    # Hot stream not present above pinch (cold was checked previously)
                    elif rowidx >= len(self.hot_utilities) and pinch == 'above' and np.sum(self._interval_heats[self.hot_streams&self.active_streams][hot_stidx, -num_of_intervals:]) == 0:
                        Q_exchanger[rowidx, colidx, intervalidx] = m.Const(0, f'Q_{rowidx}{colidx}{intervalidx+1}')
                    # Hot stream not present below pinch (cold was checked previously)
                    elif pinch == 'below' and np.sum(self._interval_heats[self.hot_streams&self.active_streams][hot_stidx, :num_of_intervals]) == 0:
                        Q_exchanger[rowidx, colidx, intervalidx] = m.Const(0, f'Q_{rowidx}{colidx}{intervalidx+1}')
                    elif forbidden[rowidx, colidx]:
                        Q_exchanger[rowidx, colidx, intervalidx] = m.Const(0, f'Q_{rowidx}{colidx}{intervalidx+1}')
                    else: # Valid match
                        Q_exchanger[rowidx, colidx, intervalidx] = m.Var(0, lb = 0, name = f'Q_{rowidx}{colidx}{intervalidx+1}')
                        has_variable += 1
                if has_variable:
                    Q_exc_tot[rowidx, colidx] = m.Intermediate(m.sum(Q_exchanger[rowidx, colidx, :]), f'Q_tot_{rowidx}{colidx}')
                if colidx >= len(self.cold_utilities): # Increase cold stream counter iff colidx represents a stream and not a utility
                    cold_stidx += 1
            if rowidx >= len(self.hot_utilities): # Increase hot stream counter iff colidx represents a stream and not a utility
                hot_stidx += 1
        Q_exchanger = Q_exchanger[:, :, ::-1] #Q_XX1 is the highest interval, Q_XX2 is the 2nd highest, and so on

        matches = np.zeros_like(Q_exc_tot, dtype = np.object) # Whether there is a heat exchanger between two streams
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
                elif rowidx < len(self.hot_utilities) and pinch == 'above' and (
                np.sum(self._interval_heats[~self.hot_streams&self.active_streams][cold_stidx, -num_of_intervals:]) == 0):
                    matches[rowidx, colidx] = m.Const(0, f'Y_{rowidx}{colidx}')
                    if pinch == 'above' and not called_by_GMS:
                        self.always_forbidden_above[rowidx, colidx] = True
                    elif pinch == 'below' and not called_by_GMS:
                        self.always_forbidden_below[rowidx, colidx] = True
                # Cold utility and hot stream, but hot stream not present below pinch
                elif colidx < len(self.cold_utilities) and pinch == 'below' and (
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
                elif forbidden[rowidx, colidx]:
                    matches[rowidx, colidx] = m.Const(0, f'Y_{rowidx}{colidx}')
                elif required[rowidx, colidx]:
                    matches[rowidx, colidx] = m.Const(1, f'Y_{rowidx}{colidx}')
                    m.Equation(lower[rowidx, colidx] <= Q_exc_tot[rowidx, colidx])
                    m.Equation(upper[rowidx, colidx] >= Q_exc_tot[rowidx, colidx])
                else:
                    matches[rowidx, colidx] = m.Var(0, lb = 0, ub = 1, integer = True, name = f'Y_{rowidx}{colidx}')
                    m.Equation(lower[rowidx, colidx]*matches[rowidx, colidx] <= Q_exc_tot[rowidx, colidx])
                    m.Equation(upper[rowidx, colidx]*matches[rowidx, colidx] >= Q_exc_tot[rowidx, colidx])
                if colidx >= len(self.cold_utilities): # Increase cold stream counter iff colidx represents a stream and not a utility
                    cold_stidx += 1
            if rowidx >= len(self.hot_utilities): # Increase hot stream counter iff colidx represents a stream and not a utility
                hot_stidx += 1

        
        # Eqn 1
        for stidx, rowidx in enumerate(range(len(self.hot_utilities), Q_exchanger.shape[0])): # stidx bc self.hot_streams has only streams, but no utilities
            for intervalidx in range(Q_exchanger.shape[2]):
                if pinch == 'above' and np.sum(self._interval_heats[self.hot_streams&self.active_streams][stidx, -num_of_intervals:]) != 0: # Create an equation iff stream is present in subnetwork
                    m.Equation(residuals[rowidx, intervalidx] - residuals[rowidx, intervalidx+1] +
                        m.sum(Q_exchanger[rowidx, :, intervalidx]) == self._interval_heats[self.hot_streams&self.active_streams][stidx, -num_of_intervals+intervalidx])
                elif pinch == 'below' and np.sum(self._interval_heats[self.hot_streams&self.active_streams][stidx, :num_of_intervals]) != 0:
                    m.Equation(residuals[rowidx, intervalidx] - residuals[rowidx, intervalidx+1] +
                        m.sum(Q_exchanger[rowidx, :, intervalidx]) == self._interval_heats[self.hot_streams&self.active_streams][stidx, intervalidx])

        # Eqn 2
        if pinch == 'above' and self.first_utility > 0:
            for rowidx in range(len(self.hot_utilities)):
                m.Equation(m.sum(Q_exchanger[rowidx, len(self.cold_utilities):, :]) == self.first_utility.value)

        # Eqn 3
        for stidx, colidx in enumerate(range(len(self.cold_utilities), Q_exchanger.shape[1])):
            for intervalidx in range(Q_exchanger.shape[2]):
                if pinch == 'above' and self._interval_heats[~self.hot_streams&self.active_streams][stidx, -num_of_intervals+intervalidx] != 0: # Create an equation iff stream is present in interval
                    m.Equation(m.sum(Q_exchanger[:, colidx, intervalidx]) == self._interval_heats[~self.hot_streams&self.active_streams][stidx, -num_of_intervals+intervalidx])
                elif pinch == 'below' and self._interval_heats[~self.hot_streams&self.active_streams][stidx, intervalidx] != 0:
                    m.Equation(m.sum(Q_exchanger[:, colidx, intervalidx]) == self._interval_heats[~self.hot_streams&self.active_streams][stidx, intervalidx])
        
        # Eqn 4
        if pinch == 'below' and self.last_utility > 0:
            for colidx in range(len(self.cold_utilities)):
                m.Equation(m.sum(Q_exchanger[len(self.hot_utilities):, colidx, :]) == self.last_utility.value)
              
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
        # Hiding the convergence info when this function is called by get_more_solutions()
        if called_by_GMS:
            disp = False
        else:
            disp = True
        m.solve(disp = disp)

        # Saving the results to variables (for ease of access)
        results = m.load_results()
        Y_results = np.zeros_like(matches, dtype = np.int8)
        Q_tot_results = np.zeros_like(Q_exc_tot, dtype = np.float)
        costs = np.zeros_like(Q_tot_results)
        # Generating names to be used in a Pandas DataFrame with the results
        row_names = self.hot_utilities.index.append(self.streams.index[self.hot_streams&self.active_streams])
        col_names = self.cold_utilities.index.append(self.streams.index[~self.hot_streams&self.active_streams])
        Y_results = pd.DataFrame(Y_results, row_names, col_names)
        Q_tot_results = pd.DataFrame(Q_tot_results, row_names, col_names)
        costs = pd.DataFrame(costs, row_names, col_names)
        U = U * U_unit
        # Populating the DataFrames with the results
        for rowidx in range(Q_exc_tot.shape[0]):
            for colidx in range(Q_exc_tot.shape[1]):
                # No matches between utilities --> Y, heats, and costs are always 0
                if rowidx < len(self.hot_utilities) and colidx < len(self.cold_utilities):
                    continue
                # Elements with nonzero values are stored as intermediates within Q_exc_tot
                # 1e-6 is rounding to prevent extremely small heats from being counted. Cutoff may need extra tuning
                elif not isinstance(Q_exc_tot[rowidx, colidx], (int, float)) and Q_exc_tot[rowidx, colidx][0] > 1e-6:
                    Q_tot_results.iat[rowidx, colidx] = Q_exc_tot[rowidx, colidx][0]
                    Y_results.iat[rowidx, colidx] = 1

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
                        # Cold stream ends at a point higher than the hot stream begins, thus only part of the hot stream can be used
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
                    area = Q_exc_tot[rowidx, colidx][0] * self.heat_unit / (U * delta_T_lm)
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
                    
        costs = np.round(costs, 2) # Money needs only 2 decimals
        # Storing the results in the HEN object
        if pinch == 'above':
            if 'Q_tot_above' not in dir(self):
                self.Q_tot_above = np.array((None, Q_tot_results)) # Having a None before or after the DF just makes things work
                self.exchanger_costs_above = np.array((None, costs))
        else:
            if 'Q_tot_below' not in dir(self):
                self.Q_tot_below = np.array((None, Q_tot_results)) # Having a None before or after the DF just makes things work
                self.exchanger_costs_below = np.array((None, costs))

        return Y_results, Q_tot_results, costs

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
        
        # 
        if GUI_oe_tree is not None:
            oeDataVector = [exchanger_name, stream1, stream2, heat, self.exchangers[exchanger_name].cost_fob, 'Active']
            print(oeDataVector)
            GUI_oe_tree.receive_new_exchanger(oeDataVector)
        
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
    def load(cls, file = None):
        if file == None:
            files = os.listdir()
            file_list = []
            for myfile in files:
                if myfile.endswith('.p'):
                    file_list.append(myfile)
            if len(file_list) != 1:
                raise ValueError('You must supply a file name (with extension) to HEN.load()\n'+
            'Alternatively, ensure there\'s only one .p file in the working directory')
            else:
                file = file_list[0]
        return pickle.load(open(file, 'rb'))
    
class Process():
    def __init__(self, sink_conc, source_conc, sink_flow, source_flow):
        self.sink_conc = sink_conc
        self.source_conc = source_conc
        self.sink_flow = sink_flow
        self.source_flow = source_flow
        self.active = True
    
    def __repr__(self):
        # Formatting the concentrations
        if isinstance(self.sink_conc, pd.Series):
            print_sink_conc = []
            print_source_conc = []
            for name in self.sink_conc.index:
                print_sink_conc.append(f'{name}: {self.sink_conc[name]}')
                print_source_conc.append(f'{name}: {self.source_conc[name]}')
        else:
            print_sink_conc = self.sink_conc
            print_source_conc = self.source_conc
        
        text =(f'A process with sink concentration = {print_sink_conc} and source concentration = {print_source_conc}\n'
             f'sink flow = {self.sink_flow} and source flow = {self.source_flow}\n')
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
