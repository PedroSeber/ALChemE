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
import pdb

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
        
        # Transforming concentrations from a container to a Pandas Series
        # A single contaminant is of the unyt.array.unyt_quantity type
        if not isinstance(sink_conc, pd.Series):
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
                self.active_processes[loc] = False
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
            loc = self.processes.index.get_loc(obj_to_del)
            self.active_processes = np.delete(self.active_processess, loc)
            del self.processes[obj_to_del]
        else:
            raise ValueError(f'{obj_to_del} not found in the processes')

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

    def solve_WReN(self, costs, upper = None, lower = None, forbidden = None, required = None):
        """
        The main function used by ALChemE to automatically set flowrates among processes, freshwater, and wastewater
        """
        # Forbidden and required matches
        if forbidden is None:
            # +1 on the rows represents "from freshwater", +1 on the columns represents "to wastewater"
            forbidden = np.zeros((len(self.processes[self.active_processes]) + 1, len(self.processes[self.active_processes]) + 1), dtype = bool)
        elif forbidden.shape != (len(self.processes[self.active_processes]) + 1, len(self.processes[self.active_processes]) + 1):
            raise ValueError('Forbidden must be a %dx%d matrix' % (len(self.processes[self.active_processes]) + 1, len(self.processes[self.active_processes]) + 1))
        if required is None:
            required = np.zeros_like(forbidden)
        elif required.shape != forbidden.shape:
            raise ValueError('Required must be a %dx%d matrix' % (forbidden.shape[0], forbidden.shape[1]))
        
        # Setting the upper heat exchanged limit for each pair of streams
        if upper is None: # Automatically set the upper limits
            upper = np.zeros_like(forbidden, dtype = np.float64)
            upper = self._get_maximum_flows(upper)
        elif isinstance(upper, (int, float)): # A single value was passed, representing a maximum threshold
            temp_upper = upper
            upper = np.zeros_like(forbidden, dtype = np.float64)
            upper = self._get_maximum_flows(upper)
            upper[upper > temp_upper] = temp_upper # Setting the given upper limit only for streams that naturally had a higher limit
        elif upper.shape != forbidden.shape: # An array-like was passed, but it has the wrong shape
            raise ValueError('Upper must be a %dx%d matrix' % (forbidden.shape[0], forbidden.shape[1]))
        # Setting the lower heat exchanged limit for each pair of streams
        if lower is None:
            lower = np.zeros_like(forbidden, dtype = np.float64)
        elif isinstance(lower, (int, float)): # A single value was passed, representing a minimum threshold
            if np.sum(lower > upper):
                if self.GUI_terminal is not None:
                    self.GUI_terminal.print2screen(f'The lower threshold you passed is greater than the maximum heat of {np.sum(lower > upper)} streams\n', True)
                raise ValueError(f'The lower threshold you passed is greater than the maximum heat of {np.sum(lower > upper)} streams')
            lower = np.ones_like(forbidden, dtype = np.float64) * lower
        elif upper.shape != forbidden.shape: # An array-like was passed, but it has the wrong shape
            raise ValueError('Lower must be a %dx%d matrix' % (forbidden.shape[0], forbidden.shape[1]))
        
        # Starting GEKKO
        m = GEKKO(remote = False)

        flows = np.zeros_like(forbidden, dtype = np.object)
        for rowidx in range(flows.shape[0]):
            for colidx in range(flows.shape[1]):
                # No freshwater goes to wastewater
                if rowidx == 0 and colidx == 0:
                    flows[rowidx, colidx] = m.Const(0, 'FW to WW')
                # Match from freshwater to a process, but one of the sink concentrations is 0. Thus, the entire sink flow must be supplied with freshwater
                elif rowidx == 0 and np.any(self.processes[self.active_processes].iat[colidx - 1].sink_conc == 0):
                    # Float is needed to convert an array of a single value to a non-array value. Array[0] does not work
                    flows[rowidx, colidx] = m.Const(float(self.processes[self.active_processes].iat[colidx - 1].sink_flow.value), f'FW to {colidx}')
                # Match from freshwater to a process
                elif rowidx == 0:
                    flows[rowidx, colidx] = m.Var(0, lb = lower[rowidx, colidx], ub = upper[rowidx, colidx], name = f'FW to {colidx}')
                # Match from a process to wastewater
                elif colidx == 0:
                    flows[rowidx, colidx] = m.Var(0, lb = lower[rowidx, colidx], ub = upper[rowidx, colidx], name = f'W{rowidx} to WW')
                # Match between two processes
                else:
                    flows[rowidx, colidx] = m.Var(0, lb = lower[rowidx, colidx], ub = upper[rowidx, colidx], name = f'W{rowidx} to {colidx}')
        
        for colidx in range(1, flows.shape[1]):
            # Eqn 1: Total mass balance for sinks
            m.Equation(m.sum(flows[:, colidx]) == self.processes[self.active_processes].iat[colidx - 1].sink_flow.value)
            # Eqn 2: For each contaminant, contaminant mass balance for sinks
            for contam in self.processes[self.active_processes].iat[0].sink_conc.index:
                conc_mat = [myprocess.source_conc[contam].value for myprocess in self.processes[self.active_processes]]
                m.Equation(m.sum(flows[1:, colidx] * conc_mat) <= self.processes[self.active_processes].iat[colidx - 1].sink_flow.value * self.processes[self.active_processes].iat[colidx - 1].sink_conc[contam].value)
        
        for rowidx in range(1, flows.shape[0]):
            # Eqn 3: Total mass balance for sources
            m.Equation(m.sum(flows[rowidx, :]) == self.processes[self.active_processes].iat[rowidx - 1].source_flow.value)
            # Eqn 4: For each contaminant, contaminant mass balance for sources
            #for contam in self.processes[self.active_processes].iat[0].sink_conc.index:
            #    conc_mat = [myprocess.source_conc[contam].value for myprocess in self.processes[self.active_processes]]
            #    m.Equation(m.sum(flows[1:, colidx] * conc_mat) == self.processes[self.active_processes].iat[colidx - 1].sink_flow.value)
        
        m.Minimize(m.sum(flows * costs))
        m.options.IMODE = 3 # Steady-state optimization
        m.options.solver = 1 # APOPT solver
        m.options.csv_write = 2
        m.options.web = 0
        m.solver_options = [# nlp sub-problem max iterations
                            'nlp_maximum_iterations 10000', \
                            # 1 = depth first, 2 = breadth first
                            'minlp_branch_method 1', \
                            # maximum deviation from whole number
                            'minlp_integer_tol 0.0001', \
                            # covergence tolerance
                            'minlp_gap_tol 0.001']
        m.solve(disp = True)

        # Saving the results to variables (for ease of access)
        results = m.load_results()
        flow_results = np.zeros_like(flows, dtype = np.float64)
        #cost_results = np.zeros_like(flow_results) # Do flow_results * costs instead
        # Generating names to be used in a Pandas DataFrame with the results
        row_names = ['FW']
        row_names.extend(self.processes.index[self.active_processes])
        col_names = ['WW']
        col_names.extend(self.processes.index[self.active_processes])
        flow_results = pd.DataFrame(flow_results, row_names, col_names)
        #cost_results = pd.DataFrame(costs, row_names, col_names)
        for rowidx in range(flows.shape[0]):
            for colidx in range(flows.shape[1]):
                # No matches between freshwater and wastewater --> Flows and costs are always 0
                if rowidx == 0 and colidx == 0:
                    continue
                # I'd need to import gekko to do proper isinstance() comparisons
                # 1e-5 is rounding to prevent extremely small heats from being counted. Cutoff may need extra tuning
                elif 'GK_Operators' in str(type(flows[rowidx, colidx])) and flows[rowidx, colidx].VALUE > 1e-5:
                    flow_results.iat[rowidx, colidx] = flows[rowidx, colidx].value.value
                elif 'GKVariable' in str(type(flows[rowidx, colidx])) and flows[rowidx, colidx][0] > 1e-5:
                    flow_results.iat[rowidx, colidx] = flows[rowidx, colidx][0]
        cost_results = flow_results * costs
        cost_results = np.round(cost_results, 2) # Money needs only 2 decimals
        flow_results = np.round(flow_results, 5) # Avoids large number of unnecessary significant digits
        self.results = pd.concat((flow_results, cost_results), keys = ['flows', 'cost'])
        

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
    def load(cls, file = None):
        if file is None:
            files = os.listdir()
            file_list = []
            for myfile in files:
                if myfile.endswith('.p'):
                    file_list.append(myfile)
            if len(file_list) != 1:
                raise ValueError('You must supply a file name (with extension) to WReN.load()\n'+
                                 'Alternatively, ensure there\'s only one .p file in the working directory')
            else:
                file = file_list[0]
        return pickle.load(open(file, 'rb'))
    
    def _get_maximum_flows(self, upper):
        """
        Auxiliary function to calculate the maximum flow rate transferable between two streams.
        Shouldn't be called by the user; rather, it is automatically called by solve_WReN().
        """
        for rowidx in range(upper.shape[0]):
            for colidx in range(upper.shape[1]):
                # No freshwater goes to wastewater
                if rowidx == 0 and colidx == 0:
                    upper[rowidx, colidx] = 0
                # Match from freshwater to a process
                elif rowidx == 0:
                    upper[rowidx, colidx] = self.processes[self.active_processes].iat[colidx - 1].sink_flow.value
                # Match from a process to wastewater
                elif colidx == 0:
                    upper[rowidx, colidx] = self.processes[self.active_processes].iat[rowidx - 1].source_flow.value
                # Match between two processes
                else:
                    upper[rowidx, colidx] = np.min((self.processes[self.active_processes].iat[rowidx - 1].source_flow.value, 
                                                    self.processes[self.active_processes].iat[colidx - 1].sink_flow.value ))
        return upper
    
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
