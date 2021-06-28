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
# Used in the WReN.save() and WReN.load() functions
import os
import pickle
# Used in the WReN.solve_WReN() functions
from gekko import GEKKO
import mystic as my
import scipy.optimize as sco
import pdb

class WReN:
    """
    A class that holds processes, used to solve WReN problems
    """
    def __init__(self, conc_unit = unyt.mg/unyt.kg, flow_unit = unyt.kg/unyt.s, GUI_terminal = None):
        self.conc_unit = conc_unit
        self.flow_unit = flow_unit
        self.processes = pd.Series()
        self.active_processes = np.array([], dtype = np.bool)
        self.GUI_terminal = GUI_terminal
    
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
        # Note: calling index at the same time the pd.Series is generated causes the units to be stripped 
        if not isinstance(sink_conc, pd.Series):
            sink_conc = pd.Series(sink_conc)
            source_conc = pd.Series(source_conc)
            if conc_names is not None:
                sink_conc.index = conc_names
                source_conc.index = conc_names

        # Generating the process object and adding it to the WReN object
        temp = pd.Series(Process(sink_conc, source_conc, sink_flow, source_flow), [process_name])
        self.processes = pd.concat([self.processes, temp])
        self.active_processes = np.append(self.active_processes, True)

        # Appending to the cost matrix, or creating it if the current process is the first
            # Other matrices exist to help the frontend
        if 'costs' in dir(self):
            col_names = ['WW']
            col_names.extend(self.processes.index[self.active_processes])
            temp = pd.DataFrame(np.zeros((1, len(self.processes) + 1)), index = [process_name], columns = col_names)
            self.costs = self.costs.append(temp)
            self.costs.iloc[:, -1] = 0.0
            self.upper = self.upper.append(temp)
            self.upper.iloc[:, -1] = 0.0
            self.lower = self.lower.append(temp)
            self.lower.iloc[:, -1] = 0.0
        else:
            self.costs = pd.DataFrame(np.zeros((2, 2)), index = ['FW', process_name], columns = ['WW', process_name])
            self.upper = pd.DataFrame(np.zeros((2, 2)), index = ['FW', process_name], columns = ['WW', process_name])
            self.lower = pd.DataFrame(np.zeros((2, 2)), index = ['FW', process_name], columns = ['WW', process_name])

        if GUI_oe_tree is not None:
            # Obtain vectors of sink/source concentrations
            sinkConcVec = []
            sourceConcVec = []
            for name in self.processes[process_name].sink_conc.index:
                sinkConcVec.append(f'{name}: {self.processes[process_name].sink_conc[name]}')
                sourceConcVec.append(f'{name}: {self.processes[process_name].source_conc[name]}')
            # Send to object explorer
            oeDataVector = [process_name, sinkConcVec, sourceConcVec, sink_flow, source_flow]
            GUI_oe_tree.receive_new_process(oeDataVector)

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
            self.active_processes = np.delete(self.active_processes, loc)
            # Removing the process from the cost matrix
            self.costs.drop(index = obj_to_del, inplace = True)
            self.costs.drop(columns = obj_to_del, inplace = True)
            del self.processes[obj_to_del]
        else:
            raise ValueError(f'{obj_to_del} not found in the processes')

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
        
        # GEKKO does not like DataFrames
        if isinstance(costs, pd.DataFrame):
            costs = np.array(costs)
        
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
        
        m.Minimize(m.sum(flows**(0.6) * costs))
        m.options.IMODE = 3 # Steady-state optimization
        m.options.solver = 1 # APOPT solver
        m.options.csv_write = 2
        m.options.web = 0
        m.solver_options = [# nlp sub-problem max iterations
                            'nlp_maximum_iterations 1000', \
                            # 1 = depth first, 2 = breadth first
                            'minlp_branch_method 1', \
                            # maximum deviation from whole number
                            'minlp_integer_tol 0.0001', \
                            # covergence tolerance
                            'minlp_gap_tol 0.001']
        m.solve(disp = False)

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
        cost_results = flow_results**(0.6) * costs
        cost_results = np.round(cost_results, 2) # Money needs only 2 decimals
        flow_results = np.round(flow_results, 5) # Avoids large number of unnecessary significant digits
        self.results = pd.concat((flow_results, cost_results), keys = ['flows', 'cost'])
        

    def solve_WReN_scipy(self, water_costs, upper = None, lower = None):
        """
        The main function used by ALChemE to automatically set flowrates among processes, freshwater, and wastewater
        """
        # Cutoff used to ignore very small flows. All elements < 1e-"cutoff_power" become 0
        cutoff_power = 5
        cutoff_tol = float(f'1e-{cutoff_power}')
        
        # Setting the flow limit for each pair of processes
        if upper is None or not np.any(upper): # Automatically set the upper limits
            upper = np.zeros((len(self.processes[self.active_processes]) + 1, len(self.processes[self.active_processes]) + 1) )
            upper = self._get_maximum_flows(upper)
        elif isinstance(upper, (int, float)): # A single value was passed, representing a maximum threshold
            temp_upper = upper
            upper = np.zeros((len(self.processes[self.active_processes]) + 1, len(self.processes[self.active_processes]) + 1) )
            upper = self._get_maximum_flows(upper)
            upper[upper > temp_upper] = temp_upper # Setting the given upper limit only for streams that naturally had a higher limit
        elif upper.shape != (len(self.processes[self.active_processes]) + 1, len(self.processes[self.active_processes]) + 1): # An array-like was passed, but it has the wrong shape
            raise ValueError('Upper must be a %dx%d matrix' % (len(self.processes[self.active_processes]) + 1, len(self.processes[self.active_processes]) + 1))
        elif isinstance(upper, pd.DataFrame):
            upper = upper.values
        
        # Setting the lower flow limit for each pair of processes
        if lower is None:
            lower = np.zeros_like(upper, dtype = np.float64)
        elif isinstance(lower, (int, float)): # A single value was passed, representing a minimum threshold
            if np.sum(lower > upper):
                if self.GUI_terminal is not None:
                    self.GUI_terminal.print2screen(f'The lower threshold you passed is greater than the maximum heat of {np.sum(lower > upper)} streams\n', True)
                raise ValueError(f'The lower threshold you passed is greater than the maximum heat of {np.sum(lower > upper)} streams')
            lower = np.ones_like(upper, dtype = np.float64) * lower
        elif lower.shape != upper.shape: # An array-like was passed, but it has the wrong shape
            raise ValueError('Lower must be a %dx%d matrix' % (upper.shape[0], upper.shape[1]))
        elif isinstance(lower, pd.DataFrame):
            lower = lower.values
        
        water_costs = water_costs.values.flatten()[1:]

        bounds = []
        for colidx in range(upper.shape[1]):
            for rowidx in range(upper.shape[0]):
                if not upper[rowidx, colidx]:
                    # SLSQP returns an error if the bounds are (0, 0)
                    bounds.append((0, cutoff_tol * 0.9))
                else:
                    bounds.append((lower[rowidx, colidx], upper[rowidx, colidx]))
        # Removing the very first element (FW to WW), as it is always 0
        bounds = bounds[1:]

        cons_len = len(self.processes[self.active_processes]) + 1 # +1 for FW coming into the process
        cons = []
        # Eqn 1: Total mass balance for sinks
        sink_flow = []
        for process in self.processes[self.active_processes]:
            sink_flow.append(process.sink_flow.value)
        for idx in range(cons_len-1): # Goes until cons_len-1 because flows to wastewater are unconstrained. Compensated with the idx+1 below.
            temp_array = np.zeros(cons_len**2 - 1, dtype = np.int8)
            temp_array[(idx+1)*cons_len - 1 : (idx+1)*cons_len + cons_len - 1] = 1
            cons.append(sco.LinearConstraint(temp_array, sink_flow[idx], sink_flow[idx]) )

        # Eqn 2: For each contaminant, contaminant mass balance for sinks
        for contam_idx, contam in enumerate(self.processes[self.active_processes].iat[0].sink_conc.index):
            sink_conc = [proc.sink_conc[contam].value for proc in self.processes[self.active_processes]]
            source_conc = [proc.source_conc[contam].value for proc in self.processes[self.active_processes]]
            for idx in range(cons_len-1): # Goes until cons_len-1 because flows to wastewater are unconstrained. Compensated with the idx+1 below.
                temp_array = np.zeros(cons_len**2 - 1)
                temp_array[(idx+1)*cons_len : (idx+1)*cons_len + cons_len - 1] = source_conc
                cons.append(sco.LinearConstraint(temp_array, 0, sink_flow[idx]*sink_conc[idx]) )
        
        # Eqn 3: Total mass balance for sources
        source_flow = []
        for process in self.processes[self.active_processes]:
            source_flow.append(process.source_flow.value)
        for idx in range(cons_len-1): # Goes until cons_len-1 because flows from freshwater are unconstrained
            temp_array = np.zeros(cons_len**2 - 1, dtype = np.int8)
            temp_array[idx :: cons_len] = 1
            cons.append(sco.LinearConstraint(temp_array, source_flow[idx], source_flow[idx]) )

        def objective(x, water_costs):
            return (x**0.6).dot(water_costs)
        cost_fun = lambda x: objective(x, water_costs)
        
        # Running the optimizer multiple times with different x0 values to increase the chance of reaching the global minimum
        rng = np.random.default_rng()
        rng_upper = np.array([elem[1] for elem in bounds])
        rng_lower = np.array([elem[0] for elem in bounds])
        success_counter = 0
        for _ in range(2000):
            x0 = rng.random(len(bounds)) * (rng_upper-rng_lower) + rng_lower

            options = {'maxiter': 2000, 'eps': 1e-10, 'ftol': 1e-10}
            # I could not get the global solver to converge with these settings. These settings led to a runtime of ~ 500 secs on my computer.
            # mysol = sco.differential_evolution(cost_fun, bounds = bounds, maxiter = 2000, popsize = 40, init = 'sobol', mutation = 1.99, constraints = cons)
            mysol = sco.minimize(cost_fun, x0, method = 'SLSQP', bounds = bounds, constraints = cons, options = options)

            if 'best_objective' not in locals() or mysol['fun'] < best_objective:
                my_x = [0]
                my_x.extend(mysol['x'])
                my_x = np.array(my_x).reshape(cons_len, cons_len).T

                # Asserting the result is valid
                # Eqn 1: Total mass balance for sinks
                if not np.allclose(np.sum(my_x, axis = 0)[1:], sink_flow, cutoff_tol, cutoff_tol):
                    continue
                
                # Eqn 2: For each contaminant, contaminant mass balance for sinks
                for contam_idx in range(len(self.processes.iat[0].sink_conc)):
                    source_conc = np.array([proc.source_conc.iloc[contam_idx].value for proc in self.processes[self.active_processes]])
                    sink_conc = np.array([proc.sink_conc.iloc[contam_idx].value for proc in self.processes[self.active_processes]])
                    # @ means matrix multiplication. Using @ works, while using * and a np.sum() does not
                    greater = source_conc@my_x[1:, 1:] > sink_flow*sink_conc
                    if np.any(greater) and not np.allclose( (source_conc@my_x[1:, 1:])[greater], (sink_flow*sink_conc)[greater], cutoff_tol, cutoff_tol):
                        break

                # Runs only if for loop above was not broken
                else:
                    # Eqn 3: Total mass balance for sources
                    if not np.allclose(np.sum(my_x[1:], axis = 1), source_flow, cutoff_tol, cutoff_tol):
                        continue

                    best_objective = mysol['fun']
                    # Generating the result DataFrames
                    flow_results = np.zeros_like(my_x, dtype = np.float64)
                    # Names of the rows and columns
                    row_names = ['FW']
                    row_names.extend(self.processes.index[self.active_processes])
                    col_names = ['WW']
                    col_names.extend(self.processes.index[self.active_processes])
                    flow_results = pd.DataFrame(my_x, row_names, col_names)
                    cost_results = flow_results**(0.6) * self.costs
                    cost_results = np.round(cost_results, 2) # Money needs only 2 decimals
                    flow_results = np.round(flow_results, cutoff_power) # Avoids large number of unnecessary significant digits
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
        if isinstance(upper, pd.DataFrame):
            upper = upper.values
        
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
