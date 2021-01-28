#################################################################################################################
# SECTION 0 - IMPORT CALLS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import unyt
from collections import namedtuple, OrderedDict
import pdb
import tkinter as tk
from tkinter import ttk
import os
import pickle

#################################################################################################################
# SECTION 1 - FRONT END

dir_path = os.path.dirname(os.path.realpath(__file__))
root = tk.Tk()

# SECTION 1.1 - HEN Control Panel & Slaves
class HENOS_control_panel:
    '''
    A class which initializes HENOS control panel.
    '''
    def __init__(self, master):
        # Determine screen dimensions
        swidth = master.winfo_screenwidth()
        sheight = master.winfo_screenheight()
        
        # Icon
        #master.iconbitmap(dir_path + '\hen_tie.ico')
        
        # Initialize HEN Object
        self.HEN_stream_analysis = HEN()
        
        # Initialize tab system
        self.tabControl = ttk.Notebook(master,
                                  width = swidth,
                                  height = sheight)
        cp_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(cp_tab, text='HENOS Control Panel')
        self.tabControl.pack(expand=1, fill='both')
        self.HENOS_user_input = HENOS_stream_input(cp_tab, self.HEN_stream_analysis)
        self.HENOS_run_analysis = HENOS_analysis_control(cp_tab, self.HEN_stream_analysis, self.tabControl)
        cp_tab.grid_rowconfigure(0, weight=0)
        cp_tab.grid_rowconfigure(1, weight=10)
        cp_tab.grid_rowconfigure(2, weight=1)
        self.HENOS_user_input.grid(column=0, row=0, rowspan=1, sticky='ew')
        self.HENOS_user_input.HEN_stream_table.grid(column=0, row=1, rowspan=80, sticky="nw")
        self.HENOS_run_analysis.grid(column=0, row=2, sticky="nsew")

class HENOS_stream_input(ttk.Frame):
    """
    A class which holds the HEN stream user input. Slave of HENOS control panel.
    """
    def __init__(self, parent, HEN_stream_analysis):
        
        # Initialize Frame properties
        ttk.Frame.__init__(self, parent, padding='0.25i')
        
        # Instantiate HEN_table object
        self.HEN_stream_table = HENOS_table(parent)
        
        # Initialize stream input components
        self.HEN_stream_labels = ['Stream Name', 'Inlet Temperature',
                             'Outlet Temperature', 'Temperature Units',
                             'Heat Capacity Rate', 'Heat Capacity Rate Units',
                             'Heat Load', 'Heat Load Units']
        self.input_entries = {}
        
        self.HEN_stream_analysis = HEN_stream_analysis
        
        # Arrange stream input components
        for row in range(2):
            for col in range(8):
                if row == 0:
                    l = ttk.Label(self, text=self.HEN_stream_labels[col])
                    l.grid(row=row, column=col, padx=10)
                else:
                    if col in [0, 1, 2, 4, 6]:
                        e = ttk.Entry(self, width=12)
                        e.grid(row=row, column=col)
                        self.input_entries[col] = e
                    elif col == 3:
                        m = create_dropdown_menu(self, ['°C', '°F', 'K'])
                        m[0].grid(row = row, column=col)
                        self.input_entries[col] = m[1]
                    elif col == 5:
                        m = create_dropdown_menu(self, ['J·kg/(K·s)', 'J·lb/(°F·s)'])
                        m[0].grid(row = row, column=col)
                        self.input_entries[col] = m[1]
                    elif col == 7:
                        m = create_dropdown_menu(self, ['J', 'kcal', 'BTU'])
                        m[0].grid(row = row, column=col)
                        self.input_entries[col] = m[1]
        
        # Initialize and arrange 'Add Stream' button
        sub_stream = ttk.Button(self, text="Add Stream", command=self.add_stream)
        sub_stream.grid(row=1, column=8, sticky='nsew')
        
    
    def add_stream(self):
        inputSV = []
        for col in range(8):
            inputSV.append(self.input_entries[col].get())
        printed_stream_values = [inputSV[0], inputSV[1] + ' ' + inputSV[3],
                                 inputSV[2] + ' ' + inputSV[3], inputSV[4] + ' ' + inputSV[5],
                                 inputSV[6] + ' ' + inputSV[7]]

        self.HEN_stream_table.receive_new_stream(printed_stream_values)
        self.HEN_stream_table.update_stream_display()
        
        HEN.add_stream(self.HEN_stream_analysis, float(inputSV[1]), float(inputSV[2]), float(inputSV[4]), 
                       stream_name=inputSV[0])
        
        self.HEN_stream_analysis_params = self.HEN_stream_analysis.get_parameters()
        

class HENOS_table(ttk.Frame):
    '''
    A class which holds the HEN stream table. Slave of HENOS control panel.
    '''
    def __init__(self, parent):
        
        ttk.Frame.__init__(self, parent, padding='0.25i')
        
        for col in range(6):
            tl = ttk.Label(self, text=['Stream Name', 'Inlet Temperature',
                                       'Outlet Temperature', 'Heat Capacity Rate',
                                       'Heat Load', 'Activate/Deactivate'][col])
            tl.grid(row = 0, column=col, padx=10)
        
        self.HEN_stream_list = []
            
    def receive_new_stream(self, input_stream_values):
        stream_info = []
        
        for col in range(5):
            stream_info.append(input_stream_values[col])
            
        self.HEN_stream_list.append(stream_info)
    
    def update_stream_display(self):
        for stream_number in range(len(self.HEN_stream_list)):
            for element_number in range(len(self.HEN_stream_list[stream_number])):
                l = ttk.Label(self, text=self.HEN_stream_list[stream_number][element_number])
                l.grid(row=stream_number+1, column=element_number, padx=10)
        

class HENOS_analysis_control(ttk.Frame):
    '''
    A class which holds the HEN analysis buttons. Slave of HENOS control panel.
    '''
    def __init__(self, parent, HEN_object, HENOS_tab_control):
        
        ttk.Frame.__init__(self, parent, padding='0.25i')
        
        self.parent_frame = parent
        self.HEN_object = HEN_object
        self.HENOS_tab_control = HENOS_tab_control
        
        tidButton = ttk.Button(self, text="Temperature Interval Diagram", command=self.generate_tid)
        ccButton = ttk.Button(self, text="Composite Curve", command=self.generate_cc)
        solutionButton = ttk.Button(self, text="Run HEN Solution Optimization")
        
        tidButton.grid(row=0, column=2, padx=10, sticky='s')
        ccButton.grid(row=1, column=2, padx=10, sticky='s')
        solutionButton.grid(row=1, column=0, padx=10, sticky='s')
    def generate_cc(self):
        self.HEN_object.make_cc(self.HENOS_tab_control)
    def generate_tid(self):
        self.HEN_object.make_tid(self.HENOS_tab_control)

def create_dropdown_menu(master, options):
    var = tk.StringVar(master)
    menu = ttk.OptionMenu(master, var, options[0], *options)
    return [menu, var]

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

#################################################################################################################
# SECTION 2 - BACK END
class HEN:
    """
    A class that holds streams and exchangers, used to solve HEN problems
    """
    def __init__(self, delta_t = 10, cold_cost = 7e-3, hot_cost = 11e-3, flow_unit = unyt.kg/unyt.s, temp_unit = unyt.degC, cp_unit = unyt.J/(unyt.delta_degC*unyt.kg)):
        self.delta_t = delta_t * temp_unit
        self.cold_cost = cold_cost
        self.hot_cost = hot_cost
        self.flow_unit = flow_unit
        self.temp_unit = temp_unit
        self.cp_unit = cp_unit
        self.streams = OrderedDict()
        self.exchangers = OrderedDict()

        # Making unyt work since it doesn't like multiplying with °C and °F
        if self.temp_unit == unyt.degC:
            self.delta_temp_unit = unyt.delta_degC
        elif self.temp_unit == unyt.degF:
            self.delta_temp_unit = unyt.delta_degF
        else:
            self.delta_temp_unit = temp_unit
    
    def add_stream(self, t1, t2, cp, flow_rate = 1, stream_name = None, temp_unit = None, cp_unit = None, flow_unit = None):
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

        # Starting array from class data
        temperatures = np.empty( (len(self.streams), 2) )
        cp_vals = np.empty( (len(self.streams), 1) )

        for idx, values in enumerate(self.streams.items()): # values[0] has the stream names, values[1] has the properties
            temperatures[idx, 0], temperatures[idx, 1] = values[1].t1, values[1].t2
            cp_vals[idx, 0] = values[1].cp * values[1].flow_rate
        
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
        enthalpy_hot = np.sum(streams_in_interval1 * cp_vals[self.hot_streams] * delta_plotted_ylines, axis = 0) # sum(FCp_hot) * delta_t
        enthalpy_cold = np.sum(streams_in_interval2 * cp_vals[~self.hot_streams] * delta_plotted_ylines, axis = 0) # sum(FCp_cold) * delta_t
        q_interval = enthalpy_hot - enthalpy_cold # sum(FCp_hot - FCp_cold) * delta_t_interval
        
        
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
            self.first_utility_loc = len(q_sum)
            print('Warning: there is no pinch point nor a first utility\n')
        
        self.last_utility = -q_sum[-1] * self.flow_unit*self.delta_temp_unit*self.cp_unit
        self.enthalpy_hot = np.insert(enthalpy_hot, 0, 0) # The first value in enthalpy_hot is defined as 0
        # Shifting the cold enthalpy so that the first value starts at positive last_utility
        self.enthalpy_cold = np.insert(enthalpy_cold[:-1], 0, -self.last_utility)
        print('The last utility is %g %s\n' % (self.last_utility, self.last_utility.units))

        # Getting heats above / below pinch for each stream
        streams_in_interval = np.zeros((len(self.streams), len(delta_plotted_ylines)), dtype = np.int8)
        streams_in_interval[self.hot_streams, :] = streams_in_interval1
        streams_in_interval[~self.hot_streams, :] = -1*streams_in_interval2
        self._interval_heats_above = streams_in_interval[:, -1-self.first_utility_loc:] * cp_vals * delta_plotted_ylines[-1-self.first_utility_loc:] # Used in the optimization step
        q_above = np.sum(self._interval_heats_above, axis = 1)
        self._interval_heats_below = streams_in_interval[:, :-1-self.first_utility_loc] * cp_vals * delta_plotted_ylines[:-1-self.first_utility_loc] # Used in the optimization step
        q_below = np.sum(self._interval_heats_below, axis = 1)
        for idx, elem in enumerate(self.streams):
            self.streams[elem].q_above = q_above[idx] * self.first_utility.units
            self.streams[elem].q_above_remaining = q_above[idx] * self.first_utility.units
            self.streams[elem].q_below = q_below[idx] * self.first_utility.units
            self.streams[elem].q_below_remaining = q_below[idx] * self.first_utility.units
            if self.streams[elem].current_t_above is None:
                self.streams[elem].current_t_above = self._plotted_ylines[self.first_utility_loc] * self.temp_unit - self.delta_t # Shifting the cold temperature by delta T
            elif self.streams[elem].current_t_below is None:
                self.streams[elem].current_t_below = self._plotted_ylines[self.first_utility_loc] * self.temp_unit

    def make_tid(self, tab_control, show_temperatures = True, show_properties = True): # Add a show_middle_temps, show_q parameter for customization
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
        axis_delta = max(self.delta_t.value, 20) # Shifts the y-axis by at least 20 degrees
        ax1.set_xlim(-0.5, len(temperatures)-0.5)
        ax1.set_ylim(np.min(temperatures) - axis_delta, np.max(temperatures) + axis_delta)
        ax1.set_ylabel(f'Hot temperatures ({self.temp_unit})')
        ax1.set_yticks(np.linspace(np.min(temperatures) - axis_delta, np.max(temperatures) + axis_delta, 11))
        q_above_text_loc = ax1.get_ylim()[1] - 0.01*(ax1.get_ylim()[1] - ax1.get_ylim()[0])
        q_below_text_loc = ax1.get_ylim()[0] + 0.04*(ax1.get_ylim()[1] - ax1.get_ylim()[0])
        cp_text_loc = ax1.get_ylim()[0] + 0.01*(ax1.get_ylim()[1] - ax1.get_ylim()[0])
        
        
        # Manipulating the temperatures so that the hot and cold values are on the same y-position, even though they're shifted by delta_t
        plotted_ylines = np.empty(temperatures.size, dtype = 'object')
        for idx in range(len(temperatures)):
            if temperatures[idx, 0] > temperatures[idx, 1]:
                my_color = 'r'
                tplot1 = temperatures[idx, 0]
                tplot2 = temperatures[idx, 1]
            else:
                my_color = 'b'
                tplot1 = temperatures[idx, 0] + self.delta_t.value
                tplot2 = temperatures[idx, 1] + self.delta_t.value

            ax1.vlines(idx, tplot1, tplot2, color = my_color, linewidth = 0.25) # Vertical line for each stream
            if show_properties:
                q_above_text = r'$Q_{Top}$ = %g %s' % (self.streams[x_tick_labels[idx]].q_above, self.streams[x_tick_labels[idx]].q_above.units)
                ax1.text(idx, q_above_text_loc, q_above_text, ha = 'center', va = 'top') # Heat above the pinch point
                q_below_text = r'$Q_{Bot}$ = %g %s' % (self.streams[x_tick_labels[idx]].q_below, self.streams[x_tick_labels[idx]].q_below.units)
                ax1.text(idx, q_below_text_loc, q_below_text, ha = 'center', va = 'bottom') # Heat below the pinch point
                cp_text = r'$C_p$ = %g %s' % (self.streams[x_tick_labels[idx]].cp, self.streams[x_tick_labels[idx]].cp.units)
                ax1.text(idx, cp_text_loc, cp_text, ha = 'center', va = 'bottom') # Heat below the pinch point
        
        # Horizontal lines for each temperature
        for elem in self._plotted_ylines:
            ax1.axhline(elem, color = 'k', linewidth = 0.25)
            if show_temperatures:
                my_label = str(elem) + str(self.temp_unit) + ' Hot side, ' + str(elem - self.delta_t.value) + str(self.temp_unit) + ' Cold side'
                ax1.text(np.mean(ax1.get_xlim()), elem, my_label, ha = 'center', va = 'bottom')
        
        # Labeling the x-axis with the stream names
        ax1.set_xticks(range(len(temperatures)))
        ax1.set_xticklabels(x_tick_labels)

        # Adding the pinch point
        ax1.axhline(self._plotted_ylines[-2-self.first_utility_loc], color = 'k', linewidth = 0.5) # Arrays start at 0 but end at -1, so we need an extra -1 in this line and the next
        if show_temperatures:
            ax1.text(np.mean(ax1.get_xlim()), self._plotted_ylines[-2-self.first_utility_loc] - 1, 'Pinch Point', ha = 'center', va = 'top')
        plt.show(block = False)
        # Embed into GUI
        generate_GUI_plot(fig1, tab_control, 'Temperature Interval Diagram')

    def make_cc(self, tab_control):
        plt.rcParams['axes.titlesize'] = 5
        plt.rcParams['axes.labelsize'] = 5
        plt.rcParams['font.size'] = 3

        fig2, ax2 = plt.subplots(dpi = 350)
        ax2.set_title('Composite Curve')
        ax2.set_ylabel(f'Temperature ({self.temp_unit})')
        ax2.set_xlabel(f'Enthalpy ({self.first_utility.units})')
        # A given interval may not have hot or cold streams. The indexing of self.plotted_ylines attempts to fix this
        # There will still be issues if all streams on one side fully skip one or more intervals
        # TODO: use streams_in_interval1 and 2 to index self.plotted_ylines 
        ax2.plot(np.cumsum(self.enthalpy_hot), self._plotted_ylines[-len(self.enthalpy_hot):], '-or', linewidth = 0.25, ms = 1.5) # Assumes the topmost interval has a hot stream
        ax2.plot(np.cumsum(self.enthalpy_cold), self._plotted_ylines[:len(self.enthalpy_cold)] - self.delta_t.value, '-ob', linewidth = 0.25, ms = 1.5) # Assumes the lowermost interval has a cold stream

        # Text showing the utilities and overlap
        top_text_loc = ax2.get_ylim()[1] - 0.03*(ax2.get_ylim()[1] - ax2.get_ylim()[0])
        cold_text_loc = ax2.get_xlim()[0] + 0.03*(ax2.get_xlim()[1] - ax2.get_xlim()[0])
        cold_text = 'Cold utility:\n%4g %s' % (self.last_utility, self.last_utility.units)
        ax2.text(cold_text_loc, top_text_loc, cold_text, ha = 'left', va = 'top')
        hot_text_loc = ax2.get_xlim()[1] - 0.03*(ax2.get_xlim()[1] - ax2.get_xlim()[0])
        hot_text = 'Hot utility:\n%4g %s' % (self.first_utility, self.first_utility.units)
        ax2.text(hot_text_loc, top_text_loc, hot_text, ha = 'right', va = 'top')
        overlap_text_loc = np.mean(ax2.get_xlim())
        overlap = np.minimum(np.cumsum(self.enthalpy_hot)[-1], np.cumsum(self.enthalpy_cold)[-1]) - np.maximum(self.enthalpy_hot[0], self.enthalpy_cold[0])
        overlap_text = 'Heat recovery:\n%4g %s' % (overlap, self.first_utility.units)
        ax2.text(overlap_text_loc, top_text_loc, overlap_text, ha = 'center', va = 'top')

        plt.show(block = False)
        # Embed into GUI
        generate_GUI_plot(fig2, tab_control, 'Composite Curve')

        """ TODO: remove whitespace around the graphs
        ax = gca;
        ti = ax.TightInset;
        ax.Position = [ti(1), ti(2), 1 - ti(1) - ti(3), 1 - ti(2) - ti(4)]; % Removing whitespace from the graph
        """

    def place_exchangers(self):
        def _objective(self, residuals = None, Q_exchanger_above = None):
            # Starting up
            if residuals is None:
                residuals_above = np.zeros( (np.sum(self.hot_streams), self._interval_heats_above.shape[-1] + 1) ) # Extra interval is the heat coming in from "above the highest interval" (that is, always 0)
            if Q_exchanger_above is None:
                Q_exchanger_above = np.zeros((np.sum(self.hot_streams), self._interval_heats_above.shape[-1], np.sum(~self.hot_streams))) # Hot streams, intervals, and cold streams

            #Q_exchanger_above is how much heat each exchanger will transfer. It's multiplied by another array to remove matches where there are no streams
            eqn1 = self._interval_heats_above[self.hot_streams] + residuals_above[:, :-1] - residuals_above[:, 1:] - np.sum(Q_exchanger_above, axis = 2)*np.array(self._interval_heats_above[~self.hot_streams], dtype = bool)
            eqn2 = self._interval_heats_above[~self.hot_streams] - np.sum(Q_exchanger_above, axis = 0)*np.array(self._interval_heats_above[self.hot_streams], dtype = bool)
            eqn3 = self.aaa - - np.sum(Q_exchanger_above, axis = 1)*np.array(self._interval_heats_above[self.hot_streams], dtype = bool)
            #for i in range(1, len(self._plotted_ylines+1)): # i is the number of intervals
                #self._interval_heats_above[:, -i] + 

    def add_exchanger(self, stream1, stream2, heat = 'auto', ref_stream = 1, t_in = None, t_out = None, pinch = 'above', exchanger_name = None, U = 100, U_unit = unyt.J/(unyt.s*unyt.m**2*unyt.delta_degC), 
        exchanger_type = 'Fixed Head', cost_a = 0, cost_b = 0, pressure = 0):

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
            if t_in is not None and t_out is not None: # Operating the exchanger using temperatures
                if ref_stream == 1: # Temperature values must be referring to only one of the streams - the first stream in this case
                    heat = self.streams[stream1].cp * self.streams[stream1].flow_rate * ((t_in - t_out)*self.delta_temp_unit)
                else:
                    heat = self.streams[stream2].cp * self.streams[stream2].flow_rate * ((t_in - t_out)*self.delta_temp_unit)
            elif type(heat) is str and heat.casefold() == 'auto':
                heat = np.minimum(np.abs(self.streams[stream1].q_above_remaining), np.abs(self.streams[stream2].q_above_remaining)) # Maximum heat exchanged is the minimum total heat between streams
            else: # Heat passed was a number, and no temperatures were passed
                heat = heat * self.streams[stream1].q_above.units

            if self.streams[stream1].q_above_remaining < 0: # We want Stream1 to be the hot stream, but it's currently the cold stream
                stream1, stream2 = stream2, stream1
            s1_q_above = self.streams[stream1].q_above_remaining - heat
            s1_t_above = self.streams[stream1].current_t_above - heat/(self.streams[stream1].cp * self.streams[stream1].flow_rate)
            s2_q_above = self.streams[stream2].q_above_remaining + heat
            s2_t_above = self.streams[stream2].current_t_above + heat/(self.streams[stream2].cp * self.streams[stream2].flow_rate)
            
            # Data validation
            if s1_t_above < s2_t_above:
                raise ValueError(f'Match is thermodynamically impossible, as the hot stream is leaving with a temperature of {s1_t_above}, while the cold stream is leaving with a temperature of {s2_t_above}')
            elif s1_t_above - s2_t_above < self.delta_t:
                    print(f"Warning: match violates minimum ΔT, which equals {self.delta_t:.4g}\nThis match's ΔT is {s1_t_above-s2_t_above:.4g}")
            
            # Recording the data
            delta_T1 = self.streams[stream1].current_t_above - s2_t_above
            delta_T2 = s1_t_above - self.streams[stream2].current_t_above
            self.streams[stream1].q_above_remaining = s1_q_above
            self.streams[stream1].current_t_above = s1_t_above
            self.streams[stream2].q_above_remaining = s2_q_above
            self.streams[stream2].current_t_above = s2_t_above

        elif pinch.casefold() == 'below' or pinch.casefold() == 'bottom' or pinch.casefold() == 'bot' or pinch.casefold == 'down':
            if t_in is not None and t_out is not None: # Operating the exchanger using temperatures
                if ref_stream == 1: # Temperature values must be referring to only one of the streams - the first stream in this case
                    heat = self.streams[stream1].cp * self.streams[stream1].flow_rate * ((t_in - t_out)*self.delta_temp_unit)
                else:
                    heat = self.streams[stream2].cp * self.streams[stream2].flow_rate * ((t_in - t_out)*self.delta_temp_unit)
            elif type(heat) is str and heat.casefold() == 'auto':
                heat = np.minimum(np.abs(self.streams[stream1].q_below_remaining), np.abs(self.streams[stream2].q_below_remaining)) # Maximum heat exchanged is the minimum total heat between streams
            else: # Heat passed was a number, and no temperatures were passed
                heat = heat * self.streams[stream1].q_above.units
            
            if self.streams[stream1].q_below_remaining < 0: # We want Stream1 to be the hot stream, but it's currently the cold stream
                stream1, stream2 = stream2, stream1
            s1_q_below = self.streams[stream1].q_below_remaining - heat
            s1_t_below = self.streams[stream1].current_t_below - heat/(self.streams[stream1].cp * self.streams[stream1].flow_rate)
            s2_q_below = self.streams[stream2].q_below_remaining + heat
            s2_t_below = self.streams[stream2].current_t_below + heat/(self.streams[stream2].cp * self.streams[stream2].flow_rate)
            
            # Data validation
            if s1_t_below < s2_t_below:
                raise ValueError(f'Match is thermodynamically impossible, as the hot stream is leaving with a temperature of {s1_t_below}, while the cold stream is leaving with a temperature of {s2_t_below}')
            elif s1_t_below - s2_t_below < self.delta_t:
                    print(f"Warning: match violates minimum ΔT, which equals {self.delta_t:.4g}\nThis match's ΔT is {s1_t_below-s2_t_below:.4g}")
            
            # Recording the data
            delta_T1 = self.streams[stream1].current_t_below - s2_t_below
            delta_T2 = s1_t_below - self.streams[stream2].current_t_below
            self.streams[stream1].q_below_remaining = s1_q_below
            self.streams[stream1].current_t_below = s1_t_below
            self.streams[stream2].q_below_remaining = s2_q_below
            self.streams[stream2].current_t_below = s2_t_below
        
        # Creating the exchanger object
        delta_T_lm = (delta_T1.value - delta_T2.value) / (np.log(delta_T1.value/delta_T2.value)) * self.delta_temp_unit
        self.exchangers[exchanger_name] = HeatExchanger(stream1, stream2, heat, pinch, U, U_unit, delta_T_lm, exchanger_type)
        
    def save(self, name):
        file_name = name + ".p"
        pickle.dump(self, open( file_name, "wb" ))

    def load(self,file):
        return pickle.load(open(file, 'rb'))
            

class Stream():
    def __init__(self, t1, t2, cp, flow_rate, flow_unit, temp_unit, cp_unit):
        self.t1 = t1 * temp_unit
        self.t2 = t2 * temp_unit
        self.cp = cp * cp_unit
        self.flow_rate = flow_rate * flow_unit
        self.q_above = None # Will be updated once pinch point is found
        self.q_below = None

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
        text =(f'{stream_type} stream with T_in = {self.t1} and T_out = {self.t2}\n'
            f'c_p = {self.cp} and flow rate = {self.flow_rate}\n')
        if self.q_above is not None:
            text += f'Above pinch: {self.q_above} total, {self.q_above_remaining} remaining, T = {self.current_t_above:.4g}\n'
            text += f'Below pinch: {self.q_below} total, {self.q_below_remaining} remaining, T = {self.current_t_below:.4g}\n'
        return text

class HeatExchanger():
    def __init__(self, stream1, stream2, heat, pinch, U, U_unit, delta_T_lm, exchanger_type):
        self.stream1 = stream1
        self.stream2 = stream2
        self.heat = heat
        self.pinch = pinch
        self.U = U * U_unit
        self.delta_T_lm = delta_T_lm
        self.area = self.heat / (self.U * self.delta_T_lm)
        self.exchanger_type = exchanger_type

        if exchanger_type == 'Fixed Head':
            self.cost_base = 0
        elif exchanger_type == 'Floating Head':
            self.cost_base = 0
        elif exchanger_type == 'U-Tube':
            self.cost_base = 0
        elif exchanger_type == 'Kettle Vaporizer':
            self.cost_base = 0

    def __repr__(self):
        text = (f'A {self.exchanger_type} heat exchanger exchanging {self.heat} between {self.stream1} and {self.stream2} {self.pinch} the pinch\n'
            f'Has a U = {self.U:.4g}, area = {self.area:.4g}, and ΔT_lm = {self.delta_T_lm:.4g}\n'
            f'Has a base cost of {self.cost_base} and a free on board cost of TBD\n')
        return text
     
## SECTION ? - RUN APPLICATION
if __name__ == '__main__':
    HEN_app = HENOS_control_panel(root)
    root.mainloop()
