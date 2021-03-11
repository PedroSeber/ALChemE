##############################################################################
# IMPORT CALLS
##############################################################################
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import unyt
from collections import namedtuple, OrderedDict
import tkinter as tk
from tkinter import ttk
from hen_design import HEN
from hen_design import generate_GUI_plot
import subprocess

# CLASSES
class HENOS_app():
    '''
    A class which holds the HENOS application. Slave of root window.
    '''
    def __init__(self, master):
        
        # Defining variables
        self.master = master
        self.style = ttk.Style()
        self.style.configure('main.TFrame')
        
        # Determine screen dimensions
        swidth = master.winfo_screenwidth()
        sheight = master.winfo_screenheight()
        top = master.winfo_toplevel()
        
        # Initialize dropdown menu
        self.HENOS_dropdown_menu = tk.Menu(top)
        top['menu'] = self.HENOS_dropdown_menu
        
        # Initialize tab systemos
        self.tabControl = ttk.Notebook(self.master, width=swidth, height=sheight)
    
        
        # Intialize control panel
        control_panel_Tab = ttk.Frame(self.tabControl)
        self.tabControl.add(control_panel_Tab, text='HENOS Control Panel')
        self.tabControl.pack(expand=1, fill='both')
        
        # Initialize HEN object
        self.HEN_object = HEN()
        
        # Initialize control panel elements
        self.HENOS_object_explorer = HENOS_object_explorer(control_panel_Tab, self.HEN_object)
        self.HENOS_si_frame = HENOS_stream_input(control_panel_Tab, self.HEN_object, self.HENOS_object_explorer)
        self.HENOS_ga_frame = HENOS_graphical_analysis_controls(control_panel_Tab, self.tabControl, self.HENOS_object_explorer, self.HEN_object)
        self.HENOS_uc_frame = HENOS_user_constraints(control_panel_Tab)
        self.HENOS_os_frame = HENOS_optimization_controls(control_panel_Tab, self.HEN_object, self.HENOS_object_explorer, self.HENOS_uc_frame)

        # Intialize dropdown menu options
        self.fileMenu = tk.Menu(self.HENOS_dropdown_menu, tearoff=0)
        self.fileMenu.add_command(label='New')
        self.fileMenu.add_command(label='Open', command=self.loadfile)
        self.fileMenu.add_command(label='Save', command=self.savefile)
        self.fileMenu.add_command(label='Save As', command=self.savefile)
        self.HENOS_dropdown_menu.add_cascade(label='File', menu=self.fileMenu)
        self.HENOS_dropdown_menu.add_cascade(label='Settings')
        
        # Placing control panel elements
        
        control_panel_Tab.rowconfigure(2, weight=1)
        control_panel_Tab.rowconfigure(3, weight=2)
        control_panel_Tab.rowconfigure(4, weight=10)
        control_panel_Tab.columnconfigure(9, weight=1)
        
        self.HENOS_si_frame.grid(row=0, rowspan=3, column=0, sticky='nsew')
        
        self.HENOS_ga_frame.grid(row=0, column=9, sticky='new')
        self.HENOS_os_frame.grid(row=2, rowspan=2, column=9,  sticky='nsew')
        
        self.HENOS_object_explorer.grid(row=3, column=0, rowspan=40, columnspan=8, sticky='nsew')
        self.HENOS_uc_frame.grid(row=4, column=9, sticky='nsew')
    
    def savefile(self):
        self.HEN_object.save('alcheme_HEN')
    
    def loadfile(self):
        self.HEN_object.load()

class HENOS_stream_input(ttk.Frame):
    '''
    A class which holds the HENOS stream input. Slave of HENOS_app.    
    '''
    def __init__(self, master, HEN_object, HEN_object_explorer):        
        # Initialize frame properties
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        

        # Defining variables
        self.HEN_object = HEN_object
        self.HEN_object_explorer = HEN_object_explorer
        self.HEN_stream_labels = ['Stream Name', 'Inlet Temperature',
                             'Outlet Temperature', '',
                             'Heat Capacity', '',
                             'Flow Rate', '', '', 'Heat Load', '']
        self.HEN_exchanger_labels = ['Exchanger Name', 'Hot Stream', 'Cold Stream', '', 'ΔT', '', 'Reference Stream', '', '', 'Heat Load', '',
                             '', 'Gauge Pressure', '',
                             'Exchanger Type', 'Cost Parameter A', 'Cost Parameter B']
        self.HEN_utility_labels = ['Utility Name', 'Utility Type', 'Temperature', '', 'Cost', '']
        self.input_entries = {}
        
        # Initialize Input Label
        siLabel = ttk.Label(self, text='Stream Input', font=('Helvetica', 10, 'bold', 'underline'))
        siLabel.grid(row=0, column=0, sticky='w')
        
        eiLabel = ttk.Label(self, text='Heat Exchanger Input', font=('Helvetica', 10, 'bold', 'underline'))
        eiLabel.grid(row=3, column=0, pady=(30,0), sticky='w')
        
        uiLabel = ttk.Label(self, text='Utility Input', font=('Helvetica', 10, 'bold', 'underline'))
        uiLabel.grid(row=8, column=0, pady=(30,0), sticky='w')
        
        # Arrange stream input components
        for row in range(1,3):
            for col in range(11):
                if row == 1 and col in [0, 1, 2, 4, 6, 9]:
                    l = ttk.Label(self, text=self.HEN_stream_labels[col])
                    l.grid(row=row, column=col, padx=10)
                elif row == 1 and col in [3, 5, 7, 10]:
                    l = ttk.Label(self, width=12)
                    l.grid(row=row, column=col, padx=10)
                else:
                    if col in [0, 1, 2, 4, 6, 9]:
                        e = ttk.Entry(self, width=12)
                        e.grid(row=row, column=col)
                        self.input_entries[str([row, col])] = e
                    elif col == 3:
                        m = create_dropdown_menu(self, ['°C', '°F', '°K'])
                        m[0].grid(row = row, column=col, sticky='w')
                        self.input_entries[str([row, col])] = m[1]
                    elif col == 5:
                        m = create_dropdown_menu(self, ['J/(kg·°C)', 'BTU/(lb·°F)', 'J/(kg·°K)'])
                        m[0].grid(row = row, column=col, sticky='w')
                        self.input_entries[str([row, col])] = m[1]
                    elif col == 7:
                        m = create_dropdown_menu(self, ['kg/s', 'lb/s'])
                        m[0].grid(row = row, column=col, sticky='w')
                        self.input_entries[str([row, col])] = m[1]
                    elif row == 2 and col == 8:
                        l = ttk.Label(self, text='OR', font=('Helvetica', 10, 'bold'))
                        l.grid(row=row, column=col, padx=(0,10), sticky='w')
                    elif col == 10:
                        m = create_dropdown_menu(self, ['W', 'kcal/s', 'BTU/s'])
                        m[0].grid(row = row, column=col, sticky='w')
                        self.input_entries[str([row, col])] = m[1]
        
        # Arrange exchanger input components
        for row in range(4, 8):
             for col in range(11):
                if row == 4: 
                    if col in [0, 1, 2, 4, 6, 9]:
                        l = ttk.Label(self, text=self.HEN_exchanger_labels[col])
                        l.grid(row=row, column=col, padx=10)
                    else:
                        l = ttk.Label(self, width=6)
                        l.grid(row=row, column=col, padx=10)
                elif row == 5:
                    if col in [0, 1, 2, 4, 9]:
                        e = ttk.Entry(self, width=12)
                        e.grid(row=row, column=col)
                        self.input_entries[str([row, col])] = e
                    elif col == 4:
                        m = create_dropdown_menu(self, ['Pa', 'psi'])
                        m[0].grid(row = row, column=col, sticky='w')
                        self.input_entries[str([row, col])] = m[1]
                    elif col == 6:
                        m = create_dropdown_menu(self, ['Hot', 'Cold'])
                        m[0].grid(row = row, column=col)
                        self.input_entries[str([row, col])] = m[1]
                    elif col == 8:
                        l = ttk.Label(self, text='OR', font=('Helvetica', 10, 'bold'))
                        l.grid(row=row, column=col, sticky='w')
                    elif col == 5:
                        m = create_dropdown_menu(self, ['°C', '°F', '°K'])
                        m[0].grid(row = row, column=col, sticky='w')
                        self.input_entries[str([row, col])] = m[1]
                    elif col == 10:
                        m = create_dropdown_menu(self, ['W', 'kcal/s', 'BTU/s'])
                        m[0].grid(row = row, column=col, sticky='w')
                        self.input_entries[str([row, col])] = m[1]
                elif row == 6:
                    if col in [0, 1, 2]:
                            l = ttk.Label(self, text=self.HEN_exchanger_labels[-3 + col])
                            l.grid(row = row, column=col, padx=10, pady=(10,0))
                    elif col == 4:
                            l = ttk.Label(self, text='Gauge Pressure')
                            l.grid(row = row, column =col, padx=10, pady=(10,0))
                    elif col == 6:
                            l = ttk.Label(self, text = 'U')
                            l.grid(row = row, column =col, padx=10, pady=(10,0))
                else:
                    if col in [0, 1, 2, 4, 5, 6, 7]:
                        if col == 0:
                            m = create_dropdown_menu(self, ['Fixed Head', 'Floating Head', 'U Tube', 'Kettle Vaporizer'])
                            m[0].grid(row = row, column=col)
                            self.input_entries[str([row, col])] = m[1]
                        elif col == 1 or col == 2 or col == 4 or col == 6:
                            e = ttk.Entry(self, width=12)
                            e.grid(row=row, column=col) 
                            self.input_entries[str([row, col])] = e
                        elif col == 5:
                            m = create_dropdown_menu(self, ['Pa', 'psi'])
                            m[0].grid(row = row, column=col, sticky='w')
                            self.input_entries[str([row, col])] = m[1]
                        elif col == 7:
                            m = create_dropdown_menu(self, ['Test'])
                            m[0].grid(row = row, column=col, sticky='w')
                            self.input_entries[str([row, col])] = m[1]
                            
        # Arrange utility input components
        for row in range(10,12):
            for col in range(6):
                if row == 10:
                    l = ttk.Label(self, text=self.HEN_utility_labels[col])
                    l.grid(row=row, column=col, padx=10)
                else:
                    if col in [0, 2, 4]:
                        e = ttk.Entry(self, width=12)
                        e.grid(row=row, column=col)
                        self.input_entries[str([row, col])] = e
                    elif col == 1:
                        m = create_dropdown_menu(self, ['Hot', 'Cold'])
                        m[0].grid(row = row, column=col)
                        self.input_entries[str([row, col])] = m[1]
                    elif col == 3:
                        m = create_dropdown_menu(self, ['°C', '°F', '°K'])
                        m[0].grid(row = row, column=col, sticky='w')
                        self.input_entries[str([row, col])] = m[1]
                    elif col == 5:
                        m = create_dropdown_menu(self, ['$/kW', '€/kW'])
                        m[0].grid(row = row, column=col, sticky='w')
                        self.input_entries[str([row, col])] = m[1]
        # Initialize and arrange 'Add Stream' button
        sub_stream = ttk.Button(self, text="Add Stream", command=self.add_stream)
        sub_stream.grid(row=2, column=11, sticky='nsew')
        
        # Initialize and arrange 'Add Exchanger' button
        sub_exchanger = ttk.Button(self, text="Add Exchanger", command=self.add_exchanger)
        sub_exchanger.grid(row=7, column=11, sticky='nsew')
        
        # Initialize and arrange 'Add Utility' button
        sub_utility = ttk.Button(self, text="Add Utility", command=self.add_utility)
        sub_utility.grid(row=11, column=11, sticky='nsew')
    
    def add_stream(self):
        # Populating raw input data vector
        raw_input = []
        for col in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]:
            rawdata = self.input_entries[str([2, col])].get()
            if rawdata == '': rawdata = None
            raw_input.append(rawdata)
            if col in [0, 1, 2, 4, 6, 9]:
                self.input_entries[str([2, col])].delete(0, 'end')
        
        # HEN object input data transfer, convert all numeric values to floats
        for ii in [1, 2, 4, 6, 8]:
            try:
                numericdata = float(raw_input[ii])
                raw_input[ii] = numericdata
            except TypeError:
                continue
        
        # Check if flow_rate = None, convert to 1 if so
        if raw_input[6] == None:
            raw_input[6] = 1
        
        # Convert temperature unit input to unyt input
        if raw_input[3] == '°C':
            self.temp_unit = unyt.degC
        elif raw_input[3] == '°F':
            self.temp_unit = unyt.degF
        else:
            self.temp_unit = unyt.degK
            
        # Convert cp unit input to unyt input
        if raw_input[5] == 'J/(kg·°C)':
            raw_input[5] = unyt.J/(unyt.delta_degC*unyt.kg)
        elif raw_input[5] == 'BTU/(lb·°F)':
            raw_input[5] = unyt.BTU/(unyt.delta_degF*unyt.lb)
        else:
            raw_input[5] = unyt.J/(unyt.delta_degK*unyt.kg)
        
        # Convert flow rate unit input to unyt input
        if raw_input[7] == 'kg/s':
            raw_input[7] = unyt.kg/unyt.s
        else:
            raw_input[7] = unyt.lb/unyt.s
        
        # Convert heat load unit input to unyt input
        if raw_input[9] == 'W':
            raw_input[9] = unyt.W
        elif raw_input[9] == 'kcal/s':
            raw_input[9] = unyt.kcal/unyt.s
        else:
            raw_input[9] = unyt.BTU/unyt.s
        
        # Add input to HEN object and data display
        self.HEN_object.add_stream(t1 = raw_input[1], t2 = raw_input[2], cp = raw_input[4], flow_rate = raw_input[6], heat = raw_input[8], stream_name = raw_input[0], HENOS_oe_tree = self.HEN_object_explorer.objectExplorer, temp_unit = self.temp_unit, cp_unit = raw_input[5], flow_unit = raw_input[7], heat_unit = raw_input[9])    

    def add_exchanger(self):
        # Define variables
        errorFlag = False
        
        # Call get_parameters()
        self.HEN_object.get_parameters()
        
        # Populating raw input data vector
        raw_input = []
        for row in [5,7]:
            if row == 5:
                for col in [0, 1, 2, 4, 5, 6, 9, 10]:
                    rawdata = self.input_entries[str([5, col])].get()
                    if rawdata == '': rawdata = None
                    raw_input.append(rawdata)
            elif row == 7:
                for col in [0, 1, 2, 4, 5, 6, 7]:
                    rawdata = self.input_entries[str([7, col])].get()
                    if rawdata == '': rawdata = None
                    raw_input.append(rawdata)
        
        # HEN object input data transfer, convert all numeric values to floats
        for ii in [3, 6, 9, 10, 11, 13]:
            try:
                numericdata = float(raw_input[ii])
                raw_input[ii] = numericdata
            except TypeError:
                continue
        
        # Check if Hot Stream and Cold Stream exist
        streamList = self.HEN_object.streams.keys()
        
        if raw_input[1] not in streamList:
            errorFlag = True
            errorMessage = 'Hot stream ' + raw_input[1] + ' does not exist'
        elif self.HEN_object.streams[raw_input[1]].stream_type != 'Hot':
            errorFlag = True
            errorMessage = 'Stream ' + raw_input[1] + ' is not a hot stream'
        
        if raw_input[2] not in streamList:
            errorFlag = True
            errorMessage = 'Cold stream ' + raw_input[2] + 'does not exist'
        elif self.HEN_object.streams[raw_input[2]].stream_type != 'Cold':
            errorFlag = True
            errorMessage = 'Stream ' + raw_input[2] + ' is not a cold stream'
        
        # Convert temperature units into unyt
        if raw_input[4] == '°C':
            raw_input[4] = unyt.degC
        elif raw_input[4] == '°F':
            raw_input[4] = unyt.degF
        else:
            raw_input[4] = unyt.degK
        
        
        # Convert heat load units into unyt
        if raw_input[7] == 'W':
            raw_input[7] = unyt.W
        elif raw_input[7] == 'kcal/s':
            raw_input[7] = unyt.kcal/unyt.s
        else:
            raw_input[7] = unyt.BTU/unyt.s
        
        # Convert pressure units into unyt
        if raw_input[-3] == 'Pa':
            raw_input[-3] = unyt.Pa
        elif raw_input[-3] == 'psi':
            raw_input[-3] = unyt.psi
        
        # Convert heat transfer coefficient units into unyt
        if raw_input[-1] == 'J/(°C·m²·s)':
            raw_input[-1] = unyt.J/(unyt.s*unyt.m**2*unyt.delta_degC)
        
        # Check if cost parameter A and B exist, set to 0
        if raw_input[9] == None:
            raw_input[9] = 0
        if raw_input[10] == None:
            raw_input[10] = 0
        
        # Check if pressure exists; if not, set to 0
        if raw_input[11] == None:
            raw_input[11] = 0
        
        # Convert reference stream to a number
        if raw_input[5] == 'Hot':
            raw_input[5] = 1
        else:
            raw_input[5] = 2
        
        # Submit exchanger to back end
        self.HEN_object.add_exchanger(stream1 = raw_input[1], stream2 = raw_input[2], ref_stream = raw_input[5], t_in = raw_input[6], t_out=0, exchanger_name = raw_input[0], exchanger_type = raw_input[8], cost_a = raw_input[9], cost_b = raw_input[10], pressure = raw_input[11], pressure_unit = raw_input[12])
        
        # If there are no errors, clear all entries
        if errorFlag == False:
            for row5col in [0, 1, 2, 4, 9]:
                self.input_entries[str([5, row5col])].delete(0, 'end')
            for row7col in [1, 2, 4, 6]:
                self.input_entries[str([7, row7col])].delete(0, 'end')
        
        print(raw_input)
        
    def add_utility(self):
        errorFlag = False
        
        raw_input = []
        for col in range(6):
            rawdata = self.input_entries[str([11, col])].get()
            if rawdata == '': rawdata = None
            raw_input.append(rawdata)        
        
        self.HEN_object.add_utility(utility_type = raw_input[1], temperature = float(raw_input[2]), cost = float(raw_input[4]), utility_name = raw_input[0], temp_unit = unyt.degC, HENOS_oe_tree = self.HEN_object_explorer.objectExplorer)#, cost_unit = raw_input[5])
        
        if errorFlag == False:
            for row in [0, 2, 4]:
                self.input_entries[str([11, row])].delete(0, 'end')
        
class HENOS_object_explorer(ttk.Frame):
    '''
    A class which holds the HENOS object explorer and visualizer. Slave of
    HENOS_app.
    '''
    def __init__(self, master, HEN_object):
        # Initialize frame properties
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Defining variables
        self.HEN_object = HEN_object
        
        # Initialize Object Explorer Label
        oeLabel = ttk.Label(self, text='Object Explorer', font=('Helvetica', 10, 'bold', 'underline'))
        oeLabel.grid(row=0, column=0, sticky='w')
        
        
        # Initialize object visualizer label
        tLabel = ttk.Label(self, text='Terminal Display', font=('Helvetica', 10, 'bold', 'underline'))
        tLabel.grid(row=41, column=0, sticky='w')
        
        # Initialize object explorer        
        self.objectExplorer = HENOS_objE_tree(self, self.HEN_object)
        
        # Initialize object visualizer
        self.objectVisualizer = HENOS_objE_display(self, self.HEN_object)
        
        # Initialize object explorer control buttons
        self.delete_stream = ttk.Button(self, text='Delete Stream', command=self.objectExplorer.delete_item)
        self.activate_deactivate_stream = ttk.Button(self, text='Activate/Deactivate Stream', command=self.objectExplorer.activate_deactivate_stream)
        self.delete_stream.grid(row=0, column=3, padx=5)
        self.activate_deactivate_stream.grid(row=0, column=2, padx=5)
        
        # Initialize object visualizer control buttons
        self.clear_display = ttk.Button(self, text='Clear Display', command=self.objectVisualizer.clearscreen)
        self.clear_display.grid(row = 41, column=3, sticky='e', padx=5)
        
        # Place object explorer and visualizer
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(41, weight=0)
        self.rowconfigure(42, weight=1)
        #self.objectExplorer.grid(row=0, column=0)
        self.objectExplorer.grid(row=1, column=0, rowspan=40, columnspan=8, padx=5, pady=(5,15), sticky='nsew')
        self.objectVisualizer.grid(row=42, column=0, rowspan=1, columnspan=8, padx=5, pady=(5,5), sticky='nsew')

class HENOS_objE_tree(ttk.Treeview):
    '''
    A class which holds the Treeview object which forms the basis of the
    object explorer. Slave of HENOS_object_explorer
    '''
    def __init__(self, master, HEN_object):
        # Initialize treeview properties
        ttk.Treeview.__init__(self, master, show='tree', selectmode='none')
        style = ttk.Style()
        style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nswe'})])
        self['columns'] = ('1', '2', '3', '4', '5')
        self.column('1', width=170)
        self.column('2', width=170)
        self.column('3', width=170)
        self.column('4', width=170)
        self.column('5', width=170)
        
        # Defining variables
        self.HEN_object = HEN_object
        self.master = master
        
        # Intialize vertical scrollbar
        self.verscrlbar = ttk.Scrollbar(self, orient='vertical', command=self.yview)
        self.verscrlbar.pack(side='right', fill='y')
        self.configure(yscrollcommand=self.verscrlbar.set)
        
        # Intialize object classes in treeview
        self.StreamNode = self.insert('', index=0, iid=0, text='STREAMS', values=('Inlet Temperature', 'Outlet Temperature', 'Heat Capacity Rate', 'Heat Load', 'Status'))
        self.HXNode = self.insert('', index=1, iid=1, text='HEAT EXCHANGERS', values=('Hot Stream', 'Cold Stream', 'Heat Exchange', 'FoB Cost', 'Status'))
        self.UtilityNode = self.insert('', index=2, iid=2, text='UTILITIES', values=('Utility Type', 'Temperature', 'Cost', '', 'Status'))
        
        # Initialize 'Single Click' Event (Show Selected Object in Object Explorer)
        self.bind('<Button-1>', self.on_click)
        self.bind("<Double-Button-1>", self.send2screen)
    
    def on_click(self, event):
        tree = event.widget
        item_name = tree.identify_row(event.y)
        if item_name:
            tags = tree.item(item_name, 'tags')
            if tags and (tags[0] == 'selectable'):
                tree.selection_set(item_name)
        
    def receive_new_stream(self, oeDataVector):
        self.insert(self.StreamNode, 'end', text=oeDataVector[0], values=(str(oeDataVector[1]), str(oeDataVector[2]), str(oeDataVector[3]), str(oeDataVector[4]), 'Active'), tags='selectable')
        
    def receive_new_exchanger(self, oeDataVector):
        self.insert(self.HXNode, 'end', text=oeDataVector[0], values=(str(oeDataVector[1]), str(oeDataVector[2]), str(oeDataVector[3]), str(oeDataVector[4]), 'Active'), tags='selectable')
    
    def receive_new_utility(self, oeDataVector):
        self.insert(self.UtilityNode, 'end', text=oeDataVector[0], values=(str(oeDataVector[1]), str(oeDataVector[2]), str(oeDataVector[3]), str(oeDataVector[4]), 'Active'), tags='selectable')
    
    def delete_item(self):
        HEN_selectedObject  = self.selection()[0]
        HEN_sO_name = self.item(HEN_selectedObject, 'text')
        self.delete(HEN_selectedObject)
        self.HEN_object.delete(HEN_sO_name)
        
    def activate_deactivate_stream(self):
        HEN_selectedObject = self.selection()
        
        for stream in HEN_selectedObject:
            HEN_sO_name = self.item(stream, 'text')
            HEN_sO_status = self.item(stream, 'values')[-1]
        
            if HEN_sO_status == 'Active':
                self.HEN_object.inactivate_stream(HEN_sO_name)
                objValues = self.item(stream, 'values')[0:-1]
                self.insert(self.StreamNode, self.index(stream), text=HEN_sO_name, values=(objValues[0],objValues[1], objValues[2], objValues[3], 'Inactive'), tags='selectable')
                self.delete(stream)
            elif HEN_sO_status == 'Inactive':
                self.HEN_object.activate_stream(HEN_sO_name)
                objValues = self.item(stream, 'values')[0:-1]
                self.insert(self.StreamNode, self.index(stream), text=HEN_sO_name, values=(objValues[0],objValues[1], objValues[2], objValues[3], 'Active'), tags='selectable')
                self.delete(stream)
        
    def send2screen(self, event):
        self.on_click(event)
        HEN_selectedObject = self.selection()
        if HEN_selectedObject != ():
            HEN_sO_name = self.item(HEN_selectedObject, 'text')
            HEN_sO_parent_iid = self.parent(HEN_selectedObject[0])
            tag2 = 0
            
            if self.item(HEN_sO_parent_iid, 'text') == 'STREAMS':
                objID = 'stream'
            elif self.item(HEN_sO_parent_iid, 'text') == 'HEAT EXCHANGERS':
                objID = 'hx'
            elif self.item(HEN_sO_parent_iid, 'text') == 'UTILITIES':
                objID = 'utility'
                tag2 = self.item(HEN_selectedObject, 'values')[0]
            
            self.master.objectVisualizer.printobj2screen(HEN_sO_name, objID, tag2)

class HENOS_objE_display(tk.Text):
    def __init__(self, master, HEN_object):
        # Initialize text properties
        tk.Text.__init__(self, master, highlightthickness=0)
        
        # Defining variables
        self.HEN_object = HEN_object
        
        # Initialize vertical scrollbar
        self.verscrlbar = ttk.Scrollbar(self, orient='vertical', command=self.yview)
        self.verscrlbar.pack(side='right', fill='y')
        self.configure(yscrollcommand=self.verscrlbar.set)
    
        # Initialize >>>
        self.insert('end', '-'*65+ '***INITIALIZED***' + '-'*65 + '\n\n')
        self.insert('end', '>>> ')
        #self.config(state='disabled')
    
    def print2screen(self, message, newcommand):
        self.insert('end', message + '\n')
        if newcommand == True:
            self.insert('end', '>>> ')
    
    def printobj2screen(self, object_name, tag, tag2):
        if object_name not in ['STREAMS', 'HEAT EXCHANGERS', 'UTILITIES']:
            commandtext = str('displaying object ' + object_name + '...\n')
            self.insert('end', commandtext)
            if tag == 'stream':
                displaytext = str(self.HEN_object.streams[object_name])
            elif tag == 'hx':
                displaytext = str(self.HEN_object.exchanger[object_name])
            elif tag == 'utility':
                if tag2 == 'hot':
                    displaytext = str(self.HEN_object.hot_utilities[object_name])
                else:
                    displaytext = str(self.HEN_object.cold_utilities[object_name])
            self.insert('end', displaytext + '\n\n')
            self.insert('end', '>>> ')
            self.see('end')
    
    def printsolutionmatrix(self, object_name):
        self.insert('end', 'Solver has converged.\n')
        self.insert('end', object_name)
        self.insert('end', '\n>>> ')

    def clearscreen(self):
        self.delete('1.0', 'end')
        self.insert('end', '-'*65 + '***INITIALIZED***' + '-'*65 + '\n\n')
        self.insert('end', '>>> ')    

class HENOS_graphical_analysis_controls(ttk.Frame):
    def __init__(self, master, tabControl, object_explorer, HEN_object):
        # Initialize frame properties
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Define variables
        self.HEN_object = HEN_object
        self.master = master
        self.tabControl = tabControl
        
        # Initialize graphical analysis label
        gaLabel = ttk.Label(self, text='Graphical Analysis', font=('Helvetica', 10, 'bold', 'underline'))
        gaLabel.grid(row=0, column=0, sticky='nw')
        
        # Initialize buttons
        generate_cc = ttk.Button(self, text='Composite Curve', command=self.make_CC)
        generate_tid = ttk.Button(self, text='TID', command=self.make_TID)
        
        # Settings
        self.showT = tk.BooleanVar()
        self.showP = tk.BooleanVar()
        show_temperatures = ttk.Checkbutton(self, text='Show Temperatures', variable=self.showT, offvalue=False, onvalue=True)
        show_properties = ttk.Checkbutton(self, text='Show Properties', variable=self.showP, offvalue=False, onvalue=True)
        
        # Place
        generate_cc.grid(row=1, column=0, padx=(0,15))
        generate_tid.grid(row=1, column=2, padx=(15,0))
        show_temperatures.grid(row=1, column=3, padx=5)
        show_properties.grid(row=1, column=4, padx=5)
    
    def make_CC(self):
        self.HEN_object.get_parameters()
        self.HEN_object.make_cc(self.tabControl)
        
    def make_TID(self):
        self.HEN_object.get_parameters()
        self.HEN_object.make_tid(self.showT.get(), self.showP.get(), self.tabControl)

class HENOS_optimization_controls(ttk.Frame):
    def __init__(self, master, HEN_object, HEN_object_explorer, HEN_uC_explorer):
        # Intialize fram properties
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        self.HEN_object = HEN_object
        self.HEN_object_explorer = HEN_object_explorer
        self.HEN_uC_explorer = HEN_uC_explorer
        self.input_entries = {}
        
        # Initialize optimization suite label
        osLabel = ttk.Label(self, text='Optimization Suite', font=('Helvetica', 10, 'bold', 'underline'))
        osLabel.grid(row=0, column=0, sticky='nw')
        
        # Initialize above/below pinch radio buttons
        self.pinchLoc = tk.StringVar()
        abPinch = ttk.Radiobutton(self, text='Above Pinch', variable=self.pinchLoc, value='top')
        blPinch = ttk.Radiobutton(self, text='Below Pinch', variable=self.pinchLoc, value='bottom')
        
        # Initialize heat limit entries
        htcLabel = ttk.Label(self, text='Heat Transfer Constraint', font=('TkDefaultFont', 9, 'italic', 'underline'))
        htcType = create_dropdown_menu(self, ['Upper Limit', 'Lower Limit'])
        #htcTypeL = ttk.Label(self, text='Type')
        htcHotL = ttk.Label(self, text='Hot Stream')
        htcColdL = ttk.Label(self, text='Cold Stream')
        htcEntryL = ttk.Label(self, text='Heat Transfer Limit')
        htcHot = ttk.Entry(self, width=12)
        htcCold = ttk.Entry(self, width=12)
        htcEntry = ttk.Entry(self, width=12)
        htcUnits = create_dropdown_menu(self, ['W', 'kcal/s', 'BTU/s'])
        htcButton = ttk.Button(self, text='Add Constraint', command=self.add_heat_limit)
        
        # Initialize forbidden/required matches
        frmLabel = ttk.Label(self, text='Stream Match Constraint', font=('TkDefaultFont', 9, 'italic', 'underline'))
        frmType = create_dropdown_menu(self, ['Forbidden', 'Required'])
        frmHotL = ttk.Label(self, text='Hot Stream')
        frmColdL = ttk.Label(self, text='Cold Stream')
        frmHot = ttk.Entry(self, width=12)
        frmCold = ttk.Entry(self, width=12)
        frmButton = ttk.Button(self, text='Add Constraint', command=self.add_spec_match)
        
        # Initialize 'Run HEN Optimization' button
        rhoButton = ttk.Button(self, text='Run HEN Optimization', command=self.run_optimization)
        
        # Arrange radio button widgets
        abPinch.grid(row=1, column=0, pady=(15,15))
        blPinch.grid(row=1, column=2, pady=(15,15))
        
        # Arrange heat transfer constraint widgets
        htcLabel.grid(row=3, column=0)
        htcHotL.grid(row=3, column=1, padx=10)
        htcColdL.grid(row=3, column=2, padx=10)
        htcEntryL.grid(row=3, column=3, padx=10)
        htcType[0].grid(row=4, column=0, padx=10)
        htcHot.grid(row=4, column=1, padx=10)
        htcCold.grid(row=4, column=2, padx=10)
        htcEntry.grid(row=4, column=3, padx=(10,0))
        htcUnits[0].grid(row=4, column=4, sticky='w', padx=(0,10))
        htcButton.grid(row=4, column=5, padx=10)
        
        # Assign values to heat transfer constraint input
        self.input_entries[str([4, 0])] = htcType[1]
        self.input_entries[str([4, 1])] = htcHot
        self.input_entries[str([4, 2])] = htcCold
        self.input_entries[str([4, 3])] = htcEntry
        self.input_entries[str([4, 4])] = htcUnits[1]
        
        # Arrange forbidden/required matches constraint widgets
        frmLabel.grid(row=6, column=0, pady=(25,0))
        frmHotL.grid(row=6, column=1, padx=10, pady=(25,0))
        frmColdL.grid(row=6, column=2, padx=10, pady=(25,0))
        frmType[0].grid(row=7, column=0, padx=10)
        frmHot.grid(row=7, column=1, padx=10)
        frmCold.grid(row=7, column=2, padx=10)
        frmButton.grid(row=7, column=5, padx=10)
        
        # Assign values to forbidden/required matches constraint input
        self.input_entries[str([7, 0])] = frmType[1]
        self.input_entries[str([7, 1])] = frmHot
        self.input_entries[str([7, 2])] = frmCold

        # Place 'Run HEN Optimization' button        
        self.columnconfigure(2, weight=1)
        rhoButton.grid(row=8, column=2, pady=(50, 0))
        
    def run_optimization(self):
        self.HEN_object_explorer.objectVisualizer.print2screen('Running HEN optimization method...', False)
        self.HEN_object.get_parameters()
        ucTree = self.HEN_uC_explorer.ucExplorer
        for constraint_type in ucTree.get_children():
            for constraint in ucTree.get_children([constraint_type]):
                dataVector = ucTree.item([constraint], 'values')
                hot_stream = dataVector[0]
                cold_stream = dataVector[1]
                hot_streamidx = self.HEN_object.streams.iloc[self.HEN_object.hot_streams].index.get_loc(hot_stream) + len(self.HEN_object.hot_utilities)
                cold_streamidx = self.HEN_object.streams.iloc[~self.HEN_object.hot_streams].index.get_loc(cold_stream) + len(self.HEN_object.cold_utilities)
                # For heat transfer limit constraints (upper/lower)
                if constraint_type == '0' or constraint_type == '1':
                    # Data sanitation for heat transfer limit units
                    heatlimitraw = dataVector[2].strip().split()
                    if heatlimitraw[1] == 'W':
                        heatlimit = float(heatlimitraw[0])*unyt.W
                    elif heatlimitraw[1] == 'kcal/s':
                        heatlimit = float(heatlimitraw[0])*unyt.cal/unyt.s
                    else:
                        heatlimit = float(heatlimitraw[0])*unyt.BTU/unyt.s
                    # Place heat limit constraint into associated matrix
                    if constraint_type == '0':
                        self.HEN_object.upper_limit[hot_streamidx, cold_streamidx] = heatlimit
                    elif constraint_type == '1':
                        self.HEN_object.lower_limit[hot_streamidx, cold_streamidx] = heatlimit
                # For stream matching constraints (forbidden/required)
                else:
                    # Place heat limit constraint into associated matrix
                    if constraint_type == '2':
                        self.HEN_object.forbidden[hot_streamidx, cold_streamidx] = True
                    elif constraint_type == '3':
                        self.HEN_object.required[hot_streamidx, cold_streamidx] = True
        
        # Check to ensure upper limit matrix is nonzero; if not, set to None
        if np.count_nonzero(self.HEN_object.upper_limit) == 0:
            self.HEN_object.upper_limit = None
        
        # NEED TO ADD CONSTRAINTS WITH UTILITIES
        
        [matchLocs, heatLocs] = self.HEN_object.place_exchangers(pinch = str(self.pinchLoc.get()), upper = self.HEN_object.upper_limit, lower = self.HEN_object.lower_limit, forbidden = self.HEN_object.forbidden, required = self.HEN_object.required)
        self.HEN_object_explorer.objectVisualizer.printsolutionmatrix(str(heatLocs))
    
    def add_heat_limit(self):
        errorFlag = False
        
        raw_input = []
        for col in range(5):
            rawdata = self.input_entries[str([4, col])].get()
            if rawdata == '': rawdata = None
            raw_input.append(rawdata)
        
        # Convert power units to unyt
        if raw_input[4] == 'W':
            raw_input[4] = unyt.W
        elif raw_input[4] == 'kcal/s':
            raw_input[4] = unyt.kcal/unyt.s
        else:
            raw_input[4] = unyt.BTU/unyt.s

        dataVec = [raw_input[1], raw_input[2], float(raw_input[3])*raw_input[4]]
        
        if raw_input[0] == 'Upper Limit':
            self.HEN_uC_explorer.ucExplorer.add_ul_constraint(dataVec)
        else:
            self.HEN_uC_explorer.ucExplorer.add_ll_constraint(dataVec)
        
        if errorFlag == False:
            for col in [1, 2, 3]:
                self.input_entries[str([4, col])].delete(0, 'end')
        
    def add_spec_match(self):
        errorFlag = False
        
        raw_input = []
        for col in range(3):
            rawdata = self.input_entries[str([7, col])].get()
            if rawdata == '': rawdata = None
            raw_input.append(rawdata)
        
        dataVec = [raw_input[1], raw_input[2]]
        
        if raw_input[0] == 'Forbidden':
            self.HEN_uC_explorer.ucExplorer.add_fm_constraint(dataVec)
        else:
            self.HEN_uC_explorer.ucExplorer.add_rm_constraint(dataVec)
            
        if errorFlag == False:
            for col in [1, 2]:
                self.input_entries[str([7, col])].delete(0, 'end')
        
class HENOS_user_constraints(ttk.Frame):
    def __init__(self, master):
        # Initialize frame properties
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Initialize user constraints label
        ucLabel = ttk.Label(self, text='User Constraints', font=('Helvetica', 10, 'bold', 'underline'))
        ucLabel.grid(row=0, column=0, sticky='nw')
        
        
        self.ucExplorer = HENOS_uC_tree(self)
        
        self.dcButton = ttk.Button(self, text='Delete Constraint', command=self.ucExplorer.delete_constraint)
        self.adcButton = ttk.Button(self, text='Activate/Deactivate Constraint')
        
        self.columnconfigure(1, weight=1)
        self.dcButton.grid(row=0, column=3)
        self.adcButton.grid(row=0, column=2)
        
        #
        
        
        
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        
        self.ucExplorer.grid(row=1, rowspan=40, column=0, columnspan=20, padx=5, pady=(5,5), sticky='nsew')
    
class HENOS_uC_tree(ttk.Treeview):
    '''
    A class which holds the Treeview object which forms the basis of the
    object explorer. Slave of HENOS_object_explorer
    '''
    def __init__(self, master):
        # Initialize treeview properties
        ttk.Treeview.__init__(self, master, show='tree', selectmode='none')
        style = ttk.Style()
        style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nswe'})])
        self['columns'] = ('1', '2', '3')
        self.column('1', width=150)
        self.column('2', width=150)
        self.column('3', width=150)
        
        # Defining variables
        self.master = master
        
        # Intialize vertical scrollbar
        self.verscrlbar = ttk.Scrollbar(self, orient='vertical', command=self.yview)
        self.verscrlbar.pack(side='right', fill='y')
        self.configure(yscrollcommand=self.verscrlbar.set)
        
        # Intialize object classes in treeview
        self.ulNode = self.insert('', index=0, iid=0, text='UPPER LIMIT', values=('Hot Stream', 'Cold Stream', 'Upper Heat Limit'))
        self.llNode = self.insert('', index=2, iid=1, text='LOWER LIMIT', values=('Hot Stream', 'Cold Stream', 'Lower Heat Limit'))
        self.fmNode = self.insert('', index=2, iid=2, text='FORBIDDEN MATCHES', values=('Hot Stream', 'Cold Stream'))
        self.rmNode = self.insert('', index=3, iid=3, text='REQUIRED MATCHES', values=('Hot Stream', 'Cold Stream'))
        
        # Initialize 'Single Click' Event (Show Selected Object in Object Explorer)
        self.bind('<Button-1>', self.on_click)
        #self.bind("<Double-Button-1>", self.send2screen)
        
    def on_click(self, event):
        tree = event.widget
        item_name = tree.identify_row(event.y)
        if item_name:
            tags = tree.item(item_name, 'tags')
            if tags and (tags[0] == 'selectable'):
                tree.selection_set(item_name)
    
    def add_ul_constraint(self, constraint_data):
        self.insert(self.ulNode, 'end', text='', values=(str(constraint_data[0]), str(constraint_data[1]), str(constraint_data[2])), tags='selectable')
        
    def add_ll_constraint(self, constraint_data):
        self.insert(self.llNode, 'end', text='', values=(str(constraint_data[0]), str(constraint_data[1]), str(constraint_data[2])), tags='selectable')
        
    def add_fm_constraint(self, constraint_data):
        self.insert(self.fmNode, 'end', text='', values=(str(constraint_data[0]), str(constraint_data[1])), tags='selectable')
        
    def add_rm_constraint(self, constraint_data):
        self.insert(self.rmNode, 'end', text='', values=(str(constraint_data[0]), str(constraint_data[1])), tags='selectable')
    
    def delete_constraint(self):
        HEN_selectedConstraint  = self.selection()[0]
        HEN_sC_name = self.item(HEN_selectedConstraint, 'text')
        self.delete(HEN_selectedConstraint)
        
# FUNCTIONS
def create_dropdown_menu(master, options):
    var = tk.StringVar(master)
    menu = ttk.OptionMenu(master, var, options[0], *options)
    return [menu, var]

##############################################################################
# RUN APPLICATION
##############################################################################
root = tk.Tk()

if __name__ == '__main__':
    HENOS = HENOS_app(root)
    HENOS.master.title('HENOS')
    root.mainloop()
