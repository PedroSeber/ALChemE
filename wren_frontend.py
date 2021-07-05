###############################################################################
# IMPORT CALLS
##############################################################################
import numpy as np
import unyt
from collections import namedtuple, OrderedDict
import tkinter as tk
from tkinter import ttk
from wren_design import WReN
import subprocess
import pdb

##############################################################################
# CLASSES
##############################################################################
class WReN_GUI_app():
    '''
    A class which holds the HEN_GUI application. Slave of root window.
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
        self.WReN_GUI_dropdown_menu = tk.Menu(top)
        top['menu'] = self.WReN_GUI_dropdown_menu
        
        # Initialize tab system
        self.tabControl = ttk.Notebook(self.master, width=swidth, height=sheight)
        
        # Intialize control panel
        control_panel_Tab = ttk.Frame(self.tabControl)
        self.tabControl.add(control_panel_Tab, text='WReN_GUI Control Panel')
        self.tabControl.pack(expand=1, fill='both')
        
        # Initialize WReN object
        self.WReN_object = WReN()
        
        # Initialize control panel elements
        self.WReN_GUI_object_explorer = WReN_object_explorer(control_panel_Tab, self.WReN_object)
        self.WReN_GUI_input = WReN_input(control_panel_Tab, self.WReN_object, self.WReN_GUI_object_explorer)
        self.WReN_GUI_user_constraints = WReN_constraint_explorer(control_panel_Tab)
        self.WReN_GUI_optimization_suite = WReN_optimization_suite(control_panel_Tab, self.WReN_object, self.WReN_GUI_object_explorer, self.WReN_GUI_user_constraints)
        
        # Placing control panel elements
        control_panel_Tab.rowconfigure(2, weight=1)
        control_panel_Tab.columnconfigure(0, weight=1)
        control_panel_Tab.columnconfigure(9, weight=1)
        
        self.WReN_GUI_input.grid(row=0, rowspan=1, column=0, columnspan=9, sticky='nsew')
    
        self.WReN_GUI_optimization_suite.grid(row=0, rowspan=2, column=9,  sticky='nsew')
        
        self.WReN_GUI_object_explorer.grid(row=1, column=0, rowspan=40, columnspan=8, sticky='nsew')
        self.WReN_GUI_user_constraints.grid(row=2, column=9, rowspan=3, sticky='nsew')
        
class WReN_input(ttk.Frame):
    '''
    A class which holds the WReN_GUI stream input. Slave of WReN_GUI_app.    
    '''
    def __init__(self, master, WReN_object, WReN_object_explorer):        
        # Initialize frame properties
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Initialize Input Label
        piLabel = ttk.Label(self, text='Process Input', font=('Helvetica', 10, 'bold', 'underline'))
        piLabel.grid(row=0, column=0, sticky='w')
        
        # Defining variables
        self.WReN_object = WReN_object
        self.WReN_oeTree = WReN_object_explorer.objectExplorer
        self.WReN_stream_labels = ['Process Name', 'Sink Concentration',
                             'Source Concentration', '', 'Contaminants',
                             'Sink Flow', 'Source Flow','']
        self.input_entries = {}
        
        # Arrange process input components
        for row in range(1,3):
            for col in range(8):
                if row == 1 and col in [0, 1, 2, 4, 5, 6]:
                    l = ttk.Label(self, text=self.WReN_stream_labels[col])
                    l.grid(row=row, column=col, padx=10)
                elif row == 1 and col in [3, 7]:
                    l = ttk.Label(self, width=12)
                    l.grid(row=row, column=col, padx=10)
                else:
                    if col in [0, 1, 2, 4, 5, 6]:
                        e = ttk.Entry(self, width=12)
                        e.grid(row=row, column=col)
                        self.input_entries[str([row, col])] = e
                    elif col == 3:
                        m = create_dropdown_menu(self, ['mg/kg', 'ppm'])
                        m[0].grid(row = row, column=col, sticky='w')
                        self.input_entries[str([row, col])] = m[1]
                    elif col == 7:
                        m = create_dropdown_menu(self, ['kg/s', 'lb/s', 'J/(kg·K)'])
                        m[0].grid(row = row, column=col, sticky='w')
                        self.input_entries[str([row, col])] = m[1]
        
        # Initialize and arrange 'Add Process' button
        sub_stream = ttk.Button(self, text="Add Process", command=self.add_process)
        sub_stream.grid(row=2, column=9, sticky='nsew')
        
    def add_process(self):
        # Populating raw input data vector
        raw_input = []
        for col in [0, 1, 2, 3, 4, 5, 6, 7]:
            rawdata = self.input_entries[str([2, col])].get()
            if rawdata == '': rawdata = None
            raw_input.append(rawdata)
            if col in [0, 1, 2, 4, 5, 6]:
                self.input_entries[str([2, col])].delete(0, 'end')

        # Check if process name is None
        if raw_input[0] is not None:
            processname = rawdata[0]
        else:
            processname = None
        
        # Split sink/source concentrations
        sinkconcRaw = raw_input[1].strip().split(",")
        sourceconcRaw = raw_input[2].strip().split(",")
        
        # Split contaminants if not None
        if raw_input[4] is not None:
            contaminants = raw_input[4].strip().split(",")
        else:
            contaminants = None
    
        # Convert sink/source concentrations to float
        sinkconc = [float(jj) for jj in sinkconcRaw]
        sourceconc = [float(kk) for kk in sourceconcRaw]
        
        # Convert sink flow rate to float
        sinkflow = float(raw_input[5])
        
        # Convert source flow rate to float if not None
        if raw_input[6] is not None:
            sourceflow = float(raw_input[6])
        else:
            sourceflow = None
        
        # Convert concentration units to unyt
        if raw_input[3] == 'mg/kg':
            conc_unit = unyt.mg/unyt.kg
        
        # Convert flow rate units to unyt
        if raw_input[-1] == 'kg/s':
            flow_unit = unyt.kg/unyt.s
        
        # Pass values to HEN object
        self.WReN_object.add_process(sinkconc, sourceconc, sinkflow, sourceflow, processname, conc_unit, flow_unit, contaminants, self.WReN_oeTree)

class WReN_object_explorer(ttk.Frame):
    '''
    A class which holds the WReN_GUI object explorer and visualizer. Slave of
    WReN_GUI_app.
    '''
    def __init__(self, master, WReN_object):
        # Initialize frame properties
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Define variables
        self.WReN_object = WReN_object
        
        # Initialize Object Explorer Label
        oeLabel = ttk.Label(self, text='Object Explorer', font=('Helvetica', 10, 'bold', 'underline'))
        oeLabel.grid(row=0, column=0, sticky='w')
        
        
        # Initialize object visualizer label
        tLabel = ttk.Label(self, text='Terminal Display', font=('Helvetica', 10, 'bold', 'underline'))
        tLabel.grid(row=41, column=0, sticky='w')
        
        # Initialize object explorer        
        self.objectExplorer = WReN_objE_tree(self, self.WReN_object)
        
        # Initialize object visualizer
        self.objectTerminal = WReN_objE_terminal(self, self.WReN_object)
        
        # Place object explorer and terminal
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(41, weight=0)
        self.rowconfigure(42, weight=1)
        self.objectExplorer.grid(row=1, column=0, rowspan=40, columnspan=8, padx=5, pady=(5,15), sticky='nsew')
        self.objectTerminal.grid(row=42, column=0, rowspan=1, columnspan=8, padx=5, pady=(5,5), sticky='nsew')
        
        
class WReN_objE_tree(ttk.Treeview):
    '''
    A class which holds the Treeview object which forms the basis of the
    object explorer. Slave of WReN_GUI_object_explorer
    '''
    def __init__(self, master, WReN_object):
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
        self.WReN_object = WReN_object
        self.master = master
        
        # Intialize vertical scrollbar
        self.verscrlbar = ttk.Scrollbar(self, orient='vertical', command=self.yview)
        self.verscrlbar.pack(side='right', fill='y')
        self.configure(yscrollcommand=self.verscrlbar.set)
        
        # Intialize object classes in treeview
        self.ProcessNode = self.insert('', index=0, iid=0, text='PROCESSES', values=('Sink Concentrations', 'Source Concentrations', 'Sink Flow Rate', 'Source Flow Rate', 'Status'))
        self.SolutionNode = self.insert('',index=1, iid=1, text='SOLUTION', values=('Cost'))
        
    def receive_new_process(self, oeDataVector):
        self.insert(self.ProcessNode, 'end', text=oeDataVector[0], values=(oeDataVector[1], oeDataVector[2], str(oeDataVector[3]), str(oeDataVector[4]), 'Active'), tags='selectable')
    
class WReN_objE_terminal(tk.Text):
    def __init__(self, master, WReN_object):
        # Initialize text properties
        tk.Text.__init__(self, master, highlightthickness=0)
        
        # Defining variables
        self.WReN_object = WReN_object
        
        # Initialize vertical scrollbar
        self.verscrlbar = ttk.Scrollbar(self, orient='vertical', command=self.yview)
        self.verscrlbar.pack(side='right', fill='y')
        self.configure(yscrollcommand=self.verscrlbar.set)
    
        # Initialize >>>
        self.insert('end', '-'*48+ '***INITIALIZED***' + '-'*48  + '\n\n')
        self.insert('end', '>>> ')
        
    def print2screen(self, message, newcommand):
        self.insert('end', message + '\n')
        if newcommand == True:
            self.insert('end', '>>> ')

class WReN_optimization_suite(ttk.Frame):
    def __init__(self, master, WReN_object, WReN_object_explorer, WReN_constraint_explorer):
        # Intialize fram properties
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Initialize optimization suite label
        osLabel = ttk.Label(self, text='Optimization Suite', font=('Helvetica', 10, 'bold', 'underline'))
        osLabel.grid(row=0, column=0, sticky='nw')
        
        # Define variables
        self.WReN_object = WReN_object
        self.WReN_constraint_explorer = WReN_constraint_explorer
        self.WReN_terminal_display = WReN_object_explorer.objectTerminal
        self.input_entries = {}
        
        # Initialize material exchange limit entries
        melLabel = ttk.Label(self, text='Flow Rate Constraint', font=('TkDefaultFont', 9, 'italic', 'underline'))
        melType = create_dropdown_menu(self, ['Upper Limit', 'Lower Limit'])
        #htcTypeL = ttk.Label(self, text='Type')
        melSourceL = ttk.Label(self, text='Source')
        melSinkL = ttk.Label(self, text='Sink')
        melEntryL = ttk.Label(self, text='Heat Transfer Limit')
        melSource = ttk.Entry(self, width=12)
        melSink = ttk.Entry(self, width=12)
        melEntry = ttk.Entry(self, width=12)
        melUnits = create_dropdown_menu(self, ['kg/s'])
        melButton = ttk.Button(self, text='Add Constraint', command=self.add_flow_limit)
        
        # Initialize forbidden/required matches
        frmLabel = ttk.Label(self, text='Match Constraint', font=('TkDefaultFont', 9, 'italic', 'underline'))
        frmType = create_dropdown_menu(self, ['Forbidden', 'Required'])
        frmSourceL = ttk.Label(self, text='Source')
        frmSinkL = ttk.Label(self, text='Sink')
        frmSource = ttk.Entry(self, width=12)
        frmSink = ttk.Entry(self, width=12)
        frmButton = ttk.Button(self, text='Add Constraint', command=self.add_spec_match)
        
        # Initialize source/sink process cost per flow setting
        sspcLabel = ttk.Label(self, text='Cost Matrix Match', font=('TkDefaultFont', 9, 'italic', 'underline'))
        sspcSourceL = ttk.Label(self, text='Source')
        sspcSinkL = ttk.Label(self, text='Sink')
        sspcCPFL = ttk.Label(self, text='Cost Per Flow')
        sspcSource = ttk.Entry(self, width=12)
        sspcSink = ttk.Entry(self, width=12)
        sspcCPF = ttk.Entry(self, width=12)
        sspcUnits = create_dropdown_menu(self, ['$/(kg/s)'])
        sspcButton = ttk.Button(self, text='Add Cost', command=self.add_sspc_match)
        
        # Initialize general cost per flow setting
        gcLabel = ttk.Label(self, text='General Cost Matrix', font=('TkDefaultFont', 9, 'italic', 'underline'))
        gcCFPL = ttk.Label(self, text='Cost Per Flow')
        gcCFP = ttk.Entry(self, width=12)
        gcUnits = create_dropdown_menu(self, ['$/(kg/s)'])
        gcFSButton = ttk.Button(self, text='          Add Cost\n(Freshwater Sources)', command=self.add_pc_freshwatersources)
        gcWSButton = ttk.Button(self, text='         Add Cost\n(Wastewater Sinks)', command=self.add_pc_wastewatersinks)
        gcRButton = ttk.Button(self, text='    Add Cost\n(All Recycles)', command=self.add_pc_recycles)
        
        # Initialize 'Run WReN Optimization' button
        rwoButton = ttk.Button(self, text='Run WReN Optimization', command=self.run_optimization)
        
        # Arrange flow constraint widgets
        melLabel.grid(row=3, column=0)
        melSourceL.grid(row=3, column=1, padx=10)
        melSinkL.grid(row=3, column=2, padx=10)
        melEntryL.grid(row=3, column=3, padx=10)
        melType[0].grid(row=4, column=0, padx=10)
        melSource.grid(row=4, column=1, padx=10)
        melSink.grid(row=4, column=2, padx=10)
        melEntry.grid(row=4, column=3, padx=(10,0))
        melUnits[0].grid(row=4, column=4, sticky='w', padx=(0,10))
        melButton.grid(row=4, column=5, padx=10)
        
        # Assign values to heat transfer constraint input
        self.input_entries[str([4, 0])] = melType[1]
        self.input_entries[str([4, 1])] = melSource
        self.input_entries[str([4, 2])] = melSink
        self.input_entries[str([4, 3])] = melEntry
        self.input_entries[str([4, 4])] = melUnits[1]
        
        # Arrange forbidden/required matches constraint widgets
        frmLabel.grid(row=6, column=0, pady=(25,0))
        frmSourceL.grid(row=6, column=1, padx=10, pady=(25,0))
        frmSinkL.grid(row=6, column=2, padx=10, pady=(25,0))
        frmType[0].grid(row=7, column=0, padx=10)
        frmSource.grid(row=7, column=1, padx=10)
        frmSink.grid(row=7, column=2, padx=10)
        frmButton.grid(row=7, column=5, padx=10)
        
        # Assign values to forbidden/required matches constraint input
        self.input_entries[str([7, 0])] = frmType[1]
        self.input_entries[str([7, 1])] = frmSource
        self.input_entries[str([7, 2])] = frmSink
        
        # Arrange source/sink process cost per flow setting widgets
        sspcLabel.grid(row=8, column=0, pady=(25,0))
        sspcSourceL.grid(row=9, column=0, padx=10)
        sspcSinkL.grid(row=9, column=1, padx=10)
        sspcCPFL.grid(row=9, column=2, padx=10)
        sspcSource.grid(row=10, column=0, padx=10)
        sspcSink.grid(row=10, column=1, padx=10)
        sspcCPF.grid(row=10, column=2, padx=(10,0))
        sspcUnits[0].grid(row=10, column=3, padx=(0,10))
        sspcButton.grid(row=10, column=4, padx=10)
        
        # Assign values to source/sink process cost per flow
        self.input_entries[str([10,0])] = sspcSource
        self.input_entries[str([10,1])] = sspcSink
        self.input_entries[str([10,2])] = sspcCPF
        self.input_entries[str([10,3])] = sspcUnits[1]
        
        # Arrange general cost per flow setting
        gcLabel.grid(row=11, column=0, pady=(25,0))
        gcCFPL.grid(row=12, column=0)
        gcCFP.grid(row=13, column=0, padx=(10,0))
        gcUnits[0].grid(row=13, column=1, padx=(0,10))
        gcFSButton.grid(row=13, column=2, padx=10)
        gcWSButton.grid(row=13, column=3, padx=10)
        gcRButton.grid(row=13, column=4, padx=10)
        
        # Assign values to general cost per flow
        self.input_entries[str([13,0])] = gcCFP
        self.input_entries[str([13,1])] = gcUnits[1]
        
        # Place 'Run WReN Optimization' button        
        self.columnconfigure(2, weight=1)
        rwoButton.grid(row=14, column=2, pady=(25,20))
    
    def add_flow_limit(self):
        raw_input = []
        for col in range(5):
            rawdata = self.input_entries[str([4, col])].get()
            if rawdata == '': rawdata = None
            raw_input.append(rawdata)
        
        # Assign variables to processes
        lcSink = raw_input[1]
        lcSource = raw_input[2]
        
        # Obtain flow rate limit
        lcLimit = float(raw_input[3])
        
        # Convert flow units to unyt
        if raw_input[-1] == 'kg/s':
            lcUnit = unyt.kg/unyt.s
        
        dataVec = [lcSink, lcSource, str(lcLimit*lcUnit)]
        
        if raw_input[0] == 'Upper Limit':
            self.WReN_constraint_explorer.ucExplorer.add_ul_constraint(dataVec)
        else:
            self.WReN_constraint_explorer.ucExplorer.add_ll_constraint(dataVec)
        
        for col in [1, 2]:
            self.input_entries[str([4, col])].delete(0, 'end')
    
    def add_spec_match(self):
        raw_input = []
        for col in range(3):
            rawdata = self.input_entries[str([7,col])].get()
            if rawdata == '': rawdata = None
            raw_input.append(rawdata)
        
        # Assign variables to processes
        smSink = raw_input[1]
        smSource = raw_input[2]
        
        dataVec = [smSink, smSource]
        
        if raw_input[0] == 'Forbidden':
            self.WReN_constraint_explorer.ucExplorer.add_fm_constraint(dataVec)
        else:
            self.WReN_constraint_explorer.ucExplorer.add_rm_constraint(dataVec)
        
        for col in [1, 2]:
            self.input_entries[str([7, col])].delete(0, 'end')
    
    def add_sspc_match(self):
        # Retrieve data
        raw_input = []
        for col in range(4):
            rawdata = self.input_entries[str([10, col])].get()
            if rawdata == '': rawdata = None
            raw_input.append(rawdata)
        
        #Assign variables to processes
        sspcSource = raw_input[0]
        sspcSink = raw_input[1]
        
        # Convert cost per flow into float
        sspcCost = float(raw_input[2])
        
        for col in [0, 1, 2]:
            self.input_entries[str([10, col])].delete(0, 'end')
        
        # Modify match
        self.WReN_object.costs.at[sspcSource,sspcSink] = sspcCost
        
        print(self.WReN_object.costs)
        
    def add_pc_freshwatersources(self):
        # Retrieve data
        gcCost = float(self.input_entries[str([13,0])].get())
        
        # Set all FW sources to gcCost
        self.WReN_object.costs.iloc[0,:] = gcCost
        
        print(self.WReN_object.costs)
        
        self.input_entries[str([13, 0])].delete(0, 'end')
        
    def add_pc_wastewatersinks(self):
        # Retrieve data
        gcCost = float(self.input_entries[str([13,0])].get())
        
        # Set all WW sources to gcCost
        self.WReN_object.costs.iloc[:,0] = gcCost
        
        print(self.WReN_object.costs)

        self.input_entries[str([13, 0])].delete(0, 'end')

    def add_pc_recycles(self):
        # Retrieve data
        gcCost = float(self.input_entries[str([13,0])].get())
        
        # Set all recycles to 
        self.WReN_object.costs.iloc[1:,1:] = gcCost
        
        print(self.WReN_object.costs)

        self.input_entries[str([13, 0])].delete(0, 'end')
    
    def run_optimization(self):
        self.WReN_terminal_display.print2screen('Running WReN optimization...', False)
        self.WReN_object.solve_WReN(self.WReN_object.costs, self.WReN_object.upper, self.WReN_object.lower)
        self.WReN_terminal_display.print2screen(f'Cost: ${self.WReN_object.results.loc["cost"].sum().sum():.2f}\n', False)
        self.WReN_terminal_display.print2screen(f'Solution Match Matrix (Flow Rate in {self.WReN_object.flow_unit})', False)
        self.WReN_terminal_display.print2screen(str(self.WReN_object.results.loc['flows']) + '\n', False)
        self.WReN_terminal_display.print2screen('Solution Match Matrix (Cost in $)', False)
        self.WReN_terminal_display.print2screen(str(self.WReN_object.results.loc['cost']) + '\n', False)
        self.WReN_terminal_display.print2screen('Solver has finished.', True)

class WReN_constraint_explorer(ttk.Frame):
    def __init__(self, master):
        # Initialize frame properties
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Initialize user constraints label
        ucLabel = ttk.Label(self, text='User Constraints', font=('Helvetica', 10, 'bold', 'underline'))
        ucLabel.grid(row=0, column=0, sticky='nw')
        
        self.ucExplorer = WReN_cE_tree(self)
        self.dcButton = ttk.Button(self, text='Delete Constraint')
        
        self.columnconfigure(1, weight=1)
        self.dcButton.grid(row=0, column=3)
    
        
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        
        self.ucExplorer.grid(row=1, rowspan=40, column=0, columnspan=20, padx=5, pady=(5,5), sticky='nsew')
        

class WReN_cE_tree(ttk.Treeview):
    '''
    A class which holds the Treeview object which forms the basis of the
    object explorer. Slave of WReN_GUI_object_explorer
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
        self.ulNode = self.insert('', index=0, iid=0, text='UPPER LIMIT', values=('Source', 'Sink', 'Upper Flow Rate Limit'))
        self.llNode = self.insert('', index=2, iid=1, text='LOWER LIMIT', values=('Source', 'Sink', 'Lower Flow Rate Limit'))
        self.fmNode = self.insert('', index=2, iid=2, text='FORBIDDEN MATCHES', values=('Source', 'Sink'))
        self.rmNode = self.insert('', index=3, iid=3, text='REQUIRED MATCHES', values=('Source', 'Sink'))
        self.cmNode = self.insert('', index=4, iid=4, text='COST MATRIX')

        # Initialize 'Single Click' Event (Show Selected Object in Object Explorer)
        self.bind('<Button-1>', self.on_click)

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

# FUNCTIONS
def create_dropdown_menu(master, options):
    var = tk.StringVar(master)
    menu = ttk.OptionMenu(master, var, options[0], *options)
    return [menu, var]
