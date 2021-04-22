##############################################################################
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
        sub_stream = ttk.Button(self, text="Add Process")
        sub_stream.grid(row=2, column=9, sticky='nsew')

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
        self.ProcessNode = self.insert('', index=0, iid=0, text='PROCESSES', values=('Source Concentration', 'Sink Concentration', 'Sink Flow Rate', 'Status'))
        self.SolutionNode = self.insert('',index=1, iid=1, text='SOLUTION', values=('Cost'))
        
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

class WReN_optimization_suite(ttk.Frame):
    def __init__(self, master, WReN_object, WReN_object_explorer, WReN_constraint_explorer):
        # Intialize fram properties
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Initialize optimization suite label
        osLabel = ttk.Label(self, text='Optimization Suite', font=('Helvetica', 10, 'bold', 'underline'))
        osLabel.grid(row=0, column=0, sticky='nw')
        
        # Define variables
        self.input_entries = {}
        
        # Initialize material exchange limit entries
        melLabel = ttk.Label(self, text='Material Exchange Constraint', font=('TkDefaultFont', 9, 'italic', 'underline'))
        melType = create_dropdown_menu(self, ['Upper Limit', 'Lower Limit'])
        #htcTypeL = ttk.Label(self, text='Type')
        melSourceL = ttk.Label(self, text='Source')
        melSinkL = ttk.Label(self, text='Sink')
        melEntryL = ttk.Label(self, text='Heat Transfer Limit')
        melSource = ttk.Entry(self, width=12)
        melSink = ttk.Entry(self, width=12)
        melEntry = ttk.Entry(self, width=12)
        melUnits = create_dropdown_menu(self, ['mg/s'])
        melButton = ttk.Button(self, text='Add Constraint')
        
        # Initialize forbidden/required matches
        frmLabel = ttk.Label(self, text='Match Constraint', font=('TkDefaultFont', 9, 'italic', 'underline'))
        frmType = create_dropdown_menu(self, ['Forbidden', 'Required'])
        frmSourceL = ttk.Label(self, text='Source')
        frmSinkL = ttk.Label(self, text='Sink')
        frmSource = ttk.Entry(self, width=12)
        frmSink = ttk.Entry(self, width=12)
        frmButton = ttk.Button(self, text='Add Constraint')
        
        # Initialize 'Run HEN Optimization' button
        rwoButton = ttk.Button(self, text='Run WReN Optimization')
        
        # Arrange heat transfer constraint widgets
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
        
        # Place 'Run WReN Optimization' button        
        self.columnconfigure(2, weight=1)
        rwoButton.grid(row=10, column=2, pady=(25,20))

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
        self.ulNode = self.insert('', index=0, iid=0, text='UPPER LIMIT', values=('Source', 'Sink', 'Upper Transfer Limit'))
        self.llNode = self.insert('', index=2, iid=1, text='LOWER LIMIT', values=('Source', 'Sink', 'Lower Transfer Limit'))
        self.fmNode = self.insert('', index=2, iid=2, text='FORBIDDEN MATCHES', values=('Source', 'Sink'))
        self.rmNode = self.insert('', index=3, iid=3, text='REQUIRED MATCHES', values=('Source', 'Sink'))

# FUNCTIONS
def create_dropdown_menu(master, options):
    var = tk.StringVar(master)
    menu = ttk.OptionMenu(master, var, options[0], *options)
    return [menu, var]









