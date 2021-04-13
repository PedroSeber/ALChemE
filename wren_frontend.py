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
        control_panel_Tab.columnconfigure(0, weight=1)
        control_panel_Tab.columnconfigure(9, weight=1)
        
        self.WReN_GUI_input.grid(row=0, rowspan=2, column=0, columnspan=9, sticky='nsew')
    
        self.WReN_GUI_optimization_suite.grid(row=0, rowspan=2, column=9,  sticky='nsew')
        
        self.WReN_GUI_object_explorer.grid(row=2, column=0, rowspan=40, columnspan=8, sticky='nsew')
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

class WReN_object_explorer(ttk.Frame):
    '''
    A class which holds the WReN_GUI object explorer and visualizer. Slave of
    HEN_GUI_app.
    '''
    def __init__(self, master, WReN_object):
        # Initialize frame properties
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Initialize Object Explorer Label
        oeLabel = ttk.Label(self, text='Object Explorer', font=('Helvetica', 10, 'bold', 'underline'))
        oeLabel.grid(row=0, column=0, sticky='w')
        
        
        # Initialize object visualizer label
        tLabel = ttk.Label(self, text='Terminal Display', font=('Helvetica', 10, 'bold', 'underline'))
        tLabel.grid(row=41, column=0, sticky='w')
        
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
        
class WReN_objE_terminal(tk.Text):
    def __init__(self, master, WReN_object):
        # Initialize text properties
        tk.Text.__init__(self, master, highlightthickness=0)

class WReN_optimization_suite(ttk.Frame):
    def __init__(self, master, WReN_object, WReN_object_explorer, WReN_constraint_explorer):
        # Intialize fram properties
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Initialize optimization suite label
        osLabel = ttk.Label(self, text='Optimization Suite', font=('Helvetica', 10, 'bold', 'underline'))
        osLabel.grid(row=0, column=0, sticky='nw')

class WReN_constraint_explorer(ttk.Frame):
    def __init__(self, master):
        # Initialize frame properties
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Initialize user constraints label
        ucLabel = ttk.Label(self, text='User Constraints', font=('Helvetica', 10, 'bold', 'underline'))
        ucLabel.grid(row=0, column=0, sticky='nw')

class WReN_cE_tree(ttk.Treeview):
    '''
    A class which holds the Treeview object which forms the basis of the
    object explorer. Slave of HEN_GUI_object_explorer
    '''
    def __init__(self, master):
        # Initialize treeview properties
        ttk.Treeview.__init__(self, master, show='tree', selectmode='none')
        style = ttk.Style()
        style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nswe'})])











