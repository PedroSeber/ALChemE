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
        
        # Intialize dropdown menu options
        self.fileMenu = tk.Menu(self.HENOS_dropdown_menu, tearoff=0)
        self.fileMenu.add_command(label='New')
        self.fileMenu.add_command(label='Open')
        self.fileMenu.add_command(label='Save')
        self.fileMenu.add_command(label='Save As')
        self.HENOS_dropdown_menu.add_cascade(label='File', menu=self.fileMenu)
        self.HENOS_dropdown_menu.add_cascade(label='Settings')
        
        # Intialize control panel
        control_panel_Tab = ttk.Frame(self.tabControl)
        self.tabControl.add(control_panel_Tab, text='HENOS Control Panel')
        self.tabControl.pack(expand=1, fill='both')
        
        # Initialize HEN object
        HEN_object = HEN()
        
        # Initialize control panel elements
        self.HENOS_object_explorer = HENOS_object_explorer(control_panel_Tab, HEN_object)
        self.HENOS_input = HENOS_input(control_panel_Tab, HEN_object, self.HENOS_object_explorer)
        self.HENOS_ga_frame = HENOS_graphical_analysis_controls(control_panel_Tab, self.tabControl, self.HENOS_object_explorer, HEN_object)
        self.HENOS_os_frame = HENOS_optimization_controls(control_panel_Tab)
        self.HENOS_uc_frame = HENOS_user_constraints(control_panel_Tab)
        
        # Placing control panel elements
        control_panel_Tab.rowconfigure(1, weight=1)
        control_panel_Tab.rowconfigure(2, weight=1)
        control_panel_Tab.columnconfigure(9, weight=1)
        control_panel_Tab.columnconfigure(11, weight=1)
        self.HENOS_input.grid(row=0, column=0)
        self.HENOS_object_explorer.grid(row=1, column=0, rowspan=40, columnspan=8, sticky='nsew')
        self.HENOS_ga_frame.grid(row=0, rowspan=2, column=9, columnspan=2, sticky='nsew')
        self.HENOS_os_frame.grid(row=0, rowspan=2, column=11, columnspan=2, sticky='nsew')
        self.HENOS_uc_frame.grid(row=2, column=9, columnspan = 4, sticky='nsew')
        # test = ttk.Button(control_panel_Tab, text='test')
        # test.grid(row=41, column=0, columnspan=8, sticky='ew')

class HENOS_input(ttk.Frame):
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
                             'Flow Rate', '', 'Heat Load', '']
        self.input_entries = {}
        
        # Initialize Input Label
        name = ttk.Label(self, text='User Input', font=('Helvetica', 10, 'bold', 'underline'))
        name.grid(row=0, column=0, sticky='w')
        
        
        # Arrange stream input components
        for row in range(1,3):
            for col in range(10):
                if row == 1 and col in [0, 1, 2, 4, 6, 8]:
                    l = ttk.Label(self, text=self.HEN_stream_labels[col])
                    l.grid(row=row, column=col, padx=10)
                elif row == 1 and col in [3, 5, 7, 9]:
                    l = ttk.Label(self, width=12)
                    l.grid(row=row, column=col, padx=10)
                else:
                    if col in [0, 1, 2, 4, 6, 8]:
                        e = ttk.Entry(self, width=12)
                        e.grid(row=row, column=col)
                        self.input_entries[col] = e
                    elif col == 3:
                        m = create_dropdown_menu(self, ['°C', '°F', '°K'])
                        m[0].grid(row = row, column=col)
                        self.input_entries[col] = m[1]
                    elif col == 5:
                        m = create_dropdown_menu(self, ['J/(kg·°C)', 'BTU/(lb·°F)', 'J/(kg·°K)'])
                        m[0].grid(row = row, column=col)
                        self.input_entries[col] = m[1]
                    elif col == 7:
                        m = create_dropdown_menu(self, ['kg/s', 'lb/s'])
                        m[0].grid(row = row, column=col)
                        self.input_entries[col] = m[1]
                    elif col == 9:
                        m = create_dropdown_menu(self, ['W', 'kcal/s', 'BTU/s'])
                        m[0].grid(row = row, column=col)
                        self.input_entries[col] = m[1]
        
        # Initialize and arrange 'Add Stream' button
        sub_stream = ttk.Button(self, text="Add Stream", command=self.add_stream)
        sub_stream.grid(row=2, column=10, sticky='nsew')
    
    def add_stream(self):
        # Populating raw input data vector
        raw_input = []
        for col in range(10):
            rawdata = self.input_entries[col].get()
            if rawdata == '': rawdata = None
            raw_input.append(rawdata)
            if col in [0, 1, 2, 4, 6, 8]:
                self.input_entries[col].delete(0, 'end')
        
        print(raw_input)
        
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
        self.HEN_object.add_stream(t1 = raw_input[1], t2 = raw_input[2], cp = raw_input[4], flow_rate = raw_input[6], heat = raw_input[8], stream_name = raw_input[0], HENOS_oe_tree = self.HEN_object_explorer.objectExplorer, temp_unit = self.temp_unit)    

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
        self.insert('end', '-'*54 + '***INITIALIZED***' + '-'*56 + '\n\n')
        self.insert('end', '>>> ')
        #self.config(state='disabled')
    
    def print2screen(self, object_name):
        if object_name not in ['STREAMS', 'HEAT EXCHANGERS', 'UTILITIES']:
            #self.config(state='normal')
            commandtext = str('displaying object ' + object_name + '...\n')
            self.insert('end', commandtext)
            displaytext = str(self.HEN_object.streams[object_name])
            self.insert('end', displaytext + '\n\n')
            self.insert('end', '>>> ')
            self.see('end')
            #self.config(state='disabled')

    def clearscreen(self):
        self.delete('1.0', 'end')
        self.insert('end', '-'*54 + '***INITIALIZED***' + '-'*56 + '\n\n')
        self.insert('end', '>>> ')    

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
        self.HXNode = self.insert('', index=1, iid=1, text='HEAT EXCHANGERS')
        self.UtilityNode = self.insert('', index=2, iid=2, text='UTILITIES')
        
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
        
    def delete_item(self):
        HEN_selectedObject  = self.selection()[0]
        HEN_sO_name = self.item(HEN_selectedObject, 'text')
        self.HEN_object.Streams([HEN_sO_name])
        self.delete(HEN_selectedObject)
        
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
            self.master.objectVisualizer.print2screen(HEN_sO_name)
        
        

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
        generate_cc.grid(row=1, column=0)
        generate_tid.grid(row=1, column=2)
        show_temperatures.grid(row=2, column=0)
        show_properties.grid(row=2, column=1)
    
    def make_CC(self):
        self.HEN_object.get_parameters()
        self.HEN_object.make_cc(self.tabControl)
        
    def make_TID(self):
        self.HEN_object.get_parameters()
        self.HEN_object.make_tid(self.showT.get(), self.showP.get(), self.tabControl)

class HENOS_optimization_controls(ttk.Frame):
    def __init__(self, master):
        # Intialize fram properties
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
    
        # Initialize optimization suite label
        osLabel = ttk.Label(self, text='Optimization Suite', font=('Helvetica', 10, 'bold', 'underline'))
        osLabel.grid(row=0, column=0, sticky='nw')

class HENOS_user_constraints(ttk.Frame):
    def __init__(self, master):
        # Initialize frame properties
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Initialize user constraints label
        ucLabel = ttk.Label(self, text='Forbidden/Required Matches', font=('Helvetica', 10, 'bold', 'underline'))
        ucLabel.grid(row=0, column=0, sticky='nw')
        

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
