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

# CLASSES
class HENOS_app():
    '''
    A class which holds the HENOS application. Slave of root window.
    '''
    def __init__(self, master):
        
        # Defining variables
        self.master = master
        self.style = ttk.Style()
        self.style.configure('main.TFrame', foreground = "black", background = "red")
        
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
        
        # Intialize control panel
        control_panel_Tab = ttk.Frame(self.tabControl)
        self.tabControl.add(control_panel_Tab, text='HENOS Control Panel')
        self.tabControl.pack(expand=1, fill='both')
        
        # Initialize HEN object
        HEN_object = HEN()
        
        # Initialize control panel elements
        self.HENOS_object_explorer = HENOS_object_explorer(control_panel_Tab, HEN_object)
        self.HENOS_stream_input = HENOS_stream_input(control_panel_Tab, HEN_object, self.HENOS_object_explorer)
        self.HENOS_oe_controls = HENOS_object_explorer_controls(control_panel_Tab, self.HENOS_object_explorer)
        
        
        # Placing control panel elements
        control_panel_Tab.rowconfigure(1, weight=1)
        self.HENOS_stream_input.grid(row=0, column=0)
        self.HENOS_object_explorer.grid(row=1, column=0, rowspan=40, columnspan=8, padx=5, sticky='nsew')
        self.HENOS_oe_controls.grid(row=1, column=9, sticky='nw')
        # test = ttk.Button(control_panel_Tab, text='test')
        # test.grid(row=41, column=0, columnspan=8, sticky='ew')

class HENOS_stream_input(ttk.Frame):
    '''
    A class which holds the HENOS stream input. Slave of HENOS_app.    
    '''
    def __init__(self, master, HEN_object, HEN_object_explorer):        
        # Initialize frame properties
        ttk.Frame.__init__(self, master, padding='0.25i')

        # Defining variables
        self.HEN_object = HEN_object
        self.HEN_object_explorer = HEN_object_explorer
        self.HEN_stream_labels = ['Stream Name', 'Inlet Temperature',
                             'Outlet Temperature', '',
                             'Heat Capacity', '',
                             'Flow Rate', '', 'Heat Load', '']
        self.input_entries = {}
        
        # Arrange stream input components
        for row in range(2):
            for col in range(10):
                if row == 0:
                    l = ttk.Label(self, text=self.HEN_stream_labels[col])
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
        sub_stream.grid(row=1, column=10, sticky='nsew')
    
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
        
        # Convert temperature unit input to unyt input
        if raw_input[3] == '°C':
            raw_input[3] = unyt.degC
        elif raw_input[3] == '°F':
            raw_input[3] = unyt.degF
        else:
            raw_input[3] = unyt.degK
            
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
        self.HEN_object.add_stream(t1 = raw_input[1], t2 = raw_input[2], cp = raw_input[4], heat = raw_input[6], stream_name = raw_input[0], HENOS_oe_tree = self.HEN_object_explorer.objectExplorer, temp_unit = raw_input[3])
    

class HENOS_object_explorer(ttk.Frame):
    '''
    A class which holds the HENOS object explorer and visualizer. Slave of
    HENOS_app.
    '''
    def __init__(self, master, HEN_object):
        # Initialize frame properties
        ttk.Frame.__init__(self, master, padding='0.25i')
        
        # Defining variables
        self.HEN_object = HEN_object
        
        # Initialize object explorer        
        self.objectExplorer = HENOS_objE_tree(self, self.HEN_object)
    
        
        # Initialize object visualizer
        self.objectVisualizer = HENOS_objE_display(self, self.HEN_object)
        
        # Place object explorer and visualizer
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(41, weight=1)
        self.objectExplorer.grid(row=0, column=0)
        self.objectExplorer.grid(row=0, column=0, rowspan=40, columnspan=8, padx=5, sticky='nsew')
        self.objectVisualizer.grid(row=41, column=0, rowspan=1, columnspan=8, padx=5, pady=25, sticky='nsew')

class HENOS_objE_display(tk.Text):
    def __init__(self, master, HEN_object):
        # Initialize text properties
        tk.Text.__init__(self, master)
        
        # Defining variables
        self.HEN_object = HEN_object
        
        # Initialize vertical scrollbar
        self.verscrlbar = ttk.Scrollbar(self, orient='vertical', command=self.yview)
        self.verscrlbar.pack(side='right', fill='y')
        self.configure(yscrollcommand=self.verscrlbar.set)
    
    def print2screen(self, object_name):
        self.delete(1.0, 'end')
        displaytext = str(self.HEN_object.streams[object_name])
        self.insert('end', displaytext)

class HENOS_objE_tree(ttk.Treeview):
    '''
    A class which holds the Treeview object which forms the basis of the
    object explorer. Slave of HENOS_object_explorer
    '''
    def __init__(self, master, HEN_object):
        # Initialize treeview properties
        ttk.Treeview.__init__(self, master, show='tree')
        style = ttk.Style()
        style.layout("Treeview", [('Treeview.treearea', {'sticky': 'nswe'})])
        self['columns'] = ('1', '2', '3', '4', '5')
        
        # Defining variables
        self.HEN_object = HEN_object
        self.master = master
        
        # Intialize vertical scrollbar
        self.verscrlbar = ttk.Scrollbar(self, orient='vertical', command=self.yview)
        self.verscrlbar.pack(side='right', fill='y')
        self.configure(yscrollcommand=self.verscrlbar.set)
        
        # Intialize object classes in treeview
        self.StreamNode = self.insert('', index=0, iid=0, text='STREAMS', values=('Inlet Temperature', 'Outlet Temperature', 'Heat Capacity Rate', 'Heat Load'))
        self.HXNode = self.insert('', index=1, iid=1, text='HEAT EXCHANGERS')
        self.UtilityNode = self.insert('', index=2, iid=2, text='UTILITIES')
        
        # Initialize 'Double Click' Event
        self.bind("<Double-1>", self.send2screen)
        
    def receive_new_stream(self, oeDataVector):
        self.insert(self.StreamNode, 'end', text=oeDataVector[0], values=(str(oeDataVector[1]), str(oeDataVector[2]), str(oeDataVector[3])))
        
    def delete_item(self):
        selected_item  = self.selection()[0]
        self.delete(selected_item)
        
    def send2screen(self, event):
        HEN_selectedObject = self.identify('item',event.x, event.y)
        print(HEN_selectedObject)
        HEN_sO_name = self.item(HEN_selectedObject, 'text')
        self.master.objectVisualizer.print2screen(HEN_sO_name)
        
        

class HENOS_object_explorer_controls(ttk.Frame):
    def __init__(self, master, object_explorer):
        # Initialize frame properties
        ttk.Frame.__init__(self, master, padding='0.25i')
        
        # Delete streams button
        delete_stream = ttk.Button(self, text='Delete Stream', command=object_explorer.objectExplorer.delete_item)
        
        delete_stream.grid(row=0, column=0)
        

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
    root.mainloop()
