##############################################################################
# IMPORT CALLS
##############################################################################
import numpy as np
import unyt
from collections import namedtuple, OrderedDict
import tkinter as tk
from tkinter import ttk
from hen_design import HEN
from hen_design import generate_GUI_plot
import subprocess
import pdb


##############################################################################
# PARENT WINDOW CLASS
##############################################################################
class HEN_GUI(tk.Toplevel):
    '''
    Toplevel window of the HEN program GUI.
    '''
    def __init__(self, parent):
        # Initialize and adjust toplevel window settings
        tk.Toplevel.__init__(self)
        self.title('ALChemE - Heat exchange network analysis tool')
        
        # Initialize and pack HEN_GUI_frame
        self.HEN_GUI_frame = HEN_GUI_frame(self)
        self.HEN_GUI_frame.pack(expand=True, fill='both')


##############################################################################
# PARENT FRAME CLASS
##############################################################################
class HEN_GUI_frame(tk.Frame):
    '''
    Parent frame of HEN program GUI. Child of HEN_GUI. Parent of all classes
    in section: CHILD FRAME CLASSES.
    '''
    def __init__(self, parent=None):
        # Initialize and adjust frame settings
        tk.Frame.__init__(self, parent, width=parent.winfo_screenwidth(),
                          height=parent.winfo_screenheight())
        self.grid_propagate(0)
        
        # Define variables
        self.HEN_object = HEN()

        # Initialize child frames
        self.HEN_GUI_stream_input = HEN_GUI_stream_input(self)
        self.HEN_GUI_heat_exchanger_input = HEN_GUI_heat_exchanger_input(self)
        self.HEN_GUI_utility_input = HEN_GUI_utility_input(self)
        self.HEN_GUI_object_explorer = HEN_GUI_object_explorer(self)
        self.HEN_GUI_terminal_display = HEN_GUI_terminal_display(self)
        self.HEN_GUI_graphics = HEN_GUI_graphics(self)
        self.HEN_GUI_optimization = HEN_GUI_optimization(self)
        self.HEN_GUI_constraint_explorer = HEN_GUI_constraint_explorer(self)

        # Pack child frames
        self.HEN_GUI_stream_input.grid(row=0, column=0, sticky='nsew')
        self.HEN_GUI_heat_exchanger_input.grid(row=1, column=0, sticky='nsew')
        self.HEN_GUI_utility_input.grid(row=2, column=0, sticky='nsew')
        self.HEN_GUI_object_explorer.grid(row=3, column=0, sticky='nsew')
        self.HEN_GUI_terminal_display.grid(row=4, column=0, sticky='nsew')
     

##############################################################################
# CHILD FRAME CLASSES
##############################################################################
class HEN_GUI_stream_input(tk.Frame):
    '''
    Frame which contains the stream user input. Child of HEN_GUI_frame.
    '''
    def __init__(self, HEN_GUI_frame):
        # Initialize and adjust Frame settings
        ttk.Frame.__init__(self, HEN_GUI_frame, padding='0.1i', relief='solid')
        
        # Define variables
        self.HEN_object = HEN_GUI_frame.HEN_object
        self.HEN_stream_input_labels = ['Stream Name', 'Temperature In',
                                        'Temperature Out', '',
                                        'Heat Capacity', '', 'Flow Rate',
                                        '', '', 'Heat Load', '']
        
        # Initialize frame label
        stream_input_label = nameFrame('Stream Input', self)

        # Initialize and pack input labels, entries, and options menus
        for row in range(1,3):
            for col in range(11): 
                # Initialize input labels
                if row==1 and col in [0, 1, 2, 4, 6, 9]:
                    iLabel = ttk.Label(self, 
                                      text=self.HEN_stream_input_labels[col])
                    iLabel.grid(row=row, column=col, padx=10)
                elif row == 1 and col in [3, 5, 7]:
                    iLabel = ttk.Label(self, width=12)
                    iLabel.grid(row=row, column=col, padx=10)
                elif row == 1 and col == 10:
                    iLabel = ttk.Label(self, width=7)
                    iLabel.grid(row=row, column=col, padx=10)
                
                # Initialize and pack input entries and option menus
                else:
                    if col in [0, 1, 2, 4, 6, 9]:
                        iEntry = ttk.Entry(self, width=12)
                        iEntry.grid(row=row, column=col)
                    elif col==3:
                        iMenu = initDropdownMenu(self, ['°C', '°F', 'K'])
                        iMenu[0].grid(row=row, column=col, sticky='w')
                    elif col == 5:
                        iMenu = initDropdownMenu(self, ['J/(kg·°C)', 'BTU/(lb·°F)', 'J/(kg·K)'])
                        iMenu[0].grid(row = row, column=col, sticky='w')
                    elif col == 7:
                        iMenu = initDropdownMenu(self, ['kg/s', 'lb/s'])
                        iMenu[0].grid(row = row, column=col, sticky='w')
                    elif row == 2 and col == 8:
                        OrLabel = ttk.Label(self, text='OR', font=('Helvetica', 10, 'bold'))
                        OrLabel.grid(row=row, column=col, padx=(0,35))
                    elif col == 10:
                        iMenu = initDropdownMenu(self, ['W', 'kcal/s', 'BTU/s'])
                        iMenu[0].grid(row = row, column=col, sticky='w', padx=(0,10))

        # Initialize 'Add Stream' button
        add_stream_button = ttk.Button(self, text='Add Stream')

        # Pack widgets
        stream_input_label.grid(row=0, column=0, sticky='w')
        add_stream_button.grid(row=2, column=11)
    
    def add_stream(self):
        pass


class HEN_GUI_heat_exchanger_input(tk.Frame):
    '''
    Frame which contains the heat exchanger user input. Child of HEN_GUI_frame.
    '''
    def __init__(self, HEN_GUI_frame):
        # Initialize and adjust Frame settings
        ttk.Frame.__init__(self, HEN_GUI_frame, padding='0.1i', relief='solid')
        
        # Define variables
        self.HEN_object = HEN_GUI_frame.HEN_object
        
        # Initialize frame label
        heat_exch_input_label = nameFrame('Heat Exchanger Input', self)

        # Pack widgets
        heat_exch_input_label.grid(row=0, column=0, sticky='w')
    
    def add_heat_exchanger(self):
        pass


class HEN_GUI_utility_input(tk.Frame):
    '''
    Frame which contains the utility user input. Child of HEN_GUI_frame.
    '''
    def __init__(self, HEN_GUI_frame):
        # Initialize and adjust Frame settings
        ttk.Frame.__init__(self, HEN_GUI_frame, padding='0.1i', relief='solid')
        
        # Define variables
        self.HEN_object = HEN_GUI_frame.HEN_object
        
        # Initialize frame label
        utility_input_label = nameFrame('Utility Input', self)

        # Pack widgets
        utility_input_label.grid(row=0, column=0, sticky='w')

    def add_utility(self):
        pass        


class HEN_GUI_object_explorer(tk.Frame):
    '''
    Frame which contains the object explorer. Child of HEN_GUI_frame. Parent
    of HEN_GUI_object_explorer_treeview.
    '''
    def __init__(self, HEN_GUI_frame):
        # Initialize and adjust Frame settings
        ttk.Frame.__init__(self, HEN_GUI_frame, padding='0.1i', relief='solid')

        # Define variables
        self.HEN_object = HEN_GUI_frame.HEN_object

        # Initialize frame label
        object_explorer_label = nameFrame('Object Explorer', self)

        # Pack widgets
        object_explorer_label.grid(row=0, column=0, sticky='w')

    def add_object(self):
        pass

    def delete_object(self):
        pass


class HEN_GUI_terminal_display(tk.Frame):
    '''
    Frame which contains the terminal display. Child of HEN_GUI_frame. Parent
    of HEN_GUI_terminal_display_text.
    '''
    def __init__(self, HEN_GUI_frame):
        # Initialize and adjust Frame settings
        ttk.Frame.__init__(self, HEN_GUI_frame, padding='0.1i', relief='solid')
        
        # Initialize frame label
        terminal_display_label = nameFrame('Terminal Display', self)

        # Pack widgets
        terminal_display_label.grid(row=0, column=0, sticky='w')

    def update_display(self):
        pass

    def clear_display(self):
        pass


class HEN_GUI_graphics(tk.Frame):
    '''
    Frame which conatins the graphics tools. Child of HEN_GUI_frame.
    '''
    def __init__(self, HEN_GUI_frame):
        # Initialize and adjust Frame settings
        ttk.Frame.__init__(self, HEN_GUI_frame, padding='0.1i', relief='solid')
        
        # Define variables
        self.HEN_object = HEN_GUI_frame.HEN_object

        # Initialize frame label
        graphics_label = nameFrame('Graphical Tools', self)

        # Pack widgets

    def show_tid(self):
        pass

    def show_composite_curve(self):
        pass

class HEN_GUI_optimization(tk.Frame):
    '''
    Frame which contains the optimization tools. Child of HEN_GUI_frame.
    '''
    def __init__(self, HEN_GUI_frame):
        # Initialize and adjust Frame setings
        ttk.Frame.__init__(self, HEN_GUI_frame, padding='0.1i', relief='solid')

        # Define variables
        self.HEN_object = HEN_GUI_frame.HEN_object

        # Initialize frame label
        optimization_label = nameFrame('Optimization Tools', self)

        # Pack widgets

    def run_optimization(self):
        pass


class HEN_GUI_constraint_explorer(tk.Frame):
    '''
    Frame which contains the constraint explorer. Child of HEN_GUI_frame.
    Parent of HEN_GUI_constraint_explorer_treeview.
    '''
    def __init__(self, HEN_GUI_frame):
        # Initialize and adjust Frame settings
        ttk.Frame.__init__(self, HEN_GUI_frame, padding='0.1i', relief='solid')

        # Initialize frame label
        constraint_explorer_label = nameFrame('Constraint Explorer', self)

        # Pack widget

    def add_object(self):
        pass

    def delete_object(self):
        pass

##############################################################################
# CHILD WIDGET CLASSES
##############################################################################
class HEN_GUI_object_explorer_treeview(ttk.Treeview):
    '''
    Treeview widget for the object explorer. Child of HEN_GUI_object_explorer.
    '''
    def __init__(self, HEN_GUI_object_explorer):
        # Initialize and adjust Treeview settings
        ttk.Treeview.__init__(self, HEN_GUI_object_explorer)


class HEN_GUI_terminal_display_text(tk.Text):
    '''
    Text widget for the terminal display. Child of HEN_GUI_terminal_display.
    '''
    def __init__(self, HEN_GUI_terminal_display):
        # Initialize and adjust Text settings
        tk.Text.__init__(self, HEN_GUI_terminal_display)


class HEN_GUI_constraint_explorer_treeview(ttk.Treeview):
    '''
    Treeview widget for the constraint explorer. Child of 
    HEN_GUI_object_explorer.
    '''
    def __init__(self, HEN_GUI_constraint_explorer):
        # Initialize and adjust Treeview settings
        ttk.Treeview.__init__(self, HEN_GUI_constraint_explorer)

##############################################################################
# LOCAL FUNCTIONS
##############################################################################
def nameFrame(name, parent):
    '''
    Description
    ----------
    Creates title labels for child frames.
    
    Parameters
    ----------
    name : str
        Frame name (e.g. 'User Input').
    parent : tk.Frame
        Parent frame (e.g. HEN_GUI_stream_input).

    Returns
    -------
    frame_label : tk.Label
        Label with text of parameter 'name.'

    '''
    frame_label = ttk.Label(parent, text=name, font=('Helvetica', 10, 'bold',
                                                    'underline'))
    return frame_label

def initDropdownMenu(parent, labels):
    '''
    Description
    ----------
    Creates dropdown menu for units input.

    Parameters
    ----------
    parent : Frame
        Parent frame (e.g. HEN_GUI_stream_input).
    labels : list
        List of label options.

    Returns
    ----------
    menu : ttk.OptionMenu
        Dropdown menu with all elements in 'labels' as options.
    var : tk.StringVar
        StringVar object to read current menu selection.
    
    '''
    var = tk.StringVar(parent)
    menu = ttk.OptionMenu(parent, var, labels[0], *labels)
    return [menu, var]
##############################################################################
# LOCAL DEBUGGING
##############################################################################
root = tk.Tk()
root.withdraw()
a = HEN_GUI(root)
a.mainloop()
