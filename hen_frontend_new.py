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
# WINDOW CLASS
##############################################################################
class HEN_GUI(tk.Toplevel):
    '''
    Toplevel window of the HEN program GUI.
    '''
    def __init__(self, parent=None):
        # Initialize and adjust toplevel window settings
        tk.Toplevel.__init__(self, parent)
        self.title('ALChemE - Heat exchanger network analysis tool')
    
        # Initialize and pack HEN_gui_frame
        self.HEN_GUI_frame = HEN_GUI_frame(parent=self)
        self.HEN_GUI_frame.pack()

##############################################################################
# PARENT FRAME CLASS
##############################################################################
class HEN_GUI_frame(tk.Frame):
    '''
    Parent frame of HEN program GUI. Child of HEN_GUI.
    '''
    def __init__(self, parent=None):
        # Initialize and adjust frame settings
        tk.Frame.__init__(self, parent, width=parent.winfo_screenwidth(),
                          height = parent.winfo_screenheight())
        
        # Initialize child frames
        
        
        # Pack child frames

##############################################################################
# CHILD FRAME CLASSES
##############################################################################
class HEN_GUI_input(tk.Frame):
    '''
    Frame which holds user input. Child of HEN_GUI_frame.
    '''
    def __init__(self, parent=HEN_GUI_frame):
        # Initialize and adjust Frame settings
        tk.Frame.__init__(self, parent)
        
        # Define variables
        pass

class HEN_GUI_object_explorer(tk.Treeview):
    def __init__(self):
        pass

class HEN_GUI_terminal_display(tk.Text):
    def __init__(self):
        pass

class HEN_GUI_graphics(tk.Frame):
    def __init__(self):
        pass

class HEN_GUI_optimizaion(tk.Frame):
    def __init__(self):
        pass

class HEN_GUI_constraint_explorer(tk.Frame):
    def __init__(self):
        pass

##############################################################################
# CHILD WIDGET CLASSES
##############################################################################
class HEN_GUI_object_explorer_treeview(tk.Treeview):
    def __init__(self):
        pass

class HEN_GUI_terminal_display_text(tk.Text):
    def __init__(self):
        pass

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
        Parent frame (e.g. HEN_GUI_input).

    Returns
    -------
    frame_label : tk.Label
        Label with text of parameter 'name.'

    '''
    frame_label = tk.Label(parent, text=name, font=('Helvetica', 10, 'bold',
                                                    'underline'))
    return frame_label

##############################################################################
# LOCAL TESTING
##############################################################################
root = tk.Tk()
a = HEN_GUI(master=root)
a.mainloop()
