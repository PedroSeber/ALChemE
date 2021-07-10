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
    def __init__(self, parent=None):
        # Initialize and adjust toplevel window settings
        tk.Toplevel.__init__(self, parent)
        self.title('ALChemE - Heat exchanger network analysis tool')
    
        # Initialize and pack HEN_GUI_frame
        self.HEN_GUI_frame = HEN_GUI_frame(parent=self)
        self.HEN_GUI_frame.pack()

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
                          height = parent.winfo_screenheight())
        
        # Initialize child frames
        
        
        # Pack child frames

##############################################################################
# CHILD FRAME CLASSES
##############################################################################
class HEN_GUI_input(tk.Frame):
    '''
    Frame which contains the user input. Child of HEN_GUI_frame.
    '''
    def __init__(self, HEN_GUI_frame, HEN_object):
        # Initialize and adjust Frame settings
        tk.Frame.__init__(self, parent=HEN_GUI_frame)
        
        # Define variables
        self.HEN_object = HEN_object
        pass

class HEN_GUI_object_explorer(tk.Treeview):
    '''
    Frame which contains the object explorer. Child of HEN_GUI_frame. Parent
    of HEN_GUI_object_explorer_treeview.
    '''
    def __init__(self):
        pass

class HEN_GUI_terminal_display(tk.Text):
    '''
    Frame which contains the terminal display. Child of HEN_GUI_frame. Parent
    of HEN_GUI_terminal_display_text.
    '''
    def __init__(self):
        pass

class HEN_GUI_graphics(tk.Frame):
    '''
    Frame which conatins the graphics tools. Child of HEN_GUI_frame.
    '''
    def __init__(self):
        pass

class HEN_GUI_optimizaion(tk.Frame):
    '''
    Frame which contains the optimization tools. Child of HEN_GUI_frame.
    '''
    def __init__(self):
        pass

class HEN_GUI_constraint_explorer(tk.Frame):
    '''
    Frame which contains the constraint explorer. Child of HEN_GUI_frame.
    Parent of HEN_GUI_constraint_explorer_treeview.
    '''
    def __init__(self):
        pass

##############################################################################
# CHILD WIDGET CLASSES
##############################################################################
class HEN_GUI_object_explorer_treeview(tk.Treeview):
    '''
    Treeview widget for the object explorer. Child of HEN_GUI_object_explorer.
    '''
    def __init__(self):
        pass

class HEN_GUI_terminal_display_text(tk.Text):
    '''
    Text widget for the terminal display. Child of HEN_GUI_terminal display.
    '''
    def __init__(self):
        pass

class HEN_GUI_constraint_explorer_treeview(tk.Treeview):
    '''
    Treeview widget for the constraint explorer. Child of 
    HEN_GUI_object_explorer.
    '''
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
