##############################################################################
# IMPORT CALLS
##############################################################################
import tkinter as tk
from tkinter import ttk
import hen_frontend
from PIL import ImageTk, Image
import pathlib

##############################################################################
# CLASSES
##############################################################################
class ALChemE_app(tk.Frame):
    '''
    A class which holds the ALChemE app. Slave of the root window.
    '''
    def __init__(self, master):
        # Defining variables
        tk.Frame.__init__(self, master)
        self.master = master
        print(master)
        
        # Set window to screen dimensions
        swidth = master.winfo_screenwidth()
        sheight = master.winfo_screenheight()
        #top = master.winfo_toplevel()
        self.master.geometry(str(swidth)+'x'+str(sheight))

        # Set logo
        logo = Image.open(str(pathlib.Path(__file__).parent.absolute()) + '\\ALChemE_logo.png')
        pixels_x, pixels_y = tuple([int(0.25 * x)  for x in logo.size])
        logoRender = ImageTk.PhotoImage(logo.resize((pixels_x, pixels_y)))        
        logoPanel = tk.Label(self.master, image=logoRender)
        logoPanel.image = logoRender

        # Define child programs
        ALChemE_HEN = HEN_GUI_frame(self.master)
        AlChemE_WReN = WReN_GUI_frame(self.master)
        # Arrange elements
        logoPanel.grid(row=0, column=1)
        ALChemE_HEN.grid(row=1, column=0)
        AlChemE_WReN.grid(row=1, column=1, sticky='nw')
        
class HEN_GUI_frame(ttk.Frame):
    '''
    A class which holds the HEN_GUI frame. Slave of ALChemE
    '''
    def __init__(self, master):
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Define elements
        HEN_GUILabel = ttk.Label(self, text='HEN Optimization', font=('Helvetica', 14, 'bold', 'underline'))
        HEN_GUIDescrip = tk.Text(self, bg='#F0F0F0', height=2, width=50, highlightthickness=0, borderwidth=0, font=('Helvetica'))
        descrip = """A program for visualizing and solving heat exchanger network \nproblems using linear optimization."""
        newHEN_GUI = ttk.Button(self, text='New Project', command=self.run_HEN_GUI)
        HEN_GUIDescrip.tag_configure('center', justify='center')
        HEN_GUIDescrip.insert('1.0', descrip)
        HEN_GUIDescrip.tag_add('center','1.0','end')
        HEN_GUIDescrip.config(state='disabled')
        
        # Set display picture
        self.henPic = Image.open(str(pathlib.Path(__file__).parent.absolute()) + '\\hen_logo.png')
        pixels_x, pixels_y = tuple([int(0.1 * x)  for x in self.henPic.size])
        self.henRender = ImageTk.PhotoImage(self.henPic.resize((pixels_x, pixels_y)))
        self.henPanel = tk.Label(self, image=self.henRender)
        self.henPanel.image = self.henPanel
        
        # Arrange elements
        HEN_GUILabel.grid(row=0, column=0)
        self.henPanel.grid(row=1, column=0)
        HEN_GUIDescrip.grid(row=2, column=0)
        newHEN_GUI.grid(row=3, column=0)
        
    def run_HEN_GUI(self):
        HEN_GUI_window = tk.Toplevel(self.master)
        HEN_GUI_window.title('ALChemE - HEN Optimization')
        hen_frontend.HEN_GUI_app(HEN_GUI_window)

class WReN_GUI_frame(ttk.Frame):
    '''
    A class which holds the WReN_GUI frmae. Slave of ALChemE
    '''
    def __init__(self, master):
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Define elements
        WReN_GUILabel = ttk.Label(self, text='WReN Optimization', font=('Helvetica', 14, 'bold', 'underline'))
        WReN_GUIDescrip = tk.Text(self, bg='#F0F0F0', height=2, width=50, highlightthickness=0, borderwidth=0, font=('Helvetica'))
        descrip = """A program for visualizing and solving water recovery network \nproblems using optimization."""
        newWReN_GUI = ttk.Button(self, text='New Project')
        WReN_GUIDescrip.tag_configure('center', justify='center')
        WReN_GUIDescrip.insert('1.0', descrip)
        WReN_GUIDescrip.tag_add('center','1.0','end')
        WReN_GUIDescrip.config(state='disabled')
        
        # Arrange elements
        WReN_GUILabel.grid(row=0, column=0)
        WReN_GUIDescrip.grid(row=1, column=0)
        newWReN_GUI.grid(row=2, column=0)
        
    #def run_WReN_GUI(self):
        #WReN_GUI_window = tk.Toplevel(self.master)
        #WReN_GUI_window.title('ALChemE - WReN Optimization')
        #frontend.WReN_GUI_app(HEN_GUI_window)
##############################################################################
# RUN APPLICATION
##############################################################################
root = tk.Tk()

if __name__ == '__main__':
    ALChemE = ALChemE_app(root)
    ALChemE.master.title('ALChemE')
    root.mainloop()
