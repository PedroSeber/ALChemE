##############################################################################
# IMPORT CALLS
##############################################################################
import tkinter as tk
from tkinter import ttk
import hen_frontend
import wren_frontend
import unyt
from PIL import ImageTk, Image
import pathlib
import platform

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
        if platform.system() == 'Windows':
            logo = Image.open(str(pathlib.Path(__file__).parent.absolute()) + '\\ALChemE_logo.png')
        else:
            logo = Image.open(str(pathlib.Path(__file__).parent.absolute()) + '/ALChemE_logo.png')
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
        if platform.system() == 'Windows':
            self.henPic = Image.open(str(pathlib.Path(__file__).parent.absolute()) + '\\hen_logo.png')
        else:
            self.henPic = Image.open(str(pathlib.Path(__file__).parent.absolute()) + '/hen_logo.png')
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
        # Initialize HEN setup window
        self.HEN_setup_window = tk.Toplevel(self.master)
        self.HEN_setup_window.title('ALChemE - HEN Optimization Setup')
        self.input_entries = {}
        
        # Initialize widgets
        HEN_sw_label = tk.Label(self.HEN_setup_window, text = 'HEN Optimization Setup', font=('Helvetica', 10, 'bold', 'underline'))
        HEN_sw_DeltT = tk.Label(self.HEN_setup_window, text ='ΔTₘ')
        # HEN_sw_HeatU = tk.Label(self.HEN_setup_window, text ='Heat Units')
        HEN_sw_DeltTE = tk.Entry(self.HEN_setup_window, width=12)
        HEN_sw_DeltTE.insert('end', '10')
        HEN_sw_DeltTU = create_dropdown_menu(self.HEN_setup_window, ['°C', '°F', 'K'])
        # HEN_sw_HeatUE = create_dropdown_menu(self.HEN_setup_window, ['J', 'kJ'])
        HEN_sw_Submit = tk.Button(self.HEN_setup_window, text='Submit', command=self.open_HEN_GUI)
        
        # Place widgets
        HEN_sw_label.grid(row=0, column=0, columnspan=3, sticky='nsew')
        HEN_sw_DeltT.grid(row=1, column=0, sticky='nsew', pady=5)
        HEN_sw_DeltTE.grid(row=1, column=1, sticky='nsew', pady=5)
        self.input_entries[str([1, 1])] = HEN_sw_DeltTE
        HEN_sw_DeltTU[0].grid(row=1, column=2, sticky='w', pady=5)
        self.input_entries[str([1, 2])] = HEN_sw_DeltTU[1]
        # HEN_sw_HeatU.grid(row=2, column=0, sticky='nsew')
        # HEN_sw_HeatUE[0].grid(row=2, column=1, sticky='nsew')
        # self.input_entries[str([2, 1])] = HEN_sw_HeatUE[1]
        HEN_sw_Submit.grid(row=3, column=0, columnspan=3)
        
    def open_HEN_GUI(self):
        # Error Flag
        errorFlag = False
        
        # Extract submission data
        dataVec = [self.input_entries[str([1, 1])].get(), self.input_entries[str([1, 2])].get()]
        
        # Sanitize minimum temperature difference input
        try:
            numericdata = float(dataVec[0])
            dataVec[0] = numericdata
        except TypeError:
            errorFlag = True
            errorMessage = 'ERROR: Invalid ΔTₘ input type'
        
        # Sanitize ΔTₘ unit input
        if dataVec[1] == '°C':
            dataVec[1] = unyt.degC
        elif dataVec[1] == '°F':
            dataVec[1] = unyt.degF
        else:
            dataVec[1] = unyt.K
        
        # Sanitize heat unit input
        # if dataVec[2] == 'J':
        #     dataVec[2] = unyt.J
        # else:
        #     dataVec[2] = unyt.kJ
        
        # Run HEN optimization program
        if errorFlag == False:
            self.HEN_setup_window.destroy()
            HEN_GUI_window = tk.Toplevel(self.master)
            HEN_GUI_window.title('ALChemE - HEN Optimization')
            hen_frontend.HEN_GUI_app(HEN_GUI_window, deltaTmin=dataVec[0], tempUnit=dataVec[1])

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
        newWReN_GUI = ttk.Button(self, text='New Project', command=self.run_WReN_GUI)
        WReN_GUIDescrip.tag_configure('center', justify='center')
        WReN_GUIDescrip.insert('1.0', descrip)
        WReN_GUIDescrip.tag_add('center','1.0','end')
        WReN_GUIDescrip.config(state='disabled')
        
        # Arrange elements
        WReN_GUILabel.grid(row=0, column=0)
        WReN_GUIDescrip.grid(row=1, column=0)
        newWReN_GUI.grid(row=2, column=0)
        
    def run_WReN_GUI(self):
        WReN_GUI_window = tk.Toplevel(self.master)
        WReN_GUI_window.title('ALChemE - WReN Optimization')
        wren_frontend.WReN_GUI_app(WReN_GUI_window)

##############################################################################
# FUNCTIONS
##############################################################################
def create_dropdown_menu(master, options):
    var = tk.StringVar(master)
    menu = ttk.OptionMenu(master, var, options[0], *options)
    return [menu, var]

##############################################################################
# RUN APPLICATION
##############################################################################
root = tk.Tk()

if __name__ == '__main__':
    ALChemE = ALChemE_app(root)
    ALChemE.master.title('ALChemE')
    root.mainloop()
