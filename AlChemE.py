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
        pixels_x, pixels_y = tuple([int(0.75 * x)  for x in logo.size])
        logoRender = ImageTk.PhotoImage(logo.resize((pixels_x, pixels_y)))        
        logoPanel = tk.Label(self.master, image=logoRender)
        logoPanel.image = logoRender
        
        # Define child programs
        HENOS = HENOS_frame(self.master)
        WRENOS = WRENOS_frame(self.master)
        # Arrange elements
        logoPanel.grid(row=0, column=1)
        HENOS.grid(row=1, column=0)
        WRENOS.grid(row=1, column=1, sticky='nw')
        
class HENOS_frame(ttk.Frame):
    '''
    A class which holds the HENOS frame. Slave of ALChemE
    '''
    def __init__(self, master):
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Define elements
        HENOSLabel = ttk.Label(self, text='HEN Optimization', font=('Helvetica', 14, 'bold', 'underline'))
        HENOSDescrip = tk.Text(self, bg='#F0F0F0', height=2, width=50, highlightthickness=0, borderwidth=0, font=('Helvetica'))
        descrip = """A program for visualizing and solving heat exchanger network \nproblems using linear optimization."""
        newHENOS = ttk.Button(self, text='New Project', command=self.run_HENOS)
        HENOSDescrip.tag_configure('center', justify='center')
        HENOSDescrip.insert('1.0', descrip)
        HENOSDescrip.tag_add('center','1.0','end')
        HENOSDescrip.config(state='disabled')
        
        # Set display picture
        self.henPic = Image.open(str(pathlib.Path(__file__).parent.absolute()) + '\\hen_logo.png')
        pixels_x, pixels_y = tuple([int(0.1 * x)  for x in self.henPic.size])
        self.henRender = ImageTk.PhotoImage(self.henPic.resize((pixels_x, pixels_y)))
        self.henPanel = tk.Label(self, image=self.henRender)
        self.henPanel.image = self.henPanel
        
        # Arrange elements
        HENOSLabel.grid(row=0, column=0)
        self.henPanel.grid(row=1, column=0)
        HENOSDescrip.grid(row=2, column=0)
        newHENOS.grid(row=3, column=0)
        
    def run_HENOS(self):
        HENOS_window = tk.Toplevel(self.master)
        HENOS_window.title('HEN Optimization Software')
        hen_frontend.HENOS_app(HENOS_window)

class WRENOS_frame(ttk.Frame):
    '''
    A class which holds the WRENOS frmae. Slave of ALChemE
    '''
    def __init__(self, master):
        ttk.Frame.__init__(self, master, padding='0.1i', relief='solid')
        
        # Define elements
        WRENOSLabel = ttk.Label(self, text='WReN Optimization', font=('Helvetica', 14, 'bold', 'underline'))
        WRENOSDescrip = tk.Text(self, bg='#F0F0F0', height=2, width=50, highlightthickness=0, borderwidth=0, font=('Helvetica'))
        descrip = """A program for visualizing and solving water recovery network \nproblems using optimization."""
        newWRENOS = ttk.Button(self, text='New Project')
        WRENOSDescrip.tag_configure('center', justify='center')
        WRENOSDescrip.insert('1.0', descrip)
        WRENOSDescrip.tag_add('center','1.0','end')
        WRENOSDescrip.config(state='disabled')
        
        # Arrange elements
        WRENOSLabel.grid(row=0, column=0)
        WRENOSDescrip.grid(row=1, column=0)
        newWRENOS.grid(row=2, column=0)
        
    #def run_WRENOS(self):
        #HENOS_window = tk.Toplevel(self.master)
        #HENOS_window.title('HEN Optimization Software')
        #frontend.HENOS_app(HENOS_window)
##############################################################################
# RUN APPLICATION
##############################################################################
root = tk.Tk()

if __name__ == '__main__':
    ALChemE = ALChemE_app(root)
    ALChemE.master.title('ALChemE')
    root.mainloop()
