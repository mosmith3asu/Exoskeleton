import matplotlib

matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
from tkinter import ttk


class mclass:
    def __init__(self, window, fig, is_max=False):
        self.window = window
        self.fig = fig
        # Initial Configuration
        self.tab_names = ["Vitals",
                          "Settings",
                          "GMM",
                          "Regression",
                          "IRL",
                          "Pot Field"]

        self.setting_names = ["Show Sensors",
                              "Show Knee Moment"]
        self.setting_states = []

        # Pack Simulation & Gait Phase Plot in Interface
        self.pack_figure()

        # Pack tabs (vitals, settings & learning models)
        self.tab = []
        self.pack_notebook()
        #self.pack_notebook_settings(self.tab[1],self.setting_names)

        # Maximize window
        if is_max:
            self.window.state("zoomed")

    def pack_figure(self):
        canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        canvas.get_tk_widget().pack(side=LEFT, fill='both')
        canvas.draw()

    def pack_notebook(self):
        self.tab_parent = ttk.Notebook(self.window)

        for name in range(len(self.tab_names)):
            self.tab.append(ttk.Frame(self.tab_parent))
            self.tab_parent.add(self.tab[name], text=self.tab_names[name])

        self.tab_parent.pack(side=LEFT, expand=1, fill='both')

    def pack_notebook_settings(self,frame,settings_names):
        self.setting_states = []
        for setting in range(len(settings_names)):
            self.setting_states.append(IntVar())
            chk = Checkbutton(frame,
                              text=self.setting_names[setting],
                              variable=self.setting_states[setting],
                              command=self.on_click)
            chk.pack(expand=YES)

    def on_click(self):
        print("variable is", self.setting_states[0].get())
        print(list(map((lambda var: var.get()), self.setting_states)))




    def update_vitals(self, vitals_data):
        frame=self.tab[0]
        try:
            num_rows = len(vitals_data)
            num_cols = len(vitals_data[0])

            # Pack Vital Labels and Data in grid
            for r in range(num_rows):
                for c in range(num_cols):
                    if c == 0:  # Bold Headers
                        self.vitals_table = Label(frame,
                                                  text=vitals_data[r][c],
                                                  font=('Arial', 10, 'bold'),
                                                  width=20,
                                                  anchor=E)
                    else:
                        self.vitals_table = Label(frame,
                                                  text=vitals_data[r][c],
                                                  width=20, )

                    self.vitals_table.grid(row=r, column=c,
                                           sticky=W + E + N + S)

            # Set Grid to expand vertically in vitals tab
            for r in range(num_rows):
                Grid.rowconfigure(frame, r, weight=1)

            # Set Grid to expand horazontally in vitals tab
            # for c in range(num_cols):
            #    Grid.columnconfigure(self.tab[0], c, weight=1)

        except:
            print("Unable to Pack Vitals Table (Empty?)")

class Checkbar(Frame):
    def __init__(self, parent=None, picks=[], side=LEFT, anchor=W):
        Frame.__init__(self, parent)
        self.vars = []
        for pick in picks:
            var = IntVar()
            chk = Checkbutton(self, text=pick, variable=var)
            chk.pack(side=side, anchor=anchor, expand=YES)
            self.vars.append(var)

    def state(self):
        return map((lambda var: var.get()), self.vars)
