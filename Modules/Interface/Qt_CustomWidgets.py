import matplotlib

matplotlib.use('Qt5Agg')
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog
from Modules.Interface.Data_Validation.QtDes_DataValidationInterface import Ui_MainWindow

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d
from Modules.Plotting import RenderMesh
import numpy as np
from scipy import ndimage
import matplotlib.cm as cm
import sys
from types import MethodType
from pathlib import Path


class MyMplCanvas2D(Canvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        self.compute_initial_figure()

        Canvas.__init__(self, self.fig)
        self.setParent(parent)

        Canvas.setSizePolicy(self,
                             QtWidgets.QSizePolicy.Expanding,
                             QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class MyMplCanvas3D(Canvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection="3d")

        self.compute_initial_figure()

        Canvas.__init__(self, self.fig)
        self.setParent(parent)

        Canvas.setSizePolicy(self,
                             QtWidgets.QSizePolicy.Expanding,
                             QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class MplWidget2D(MyMplCanvas2D):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        MyMplCanvas2D.__init__(self, *args, **kwargs)

    def update_figure_plot(self, data):
        self.axes.cla()
        self.axes.plot(data[:, 0], data[:, 1])
        self.draw()

    def update_figure_groundforces(self,data):
        self.axes.cla()
        t = np.arange(0, len(data[:, 0]))
        self.axes.plot(t, data[:, 0])
        self.axes.plot(t, data[:, 1])
        self.axes.plot(t, data[:, 2])
        self.axes.plot(t, data[:, 3])
        self.axes.plot(t, data[:, 4])
        self.draw()

    def update_figure_gaitphase(self,data):
        self.axes.cla()
        t = np.arange(0, len(data))
        self.axes.plot(t, data)
        self.draw()



class MplWidget3D(MyMplCanvas3D):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        MyMplCanvas3D.__init__(self, *args, **kwargs)

    def update_figure_plot(self, data):
        self.axes.cla()
        self.axes.plot(data[:, 0], data[:, 1])
        self.draw()

    def update_figure_sim(self,rotM_thigh,rotM_femur):
        self.axes.cla()

        #Generate Leg Mesh
        LegMesh, hip, knee, ankle, kneeA, hipA = RenderMesh.generate_legmesh_Matrix(rotM_thigh,rotM_femur)

        # Add Leg Mesh to plot
        for m in LegMesh:
            self.axes.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))

        # Auto scale to the mesh size
        #scale = np.concatenate([m.points for m in LegMesh]).flatten("C")
        #self.axes.auto_scale_xyz(scale, scale, scale)

        size = 0.7
        self.axes.set_xlim(-size,size)
        self.axes.set_ylim(-size, size)
        self.axes.set_zlim(-size, size)
        self.draw()
        return kneeA, hipA

    def centroid(self,data):
        data = np.array(data)

        #print(data)
        if len(data)==18:
            data = data.reshape(3, 6)
        elif len(data)==10:
            data = data.reshape(-1, 5)
        cm = ndimage.measurements.center_of_mass(data)
        mag = np.sum(data)
        return cm,mag

    def update_figure_pressuresensor(self,sensor_data,sensor_sel,thresh_filter=True,noise_Thresh = 0.2):
        self.axes.cla()
        x_loc_tot = [1, 2, 3, 4, 5, 6,  # first sensor
                 1, 2, 3, 4, 5, 6,  # first sensor
                 1, 2, 3, 4, 5, 6,
                 1, 2, 3, 4, 5,  # second sensor
                 1, 2, 3, 4, 5  # second sensor
                 ]
        y_loc_tot = [1, 1, 1, 1, 1, 1,  # first sensor
                 2, 2, 2, 2, 2, 2,  # first sensor
                 3, 3, 3, 3, 3, 3,
                 1, 1, 1, 1, 1,  # second sensor
                 2, 2, 2, 2, 2  # second sensor
                 ]
        if sensor_sel == "thigh":
            x_loc = x_loc_tot[:18]
            y_loc = y_loc_tot[:18]
            mag_array = sensor_data[:18]
        elif sensor_sel== "femur":
            x_loc = x_loc_tot[18:28]
            y_loc = y_loc_tot[18:28]
            mag_array = sensor_data[18:28]

        if thresh_filter:
            for i in range(len(mag_array)):
                if mag_array[i] < max(mag_array) * noise_Thresh or mag_array[i] <= 2:
                    mag_array[i] = 0.0

        # Find centroids
        cm, mag = self.centroid(mag_array)

        # Add new center of mass location to the histogram array
        x_loc.append(cm[1] + 1)
        y_loc.append(cm[0]+1)

        # Add new magnitude location to the histogram array
        mag_array.append(mag)

        gridRes = 0.7
        z_loc = np.zeros(len(x_loc))
        dx = np.ones(len(x_loc))*gridRes
        dy = np.ones(len(x_loc))*gridRes
        #Plot Data
        if sensor_sel=="thigh":
            self.axes.set_xlim([1, 7])
            self.axes.set_ylim([1, 4])
            self.axes.set_zlim([0, 150])
        elif sensor_sel=="femur":
            self.axes.set_xlim([1, 6])
            self.axes.set_ylim([1, 4])
            self.axes.set_zlim([0, 150])
        #print("made it this far")

        # Cmap
        if False:
            length = len(x_loc)
            cmap = cm.get_cmap('jet')  # Get desired colormap
            max_height = np.max(mag_array[0:length-1])  # get range of colorbars
            min_height = np.min(mag_array[0:length-1])

            # scale each z to [0,1], and get their rgb values
            rgba = [cmap((k - min_height) / max_height) for k in mag_array[0:length-1]]

        try:
            length = len(x_loc)
            self.axes.bar3d(x_loc[0:length-1], y_loc[0:length-1], z_loc[0:length-1],
                     dx[0:length-1], dy[0:length-1], mag_array[0:length-1],
                     )
            self.axes.bar3d(x_loc[- 1], y_loc[- 1], z_loc[- 1], dx[- 1],
                     dy[- 1], mag_array[- 1], color = 'red')
            self.draw()

            net_force = '('+'{0:.1f}'.format(cm[1])+','+'{0:.1f}'.format(cm[0])+')'+' '+ '{0:.1f}'.format(mag)

            return net_force

        except: #Error Handling for bad data
            print("error in pressure sensor Qt Widget")


# class MplCanvas2D(Canvas):
#     def __init__(self):
#         self.fig = Figure()
#         self.ax = self.fig.add_subplot(111)
#         Canvas.__init__(self, self.fig)
#         Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
#         Canvas.updateGeometry(self)
#
# class MplWidget2D(QtWidgets.QWidget):
#     def __init__(self, parent=None):
#         QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
#         self.canvas = MplCanvas2D()                  # Create canvas object
#         self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
#         self.vbl.addWidget(self.canvas)
#         self.setLayout(self.vbl)
#     def update(self,new_canvas):
#         self.vbl.replaceWidget(self.canvas,new_canvas)
#         self.setLayout(self.vbl)

# class MyQThread(QtCore.QThread):
#     # Signals to relay thread progress to the main GUI thread
#     progressSignal = QtCore.Signal(int)
#     completeSignal = QtCore.Signal(str)
#
#     def __init__(self, parent=None):
#         super(MyQThread, self).__init__(parent)
#         # You can change variables defined here after initialization - but before calling start()
#         self.maxRange = 100
#         self.completionMessage = "done."
#
#     def run(self):
#         # blocking code goes here
#         emitStep = int(self.maxRange/100.0) # how many iterations correspond to 1% on the progress bar
#
#         for i in range(self.maxRange):
#             time.sleep(0.01)
#
#             if i%emitStep==0:
#                 self.progressSignal.emit(i/emitStep)
#
#         self.completeSignal.emit(self.completionMessage)

# class MplCanvas3D(MyMplCanvas3D):
#     def __init__(self):
#         self.fig = Figure()
#         self.ax = self.fig.add_subplot(111, projection="3d")
#         Canvas.__init__(self, self.fig)
#         Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
#         Canvas.updateGeometry(self)
#
# Matplotlib widget
# class MplWidget3D(MyMplCanvas3D):
#     def __init__(self, parent=None):
#         QtWidgets.QWidget.__init__(self, parent)  # Inherit from QWidget
#         self.canvas = MplCanvas3D()  # Create canvas object
#         self.vbl = QtWidgets.QVBoxLayout()  # Set box for plotting
#         self.vbl.addWidget(self.canvas)
#         self.setLayout(self.vbl)
#
#     def update(self, new_canvas):
#         self.vbl.replaceWidget(self.canvas, new_canvas)
#         self.setLayout(self.vbl)
#
#
#
# class MplWidget2D(MyMplCanvas):
#     """A canvas that updates itself every second with a new plot."""
#
#     def __init__(self, *args, **kwargs):
#         MyMplCanvas.__init__(self, *args, **kwargs)
#         timer = QtCore.QTimer(self)
#         timer.timeout.connect(self.update_figure)
#         timer.start(1000)
#
#     def compute_initial_figure(self):
#         self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')
#
#     def update_figure(self):
#         # Build a list of 4 random integers between 0 and 10 (both inclusive)
#         #l = [random.randint(0, 10) for i in range(4)]
#         x = []
#         y=[]
#         self.axes.cla()
#         self.axes.plot(x, y, 'r')
#         self.draw()
