# from Modules.Interface.Data_Validation.Qt_DataValidation_Display import ui

import matplotlib

matplotlib.use('Qt5Agg')
from PyQt5 import QtCore
from Modules import Old_Data_Handler
from Modules.Plotting import RenderMesh
import time
import numpy as np

not_init = True
not_imported = True

index_start = 0
index = index_start
index_speed = 20

data = Old_Data_Handler.trial_data()
groundreaction_data = np.array([[0, 0, 0, 0, 0]])
kneeAng_array = []
debug = False


def Qt_backend(ui):
    global index
    global index_speed
    global groundreaction_data
    global kneeAng_array
    global not_init
    global not_imported
    global index_start
    global index_speed
    global not_imported

    try:
        # Check if widgets are enabled
        sim_enabled = ui.simulation_enable.isChecked()
        ps_enabled = ui.pressuresensor_enable.isChecked()
        gr_enabled = ui.groundreaction_enable.isChecked()
        run_enable = ui.runfile_button.isChecked()

        # initialization of ui
        if index == index_start and not_init:
            ui.file_names_combo.addItems(data.csv_names)
            ui.file_names_combo.setCurrentIndex(0)
            ui.runfile_button.setCheckable(True)

            ui.indexspeed_spinner.setValue(index_speed)
            ui.indexstart_spinner.setValue(index_start)

            ui.importfromserial_frame.hide()
            not_init = False

        # if simulation is not running, reinitialize
        if run_enable == False:
            not_imported = True
            index = index_start
            groundreaction_data = np.array([[0, 0, 0, 0, 0]])
            kneeAng_array = []

            index_speed = ui.indexspeed_spinner.value()
            index_start = ui.indexstart_spinner.value()
            ui.index_display.setText(str(index_start))

        if run_enable:
            patient_index = ui.file_names_combo.currentIndex()
            patient = data.patient_data[patient_index]

            # Move index forward at specified speed
            if index <= len(patient) - index_speed - 1:
                index = index + index_speed

            # Simulation and gait phase
            if sim_enabled:
                # Intialize IMU data
                # MOVE INITIALIZATION TO ROT MATRIX AND DOT THE TWO
                quat_thigh = patient[index, 1:5] + ([1., 0., 0., 0.] - patient[1, 1:5])
                quat_femur = patient[index, 5:9] + ([1., 0., 0., 0.] - patient[1, 5:9])
                print("qaut thigh",quat_thigh)

                # Convert quaternions to rotation matrices
                rotM_thigh = RenderMesh.quat2rotM(quat_thigh)
                rotM_femur = RenderMesh.quat2rotM(quat_femur)

                # Flip axes for display
                init_rotM = np.array([[0., -1., 0.],
                                      [1., 0., 0.],
                                      [0., 0., -1.]])

                rotM_thigh = rotM_thigh.dot(init_rotM)
                rotM_femur = rotM_femur.dot(init_rotM)

                hipAng, kneeAng = ui.simulation_widget.update_figure_sim(rotM_thigh, rotM_femur)

                # calculated knee angle and add to cumulative array
                kneeAng = 90 - kneeAng
                kneeAng_array.append(kneeAng)
                ui.gaitphase_widget.update_figure_gaitphase(kneeAng_array)

                # add hip and knee angles to interface display
                hipAng = str(int(hipAng))
                kneeAng = str(int(kneeAng))
                ui.kneeA_display.setText(kneeAng)
                ui.hipA_display.setText(hipAng)

            # Pressure Sensor
            if ps_enabled:
                sensor_data_array = patient[index, 9:37]
                sensor_data = sensor_data_array.tolist()
                net_force_thigh = ui.pressuresensor_thigh_widget.update_figure_pressuresensor(sensor_data, "thigh")
                net_force_femur = ui.pressuresensor_femur_widget.update_figure_pressuresensor(sensor_data, "femur")
                ui.thighforce_display.setText(net_force_thigh)
                ui.femurforce_display.setText(net_force_femur)

            # Ground Reaction Forces
            if gr_enabled:
                groundreaction_data_new = np.array([patient[index, 37:43]])
                groundreaction_data = np.append(groundreaction_data, groundreaction_data_new, axis=0)
                ui.groundreaction_widget.update_figure_groundforces(groundreaction_data)

            ui.index_display.setText(str(index))
    except ValueError as error:
        print(error)


class Qt_ThreadClass(QtCore.QThread):
    # Create the signal
    # sig = QtCore.pyqtSignal(int)

    def __init__(self, ui):
        self.ui = ui
        parent = None
        super(Qt_ThreadClass, self).__init__(parent)
        self.thread_delay = 0.1
        print("class thread init")

    def run(self):
        while True:
            Qt_backend(self.ui)
            time.sleep(self.thread_delay)
