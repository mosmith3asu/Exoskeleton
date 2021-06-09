from Modules.Utils.DataHandler import Data_Handler
import matplotlib.pyplot as plt
from Modules.Plotting import InteractiveLegend
from numpy import genfromtxt
import numpy as np
import pickle
from Modules.Utils.Import_Utilis import add_show_fig, add_new_dir


##########################################################################
# GLOBAL CONTROL PARAMETERS###############################################

raw_quat_data = [
    #['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\10_16_20_LabIMU.csv', [0, 2068, 3765, 5814, 7824, 9499], False],
    #['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\IMU_test.csv', [0], False],
    #['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\IMU_test.csv', [0], True],
    #['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\IMU_MocapTrial.csv', [0], False],
    #['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\IMU_MocapTrial.csv', [0], True],
    #['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal1.csv', [0], False],
    #['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal2.csv', [268], False],
    #['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal3_180.csv', [0], False],
    #['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal3_180.csv', [0], True],

    #['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P3S1.csv', [0],False], #[0,786, 5228, 9453,13219, 16523, 19910],
    #['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P3S1.csv', [0],True], #[0,786, 5228, 9453,13219, 16523, 19910],
    #['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P3S3.csv', [0,10850, 18580, 40731],False],
    #['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P3S3.csv', [0,10850, 18580, 40731],True],

    ['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P4S1.csv', [0], True],
    ['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P4S2.csv', [1500,10364,13051,20501], True],
    ['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P4S3.csv', [560,2451], True],
]

# Plot Enables and Parameters
RAW_DATA = True
FILT_DATA = True
INIT_DATA = False
NUM_FEATURES =1
patient_num=-1
l_filter_iter = 2

# Calculation function parameters
FIRST_QUAT = 'thigh'
PLOT_MOCAP = True
plot_output = True
save_output = False
SAVE_PLOT = True
FOLDER_NAME = "ProcessedData"
if save_output or SAVE_PLOT: folder_path = add_new_dir(FOLDER_NAME)

figs=[]
figs_name=[]
DataHandler = Data_Handler()
DataHandler.output_dir = folder_path

def run(raw_quat_path,init_indexs,is_inverted):
    global patient_num
    global figs
    global figs_name

    data_name = raw_quat_path.split("\\")[-1].split(".")[0]

    # QUAT CALCULATION #######################################################
    KHP_filt,KHP_raw=DataHandler.quat2KneeHipShank(raw_quat_path,
                                                   init_idxs=init_indexs,
                                                   first_quat=FIRST_QUAT,
                                                   name_prefix=data_name,
                                                   plot_output=plot_output,
                                                   save_output=save_output,
                                                   filter_iter =l_filter_iter,
                                                   filtered=True
                                                   )

    # PLOTTING ###############################################################

    # Set up figure if its new patient
    try:
        new_patient_num = int(data_name.split("P")[-1].split("S")[0])
        fig_title = f'Patient {new_patient_num}'
    except:
        new_patient_num = 0
        fig_title = f'{data_name}'

    print(f'|\t P_num {patient_num}, New P_num {patient_num}')

    if new_patient_num!=patient_num:
        print(f'|\t NEW FIG')
        figs.append(plt.figure(fig_title))
        figs_name.append(fig_title)

    patient_num = new_patient_num

    # Plot data
    titles = ['Knee Displacement', 'Hip Displacement', 'Shank Displacment']
    legends = []
    axs=[]
    for col in range(NUM_FEATURES):
        # add subplot
        n = len(figs[-1].axes)
        for i in range(n):
            figs[-1].axes[i].change_geometry(n + 1, 1, i + 1)
        axs.append(figs[-1].add_subplot(n + 1, 1, n + 1))


        if RAW_DATA:
            axs[col].plot(KHP_raw[:,col], label = 'Raw Displacement' ) # Plot raw data

        if FILT_DATA:
            axs[col].plot(KHP_filt[:, col], label = 'Filtered Displacement') # Plot Filtered Data
            axs[col].vlines(init_indexs, figs[-1].axes[col].get_ylim()[0], figs[-1].axes[col].get_ylim()[1], colors='r', # Plot init points
                           linestyles=':', linewidth=1, label='Reinitialization Point')

        axs[col].set_title(titles[col]+f'   Data= {data_name}' + f'   Inverted = {is_inverted}')
        leg = axs[col].legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                            ncol=1, borderaxespad=0)
        legends.append(leg)
        figs[-1].subplots_adjust(right=0.8)

    return InteractiveLegend.create(legends)

def plot_mocap():
    global figs
    global patient_num
    patient_num=-1
    IMU = [['C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\IMU_MocapTrial.csv', [0], True]]
    MOCAP = 'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Mocap_MocapTrial.csv'

    mocap_data = genfromtxt(MOCAP, delimiter=',')
    for path, init_idxs, is_inverted in IMU:
        leg = run(path, init_idxs, is_inverted)

    mocap_peaks = [842,1124]
    mocap_diff = mocap_peaks[1]-mocap_peaks[0]
    imu_peaks = [1967,2255]
    imu_diff =imu_peaks[1]-imu_peaks[0]
    scale = imu_diff/mocap_diff
    offset = 1965-856
    t=np.arange(np.size(mocap_data[:,0])).reshape(-1,1)
    t=t*scale+offset
    figs[-1].axes[0].plot(t,mocap_data[:, 0],linestyle=":")


if __name__=="__main__":
    for path,init_idxs,is_inverted in raw_quat_data:
        leg = run(path,init_idxs,is_inverted)

    if PLOT_MOCAP: plot_mocap()

    if SAVE_PLOT:
        add_show_fig(folder_path)
        i=0
        for fig in figs:
            from datetime import datetime
            time_begin = datetime.now()  # current date and time
            date = time_begin.strftime("%m_%d_%Y")

            dir = DataHandler.output_dir
            name = f'\\{date}_fig_{figs_name[i]}.pkl'
            path = dir + name
            with open(path, 'wb') as fig_file:
                pickle.dump(fig, fig_file)
            i +=1

    plt.show()

