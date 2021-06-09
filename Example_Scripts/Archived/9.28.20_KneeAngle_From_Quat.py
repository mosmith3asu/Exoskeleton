from math import *
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from scipy import signal
from scipy.signal import butter, filtfilt
fig, ax = plt.subplots(3)

patient = 3
session = 3

data_path_csv = 'C:\\Users\\mason\\Desktop\\Patient_files\\Patient3S3.csv'

def angBetweenQuat(quat1, quat2):
    # Create an initizialation parameter to account for initial Knee Angle
    # This offset method is being improved through dynamic initialization

    # Create quaternion object for first quat
    quat1 = Quaternion(quat1)

    # Create quaternion object for second quat
    # This quaternion is the inverse rotation
    quat2 = Quaternion(quat2).inverse

    # Combine the rotations to get a net rotation
    qd = quat1*quat2

    # Rotate some vector to get a new vector
    vector = np.array([0.,0.,1.])
    vector_prime = qd.rotate(vector)

    # normalize vectors (redundant but safe)
    vector = vector / np.linalg.norm(vector)
    vector_prime = vector_prime  / np.linalg.norm(vector_prime)

    # Measure the angles between the vectors
    dot_product = np.dot(vector, vector_prime)
    angle = degrees(np.arccos(dot_product))

    # Transform data
    angle = angle

    return angle
def fix_inverted_quat(quat,axes_transform = np.array([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])):

    quat_matrix = np.array(quat.rotation_matrix)
    quat_matrix_trans = quat_matrix.dot(axes_transform)
    quat = Quaternion(matrix=quat_matrix_trans)
    return quat
def fix_data_withline(data, range):
    start,stop = range
    start_y = data[start]
    stop_y = data[stop]
    between_x = stop-start
    intermediate_vals = np.linspace(start_y,stop_y,num=between_x)

    i = 0
    for val in intermediate_vals:
        data[start+i]=val
        i=i+1
    return data
def fix_data_offsetline(data,offset_after_x,ref_x,manual_offset = 0):
    ref_y = data[ref_x]
    y = data[offset_after_x]
    diff = y - ref_y
    for i in range(len(data) - offset_after_x):
        data[i + offset_after_x] = data[i + offset_after_x] - diff + manual_offset

    data = fix_data_withline(data,(ref_x,offset_after_x))
    return data

if __name__ == "__main__":
    # Import data
    data = genfromtxt(data_path_csv, delimiter=',')

    # Remove header row
    data = data[1:,:]
    init_pts = []

    #initialize knee angle list
    knee_angles = []

    ref_i = 300
    thigh_quat_init = Quaternion(data[ref_i, 4:8])
    femur_quat_init = Quaternion(data[ref_i, 0:4])

    thigh_init = thigh_quat_init.inverse
    femur_init = femur_quat_init.inverse

    # iterate through data and calculate knee angle
    for line in range(len(data)):
        print(line,len(data))
        thigh_quat = Quaternion(data[line,4:8])*thigh_init
        femur_quat = Quaternion(data[line,0:4])*femur_init

        #thigh_quat = fix_inverted_quat(thigh_quat)
        femur_quat = fix_inverted_quat(femur_quat)
        angle = angBetweenQuat(thigh_quat, femur_quat)

        knee_angles.append(angle)

    # Plot raw knee angle data
    knee_angles_raw = knee_angles
    xs = np.arange(0,len(knee_angles)).reshape((-1,1))
    knee_angles_raw = np.array(knee_angles_raw).reshape((-1, 1))
    plot_data_raw = np.append(xs, knee_angles_raw, axis=1)

    ax[0].scatter(plot_data_raw[:,0], plot_data_raw[:,1],s=1)
    ax[0].set(title=f"Knee Angle Patient {patient} S{session} (Raw)")

    # Begin Fixing/Patching Data
    if session == 1:
        knee_angles = [-1*angle+30 for angle in knee_angles]

        knee_angles = fix_data_offsetline(knee_angles,2665,2617,manual_offset=7)
        knee_angles = fix_data_offsetline(knee_angles, 5646, 5633, manual_offset=-13)
        knee_angles = fix_data_offsetline(knee_angles, 7431, 7396, manual_offset=-7)
        knee_angles = fix_data_offsetline(knee_angles, 10818, 10803, manual_offset=20)
        knee_angles = fix_data_withline(knee_angles, (8167, 8180)) # asymptotic behavior
        knee_angles = fix_data_withline(knee_angles, (12143, 12170))  # asymptotic behavior
        knee_angles = fix_data_withline(knee_angles, (15466, 15496))  # asymptotic behavior
        knee_angles = fix_data_withline(knee_angles, (19096, 19131))  # asymptotic behavior
        knee_angles = fix_data_withline(knee_angles, (21935, 21965))  # asymptotic behavior


    if session == 3:
        for index in range(13695, len(knee_angles)):
            knee_angles[index] = -1. * knee_angles[index]+67
        knee_angles = fix_data_withline(knee_angles, (13684, 13725))  # asymptotic behavior
        knee_angles = fix_data_withline(knee_angles, (15556,15572))  # asymptotic behavior
        knee_angles = fix_data_withline(knee_angles, (26182, 26209))  # asymptotic behavior
        knee_angles = fix_data_withline(knee_angles, (38923,38943))  # asymptotic behavior
        knee_angles = fix_data_withline(knee_angles, (45851,45880))  # asymptotic behavior

    # format lists and plot  processed data
    knee_angles_filt = knee_angles # Save copy of knee angle list
    knee_angles = np.array(knee_angles).reshape((-1,1))
    plot_data = np.append(xs,knee_angles,axis=1)

    #ax[1].scatter(plot_data[:,0], plot_data[:,1],s=1)
    ax[1].plot(plot_data[:, 0], plot_data[:, 1],color="red")
    ax[1].set(title=f"Knee Angle Patient {patient} S{session} (Repaired) ")


    # Apply Butterworth Filter
    filter_order = 2
    w_n = 3
    f_s = 6000  # Sample freq

    #knee_angles = np.array(knee_angles).reshape((1, -1)).tolist()

    sos = signal.butter(filter_order, w_n, 'hp', fs=f_s, output='sos')
    knee_angles_filt = signal.sosfilt(sos, knee_angles_filt)

    if session == 3:
        knee_angles_filt = [angle + 12 for angle in knee_angles_filt]
        print(np.shape(knee_angles_filt))
        #knee_angles_filt = fix_data_withline(knee_angles_filt,(4100,11411))
        #knee_angles_filt = fix_data_withline(knee_angles_filt, (15508,21667))
        #knee_angles_filt = fix_data_withline(knee_angles_filt, (26776, 34707))
        #knee_angles_filt = fix_data_withline(knee_angles_filt, (34707,41422))
        #for i in range(4100,11411):
        #    knee_angles_filt[i] =0
        #for i in range(15542,21667):
        #    knee_angles_filt[i] =0

    knee_angles_filt = np.array(knee_angles_filt).reshape((-1, 1))
    plot_data_filt = np.append(xs, knee_angles_filt, axis=1)

    #ax[2].scatter(plot_data_filt[:, 0], plot_data_filt[:, 1], s=1)
    ax[2].plot(plot_data_filt[:, 0], plot_data_filt[:, 1],color="green")
    ax[2].set(title=f"Knee Angle Patient {patient} S{session} "
                    f"(Butterworth High Pass Filter f_s = {f_s},w_n={w_n},order={filter_order}) ")
    plt.show()



