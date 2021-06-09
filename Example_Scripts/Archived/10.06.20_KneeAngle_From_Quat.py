from math import *

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt,savetxt
from numpy.linalg import norm as mag
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

from Modules.Utils import Data_Filters


def angBetweenQuat(quat1, quat2, CALC_FROM_SCALAR=False, RETURN_DEGREES=True):
    quat_prod = quat1 * quat2.inverse
    quat_prod = quat_prod.normalised

    if CALC_FROM_SCALAR:
        # Method 1 Scalar Value of Quatnernion
        scaler = abs(quat_prod.scalar)
        angle = 2 * acos(scaler)
    else:
        # Method 2 Magnitude of Quaternion Vector
        norm = quat_prod.vector
        norm = mag(norm)
        angle = 2 * asin(norm)

    if RETURN_DEGREES == True:
        return degrees(angle)
    else:
        return angle

    return angle


def angBetweenV(v1, v2):
    # check if there is an array of zeros at hip

    # calculate angel
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = degrees(np.arccos(dot_product))
    return angle


def rel_rot(quat, target_quat=Quaternion([1., 0., 0., 0.])):
    quat_init = target_quat * quat.inverse
    quat_init = quat_init.normalised
    return quat_init


def gen_knee_angles(data, show_uninit=False, show_init=True, init_pt=0,
                    start_displace=0, is_inverted=False):
    thigh_quats = data[:, 0:4]
    femur_quats = data[:, 4:8]

    if is_inverted:
        print("inverting")
        # for i in range(len(femur_quats)):
        # inverse_correction = Quaternion(axis=[1.0, 0.0, 0.0], degrees=180)
        # quat_axis = Quaternion(femur_quats[i, :]).axis
        # inverse_correction = Quaternion(axis=quat_axis, degrees=180)
        # femur_quats[i] = (inverse_correction * Quaternion(femur_quats[i, :])).elements

    knee_angles = []
    knee_angles_init = []

    for i in range(len(data)):
        thigh_quat = Quaternion(thigh_quats[i, :]).normalised
        femur_quat = Quaternion(femur_quats[i, :]).normalised
        vec = np.array([1., 0., 0.])

        # Uninitialized knee angles
        if show_uninit:
            v1 = thigh_quat.rotate(vec)
            v2 = femur_quat.rotate(vec)
            angle = angBetweenV(v1, v2)
            knee_angles.append(angle)
        else:
            knee_angles = np.arange(0, len(data))

        # Initializad knee angles
        if show_init:
            thigh_init = rel_rot(Quaternion(thigh_quats[init_pt, :]))
            femur_init = rel_rot(Quaternion(femur_quats[init_pt, :]))

            init_displace_quat_thigh = Quaternion(axis=thigh_quat.axis, degrees=start_displace / 2.)
            init_displace_quat_femur = Quaternion(axis=femur_quat.axis, degrees=start_displace / 2.)

            thigh_quat = thigh_init * thigh_quat
            thigh_quat = init_displace_quat_thigh * thigh_quat

            femur_quat = femur_init * femur_quat
            thigh_quat = -init_displace_quat_femur * thigh_quat

            vec = np.array([1., 0., 0.])
            v1 = thigh_quat.rotate(vec)
            v2 = femur_quat.rotate(vec)
            angle = angBetweenV(v1, v2)  # + start_displace
            knee_angles_init.append(angle)
            #
        else:
            knee_angles_init = np.arange(0, len(data))

    return knee_angles, knee_angles_init


if __name__ == "__main__":
    # Control Parameters
    filtered = True                 # whether to replace uninitialized knee angles with filtered knee angles
    n_iter = 20                     #lfilter parameter
    oreder=4                        #lfilter parameter

    PLOT_SHOE = False               # include plot with ground reaction forces overlay
    PLOT_MOCAP = True               # Include motion capture plat

    export_lastfiles_kneeAng = False # whether to export final angles to CSV only from last file in data_paths
    date="10_12_20"                  # appended to file name

    starting_displacment = 9  # what the patient started their knee displacment at

    # Import Data
    mocap_path = "C:\\Users\\mason\\Desktop\\Patient_files\\RawQuats\\Mocap_MocapTrial.csv"
    imu_path = "C:\\Users\\mason\\Desktop\\Patient_files\\RawQuats\\IMU_MocapTrial.csv"
    shoe_path = "C:\\Users\\mason\\Desktop\\Patient_files\\RawQuats\\Patient3S3_shoe.csv"

    data_paths = np.array([  # [{data path}, {initialization/reference point},{IMUs inverted}]
        ["C:\\Users\\mason\\Desktop\\Patient_files\\RawQuats\\Personal1.csv",150,False],
        ["C:\\Users\\mason\\Desktop\\Patient_files\\RawQuats\\Personal2.csv",271,False],
        ["C:\\Users\\mason\\Desktop\\Patient_files\\RawQuats\\Personal3_180.csv", 150, True],
        #["C:\\Users\\mason\\Desktop\\Patient_files\\Patient3S1.csv", 348, True],
        #["C:\\Users\\mason\\Desktop\\Patient_files\\Patient3S2.csv",27282, True],
        #["C:\\Users\\mason\\Desktop\\Patient_files\\Patient3S3.csv", 459, True]
    ])

    # Set up figure
    fig = plt.figure()

    # Plot if enabled
    if PLOT_SHOE:
        # plot shoe data
        ax = fig.add_subplot(111)
        # import data
        data = genfromtxt("C:\\Users\\mason\\Desktop\\Patient_files\\Patient3S3.csv", delimiter=',')

        # calc knee angles
        knee_angles, knee_angles_init = gen_knee_angles(data, show_uninit=True, init_pt=11035,
                                                        start_displace=0,
                                                        is_inverted=True)

        ax.plot(knee_angles)
        ax.plot(knee_angles_init)
        data = genfromtxt(shoe_path, delimiter=',')

        shoe_touching = []
        zeros = []
        for force1, force2, force3, force4 in data[:, 0:4]:
            if force2 >= 210 or force3 >= 210 or force4 >= 155:  # If shoe is touching
                # shoe_touching.append(max(knee_angles_init))
                shoe_touching.append(0)
            else:
                # shoe_touching.append(max(knee_angles_init))
                shoe_touching.append(max(knee_angles))
            zeros.append(0)

        t = np.arange(0, len(shoe_touching))
        ax.fill_between(t, shoe_touching, 0, alpha=0.5, color="orange")
        ax.plot(shoe_touching)
        ax.legend(["Knee Angles", "swing phase"])

        plt.show()

    if PLOT_MOCAP:
        ax = fig.add_subplot(111)
        # Plot Mocap
        data = genfromtxt(imu_path, delimiter=',')

        data = data[(2254 - 1115):, :]
        print(len(data))
        knee_angles, knee_angles_init = gen_knee_angles(data, show_uninit=True, init_pt=1957,  # 1660,
                                                        start_displace=8)
        scaled_t = np.arange(0, 9420, step=9420. / 9621.)
        ax.plot(scaled_t, knee_angles,linestyle=":")
        ax.plot(scaled_t, knee_angles_init)

        if filtered:
            knee_angles_filt = Data_Filters.lfilter(knee_angles_init, n_iter)
            ax.plot(scaled_t, knee_angles_filt)
            #knee_angles_init = Data_Filters.savgol_filter(knee_angles_init,2,5)

        data = genfromtxt(mocap_path, delimiter=',')
        print(len(data))
        v1 = [1., 0., 0.]
        converted_angles = []
        for x, y, z in data:
            # r = Rotation.from_rotvec(np.array([radians(x),radians(y),radians(z)]))
            r = Rotation.from_rotvec(np.array([radians(z), radians(y), radians(x)]))
            v2 = r.apply(v1)
            angle = angBetweenV(v1, v2)
            converted_angles.append(angle)

        ax.plot(converted_angles)
        if filtered:
            ax.legend(["IMU Raw", "IMU Adjusted Frame","IMU HP", "Mocap Converted"])
        else:
            ax.legend(["IMU Raw", "IMU Adjusted Frame", "Mocap Converted"])

        name = "Mocap"
        ax.set(ylabel='Knee Displacment', title=name)
        ax.grid()
        ax.set_xticks([], minor=True)

    # Iterate through list of paths
    for path, init_pt, is_inverted in data_paths:
        # convert init point ot int
        init_pt = int(init_pt)
        if is_inverted == "True":
            is_inverted = True
        else:
            is_inverted = False

        # add subplot
        n = len(fig.axes)
        for i in range(n):
            fig.axes[i].change_geometry(n + 1, 1, i + 1)
        ax = fig.add_subplot(n + 1, 1, n + 1)

        # import data
        data = genfromtxt(path, delimiter=',')
        # calc knee angles
        knee_angles, knee_angles_init = gen_knee_angles(data, show_uninit=True, init_pt=init_pt,
                                                        start_displace=starting_displacment,
                                                        is_inverted=is_inverted)

        ax.plot(knee_angles, linestyle=":")
        ax.plot(knee_angles_init)

        ax.legend(["Raw", "Adjusted Frame"])
        if filtered:
            knee_angles_filt = Data_Filters.lfilter(knee_angles_init, n_iter)
            ax.plot(knee_angles_filt)
            ax.legend(["Raw", "Adjusted Frame", "HP Filter"])
            #knee_angles_init = Data_Filters.savgol_filter(knee_angles_init, 2, 5)


        # Plot Data


        name = path.split("\\")[-1]
        ax.set(ylabel='Knee Displacment', title=name)
        ax.grid()
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        print(f'finished {name}')

    # Export knee angles if enabled
    if export_lastfiles_kneeAng:
        savetxt(date+'_kneeAngles_'+name, knee_angles_init, delimiter=',')
        print("exported csv")

    plt.show()
