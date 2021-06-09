from math import *

import numpy as np
from numpy.linalg import norm as mag
from pyquaternion import Quaternion

from Modules.Plotting import Simple_Plot
from Modules import Old_Data_Handler


def fix_data_withline(data, range):
    start, stop = range
    start_y = data[start, 1]
    stop_y = data[stop, 1]
    between_x = stop - start
    intermediate_vals = np.linspace(start_y, stop_y, num=between_x)

    i = 0
    for val in intermediate_vals:
        data[start + i, 1] = val
        i = i + 1
    return data

def angBetweenQuat(quat1, quat2, CALC_FROM_SCALAR=False, RETURN_DEGREES=True):
    # The quat product is expressed and the combination of two quaternions
    # the multiplying by the inverse yields the net rotation in the form of a quaternion
    quat1 = quat1.normalised
    quat2 = quat2.normalised
    quat_prod = quat1 * quat2.inverse
    quat_prod = quat_prod.normalised

    # Quaternions expressed as q = q_0 + q_vec = q_0 + ||q_vec|| q_unitvec
    # where q_0 is the scalar part of the quaternion and q_vec is the vector part
    #  q can the be expressed as q = cos(theta/2) + sin(theta/2) * q_unitvec
    # this gives 2 methods for calculating theta that give the same result

    if CALC_FROM_SCALAR:
        # Method 1 Scalar Value of Quatnernion
        scaler = abs(quat_prod.scalar)
        #scaler = quat_prod.scalar
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

if __name__ == "__main__":
    # Patient details
    patient = 3
    session = 3
    if patient == 3 and session == 1:
        imu_path = "C:\\Users\\mason\\Desktop\\Patient_files\Patient3S1.csv"
    if patient == 3 and session == 3:
        #imu_path = "C:\\Users\\mason\\Desktop\\Patient_files\Patient3S3_fixed.csv"
        imu_path ="C:\\Users\\mason\\Desktop\\Patient_files\\Patient3S2.csv"

    # Import Data
    data = Old_Data_Handler.trial_data()
    imu_data = data.import_custom_path(imu_path) #custom import from csv

    imu_x = np.array([[0, 0]])
    knee_angles = np.array([[0, 0]])
    zero_line = np.array([[0, 0]])  #plotting utility

    for index in range(len(imu_data)):
        # get the quaternion from array
        thigh_quat = Quaternion(imu_data[index, 4:8]).normalised
        femur_quat = Quaternion(imu_data[index, 0:4]).normalised

        # calculate knee angle and append to array
        angle = angBetweenQuat(thigh_quat, femur_quat)
        knee_angles = np.append(knee_angles, np.array([[index, angle]]), axis=0)
        zero_line = np.append(zero_line, np.array([[index, 0]]), axis=0)

    # Fix some unstable points
    #knee_angles = fix_data_withline(knee_angles, (11588,11640))  # asymptotic behavior
    #knee_angles = fix_data_withline(knee_angles, (11434,11513))  # asymptotic behavior
    #knee_angles = fix_data_withline(knee_angles, (15554,15572))  # asymptotic behavior
    #knee_angles = fix_data_withline(knee_angles, (26187, 26206))  # asymptotic behavior
    #knee_angles = fix_data_withline(knee_angles, (38924, 38957))  # asymptotic behavior
    #knee_angles = fix_data_withline(knee_angles, (45851, 45874))  # asymptotic behavior

    # Save knee angles as csv
    #savetxt('kneeAngles_P3S3.csv', knee_angles, delimiter=',')

    # Plotting
    title = f"Patient {patient} Sesssion {session}"
    c_x = 'blue'
    c_y = 'red'
    c_z = 'purple'
    imu_linestyle = '-'
    dashed = ':'
    plt_comps = (None, None, None)

    # I use custom plotting tools for simplicity.
    # You may plot with regular Matplotlib plot(knee_angles[:,0],knee_angles[:,1])
    plt_comps = Simple_Plot.plot2D(knee_angles, title, color=c_y, plt_components=plt_comps, linestyle=imu_linestyle)
    plt_comps = Simple_Plot.plot2D(zero_line, title, color="black", plt_components=plt_comps, linestyle=dashed)

    plt, fig, ax = plt_comps

    # x_range = (2200,2720)
    # x_range = (2200, 9000)
    # plt.xlim(x_range)

   #y_range = (-5, 50)
    y_range = (-50, 180)
    plt.ylim(y_range)

    plt.ylabel("Knee Displacment (Deg)")
    plt.xlabel("Time")

    plt.show()

