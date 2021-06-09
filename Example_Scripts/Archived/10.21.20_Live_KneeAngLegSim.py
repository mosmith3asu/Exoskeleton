from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from Modules.Utils.Import_Utilis import *
from Modules.Plotting import RenderMesh
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from math import *
from time import sleep
from Modules.Utils.DataHandler import Data_Handler
data_handler = Data_Handler()
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

def initialize_quats(quats, init_quats, start_displace=0):

    thigh_quat = quats[0:4]
    femur_quat = quats[4:8]
    thigh_quat = Quaternion(thigh_quat).normalised
    femur_quat = Quaternion(femur_quat).normalised

    thigh_init_quat = init_quats[0:4]
    femur_init_quat = init_quats[4:8]
    thigh_init = rel_rot(Quaternion(thigh_init_quat))
    femur_init = rel_rot(Quaternion(femur_init_quat))

    init_displace_quat_thigh = Quaternion(axis=thigh_quat.axis, degrees=start_displace / 2.)
    init_displace_quat_femur = Quaternion(axis=femur_quat.axis, degrees=start_displace / 2.)

    thigh_quat = thigh_init * thigh_quat
    thigh_quat = init_displace_quat_thigh * thigh_quat

    femur_quat = femur_init * femur_quat
    thigh_quat = -init_displace_quat_femur * thigh_quat

    return thigh_quat,femur_quat

render_as_line = True  # Render line instead of leg STL
sim_enabled = False
knee_angles = []
hip_angles = []
shank_angles = []
# Object for importing serial data
compare_vec = [1,0,0]
ser = serial_quat("COM3")

print("Please keep knee at zero displacment")
print("Initializing in 3 seconds...")
sleep(3)
initial_data= ser.read_line()
thigh_init = initial_data[0:4]
femur_init = initial_data[4:8]

init_quats = np.append(thigh_init,femur_init)
print("Initialized")
print("Initial Quats:")
print(f"Tquat {thigh_init}")
print(f"Fquat {femur_init}\n")

# Create Figure
fig = plt.figure(figsize=plt.figaspect(0.5))
ax_kneeAngles = fig.add_subplot(3,1,1)
#ax_sim1 = fig.add_subplot(2, 2, 3, projection='3d')
ax_sim1 = fig.add_subplot(3, 2, 3)
ax_sim2 = fig.add_subplot(3, 2, 4)
ax_legsim = fig.add_subplot(3, 2, 5 ,projection='3d')


def update(i):
    # Loop through all data values
    # for plot_num in range(len(patient_displayed)):
    ax_sim1.clear()
    ax_sim2.clear()
    ax_kneeAngles.clear()

    ax_sim1.set_title('Hip')
    ax_sim2.set_title('Shank')
    ax_kneeAngles.set_title('Knee')
    #size = 0.7
    #ax_sim1.set_xlim(-size, size)
    #ax_sim1.set_ylim(-size, size)
    #ax_sim1.set_zlim(-size, size)
    #ax_sim2.set_xlim(-size, size)
    #ax_sim2.set_ylim(-size, size)
    #ax_sim2.set_zlim(-size, size)

    import_data= ser.read_line()
    print("Serial Import:",import_data)
    thigh_quat = import_data[0:4]
    femur_quat = import_data[4:8]
    # thigh_quat, femur_quat = initialize_quats(import_data, init_quats)
    # print(f'Tquat:{thigh_quat.elements}')
    # print(f'Fquat: {femur_quat.elements}\n')
    #
    # vec = np.array([1., 0., 0.])
    # v1 = thigh_quat.rotate(vec)
    # v2 = femur_quat.rotate(vec)
    # angle = angBetweenV(v1, v2)  # + start_displace
    # hip = angBetweenV(v1, compare_vec)
    # shank = angBetweenV(v2, compare_vec)
    #
    # hip_angles.append(hip)
    # if len(hip_angles) > 50:
    #     del hip_angles[0]
    #
    # shank_angles.append(shank)
    # if len(shank_angles) > 50:
    #     del shank_angles[0]
    #
    #
    # knee_angles.append(angle)
    # if len(knee_angles)>50:
    #     del knee_angles[0]
    #
    null_vec = [1,0,0]
    thigh_quat, femur_quat = data_handler.quat2KneeHipShank_initialize_quats(import_data, init_quats)
    v1 = thigh_quat.rotate(null_vec)
    v2 = femur_quat.rotate(null_vec)

    angle = data_handler.quat2KneeHipShank_angBetweenV(v1, v2)
    knee_angles.append(angle)
    if len(knee_angles)>50:
        del knee_angles[0]

    hip = data_handler.quat2KneeHipShank_angBetweenV(v1, null_vec)
    hip_angles.append(hip)
    if len(hip_angles) > 50:
        del hip_angles[0]

    shank = data_handler.quat2KneeHipShank_angBetweenV(v2, null_vec)
    shank_angles.append(shank)
    if len(shank_angles) > 50:
        del shank_angles[0]


    ax_sim1.plot(hip_angles) # bottom right
    ax_sim2.plot(shank_angles) # bottom left
    ax_kneeAngles.plot(knee_angles)
    # Simulation and gait phase
    if sim_enabled:
        ax_legsim.clear()
        # Intialize IMU data
        #thigh_quat = import_data[0:4]
        #femur_quat = import_data[4:8]


        # Convert ypr to rotation matrices
        #thigh_quat= thigh_quat.elements
        #femur_quat = femur_quat.elements
        #rotM_thigh = RenderMesh.quat2rotM(thigh_quat)
        #rotM_femur = RenderMesh.quat2rotM(femur_quat)
        # Render leg mesh and calculate statistics
        #LegMesh, hip, knee, ankle, kneeA, hipA = RenderMesh.generate_legmesh_Matrix(rotM_thigh, rotM_femur)

        LegMesh, hip, knee, ankle, kneeA, hipA = RenderMesh.generate_legmesh_Quat(thigh_quat, femur_quat)

        if render_as_line:
            x = [hip[0], knee[0]]
            y = [hip[1], knee[1]]
            z = [hip[2], knee[2]]
            ax_legsim.plot(x, y, z, linewidth=3.0)

            x = [knee[0], knee[0] + ankle[0]]
            y = [knee[1], knee[1] + ankle[1]]
            z = [knee[2], knee[2] + ankle[2]]
            ax_legsim.plot(x, y, z, linewidth=3.0)
        else:
            # Add Leg Mesh to plot
            for m in LegMesh:
                ax_legsim.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))
                ax_legsim.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))

        ax_legsim.set_zlim((-1,0.1))
        ax_legsim.set_ylim((-1, 1))
        ax_legsim.set_xlim((-1,1))




if __name__ == "__main__":
    anim = animation.FuncAnimation(fig, update, interval=1)
    plt.show()
