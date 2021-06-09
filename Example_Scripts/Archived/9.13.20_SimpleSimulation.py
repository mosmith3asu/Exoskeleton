import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from numpy import genfromtxt
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Modules.Plotting import RenderMesh

plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\ffmpeg-20200831-4a11a6f-win64-static\\bin\\ffmpeg.exe'

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# Parameters (CAN PLAY WITH SETTINGS TO FIND RIGHT SIM)
index_start_l = [1000, 6000]  # Starting index used as initial/reference frame
# index_start_l = [1000, 6000]  # Decent reference points
index_speed_l = [6, 6]  # Index multiplier to speed up simulation by skipping points
render_as_line = False  # Render line instead of leg STL

# Constants
n_frames = 5000  # Number of times the animation updates
patient_displayed = [0, 1]  # what data to be displayed
sim_enabled = True

# Import data
patient_data = []
# import_data = genfromtxt('C:\\Users\\mason\\Desktop\\ML_MasonSmithGit\\Verification\\Patient2_ypr_backup_s1.csv', delimiter=',')
#import_data = genfromtxt("C:\\Users\\mason\\Desktop\\Patient_files\Patient3S3_fixed.csv",
#                         delimiter=',')
import_data = genfromtxt("C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal1.csv",
                         delimiter=',')


import_data = import_data[1:, :]
print(np.shape(import_data))
patient_data.append(import_data)
import_data = genfromtxt('C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\IMU_MocapTrial.csv',
                         delimiter=',')
import_data = import_data[1:, :]
patient_data.append(import_data)
print(np.shape(import_data))

# Create Figure
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = []
ax.append(fig.add_subplot(2, 2, 1, projection='3d'))
ax.append(fig.add_subplot(2, 2, 2, projection='3d'))
ax.append(fig.add_subplot(2, 2, 3))
ax.append(fig.add_subplot(2, 2, 4))

patient_gaitphase = []
patient1_KneeA = np.array([[index_start_l[0], 0]])
patient2_KneeA = np.array([[index_start_l[1], 0]])
patient_gaitphase.append(patient1_KneeA)
patient_gaitphase.append(patient2_KneeA)


def update(i):
    global patient_gaitphase
    # Loop through all data values
    # for plot_num in range(len(patient_displayed)):
    plot_num = 0
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    ax[3].clear()
    ax[0].set_title("Patient 1 s3: Low Impairment \n(Date:7.10.20)")
    ax[1].set_title("Patient 2 s3: High Impairment \n(Date:8.7.20)")
    ax[2].set_title("Patient 1 s3: Knee Displacment vs Time")
    ax[3].set_title("Patient 2 s3: Knee Displacment vs Time")

    for patient_num in patient_displayed:
        # Configure plot
        size = 0.7
        ax[plot_num].set_xlim(-size, size)
        ax[plot_num].set_ylim(-size, size)
        ax[plot_num].set_zlim(-size, size)
        patient = patient_data[patient_num]

        # Get index references
        index_start = index_start_l[plot_num]
        index_speed = index_speed_l[plot_num]
        index = index_start + i * index_speed

        # Simulation and gait phase
        if sim_enabled:
            # Intialize IMU data
            ypr_thigh = patient[index, 4:7] - patient[index_start_l[patient_num], 4:7]
            ypr_femur = patient[index, 1:4] - patient[index_start_l[patient_num], 1:4]

            # Convert ypr to rotation matrices
            rotM_thigh = RenderMesh.ypr2rotM(ypr_thigh)
            rotM_femur = RenderMesh.ypr2rotM(ypr_femur)

            # Render leg mesh and calculate statistics
            LegMesh, hip, knee, ankle, kneeA, hipA = RenderMesh.generate_legmesh_Matrix(rotM_thigh, rotM_femur)

            if render_as_line:
                x = [hip[0], knee[0]]
                y = [hip[1], knee[1]]
                z = [hip[2], knee[2]]
                ax[plot_num].plot(x, y, z, linewidth=3.0)

                x = [knee[0], knee[0] + ankle[0]]
                y = [knee[1], knee[1] + ankle[1]]
                z = [knee[2], knee[2] + ankle[2]]
                ax[plot_num].plot(x, y, z, linewidth=3.0)
            else:
                # Add Leg Mesh to plot
                for m in LegMesh:
                    ax[plot_num].add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))

            # Add new point to array
            new_point = np.array([[index, kneeA]])
            patient_gaitphase[patient_num] = np.append(patient_gaitphase[patient_num], new_point, axis=0)

            if len(patient_gaitphase[patient_num][:, 1]) > 100:
                patient_gaitphase[patient_num] = np.delete(patient_gaitphase[patient_num], 0, 0)

            ax[patient_num + 2].plot(patient_gaitphase[patient_num][:, 0], patient_gaitphase[patient_num][:, 1],
                                     color="blue")
        plot_num = plot_num + 1
        print(f'p{patient_num}: kneeA({kneeA}) i({index}) len:{len(patient_gaitphase[patient_num][:, 1])}')
        print(f'progress: {i}/{n_frames}')

    print()


if __name__ == "__main__":
    anim = animation.FuncAnimation(fig, update, interval=10, frames=n_frames)
    #anim.save('filename.mp4',writer=writer)
    plt.show()
