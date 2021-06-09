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
index_start = 1000      # Starting index used as initial/reference frame
index_speed = 6         # Index multiplier to speed up simulation by skipping points
render_as_line = True   # Render line instead of leg STL

# Constants
n_frames = 5000             # Number of times the animation updates
patient_displayed = [0, 1]  # what data to be displayed
sim_enabled = True

# Import data
patient_data = []
import_data = genfromtxt("C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal1.csv",delimiter=',')


# Trackers
list_KA = np.empty((0,2))

# Create Figure
fig = plt.figure(figsize=plt.figaspect(0.5))
# ax = fig.add_subplot(2, 2, 1, projection='3d')
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax_SA = fig.add_subplot(2, 2, 2)
ax_HA = fig.add_subplot(2, 2, 3)
ax_KA = fig.add_subplot(2, 2, 4)
size = 0.7
annote_loc = 0.8*size
def trim(array):
    max_len = 100
    if len(array[:, 1]) > max_len:
        array = np.delete(array, 0, 0)
    return array

def update(i):
    global list_KA
    # Get index references
    index = index_start + i * index_speed

    ############################################
    ## Simulation ##############################
    ############################################
    ax.clear()
    ax.text(annote_loc, annote_loc, annote_loc, f"Index={i}")

    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_zlim(-size, size)
    patient = patient_data

    # Simulation and gait phase
    if sim_enabled:
        ax.set_title("Leg Simulation)")
        # Intialize IMU data
        thigh_quat = import_data[index, 0:4]
        femur_quat = import_data[index, 4:8]
        #ypr_femur = patient[index, 1:4] - patient[index_start_l[patient_num], 1:4]

        # Convert ypr to rotation matrices
        # rotM_thigh = RenderMesh.ypr2rotM(ypr_thigh)
        # rotM_femur = RenderMesh.ypr2rotM(ypr_femur)
        rotM_thigh = RenderMesh.quat2rotM(thigh_quat)
        rotM_femur = RenderMesh.quat2rotM(femur_quat)

        # Render leg mesh and calculate statistics
        LegMesh, hip, knee, ankle, kneeA, hipA = RenderMesh.generate_legmesh_Matrix(rotM_thigh, rotM_femur)

        if render_as_line:
            x = [hip[0], knee[0]]
            y = [hip[1], knee[1]]
            z = [hip[2], knee[2]]
            ax.plot(x, y, z, linewidth=3.0)

            x = [knee[0], knee[0] + ankle[0]]
            y = [knee[1], knee[1] + ankle[1]]
            z = [knee[2], knee[2] + ankle[2]]
            ax.plot(x, y, z, linewidth=3.0)
        else:
            # Add Leg Mesh to plot
            for m in LegMesh:
                ax.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))

    ############################################
    ## Joint Caclulations ##############################
    ############################################
    ax_SA.clear()
    ax_KA.clear()
    ax_HA.clear()

    ax_SA.set_title("Shank Displacment vs Time")
    ax_KA.set_title("Knee Displacment vs Time")
    ax_HA.set_title("Hip Displacment vs Time")

    # Knee Angle
    list_KA =trim(np.append(list_KA, np.array([[index, kneeA]]), axis=0))
    ax_KA.plot(list_KA[:,0],list_KA[:,1])

    # Hip Angle
    list_HA = trim(np.append(list_KA, np.array([[index, hipA]]), axis=0))


if __name__ == "__main__":
    anim = animation.FuncAnimation(fig, update, interval=1, frames=n_frames)
    #anim.save('filename.mp4',writer=writer)
    plt.show()
