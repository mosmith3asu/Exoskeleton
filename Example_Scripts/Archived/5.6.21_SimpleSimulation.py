import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from numpy import genfromtxt
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sin,cos
from Modules.Plotting import RenderMesh

plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\ffmpeg-20200831-4a11a6f-win64-static\\bin\\ffmpeg.exe'

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# Parameters (CAN PLAY WITH SETTINGS TO FIND RIGHT SIM)
index_start = 1000      # Starting index used as initial/reference frame
index_speed = 1         # Index multiplier to speed up simulation by skipping points
render_as_line = True   # Render line instead of leg STL

# Constants
n_frames = 5000             # Number of times the animation updates
patient_displayed = [0, 1]  # what data to be displayed
sim_enabled = True

# Import data
patient_data = []
import_data = genfromtxt("C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal1.csv",delimiter=',')


# Trackers

init_ang = None


list_KA = np.empty((0,2))
list_HA = np.empty((0,2))

# Create Figure
fig = plt.figure(figsize=plt.figaspect(0.5))
# ax = fig.add_subplot(2, 2, 1, projection='3d')
ax = fig.add_subplot(2, 2, 1)
ax_SA = fig.add_subplot(2, 2, 2)
ax_HA = fig.add_subplot(2, 2, 3)
ax_KA = fig.add_subplot(2, 2, 4)
size = 2
annote_loc = 0.8*size
deg2rad = 3.14/180.0
def trim(array):
    max_len = 100
    if len(array[:, 1]) > max_len:
        array = np.delete(array, 0, 0)
    return array

def update(i):
    global list_KA
    global init_ang

    # Get index references
    index = index_start + i * index_speed

    ############################################
    ## Simulation ##############################
    ############################################
    ax.clear()
    ax.text(annote_loc, annote_loc, f"Index={index}")

    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    patient = patient_data

    # Simulation and gait phase
    if sim_enabled:
        ax.set_title("Leg Simulation)")
        # Intialize IMU data
        thigh_quat = import_data[index, 0:4]
        shank_quat = import_data[index, 4:8]
        #ypr_femur = patient[index, 1:4] - patient[index_start_l[patient_num], 1:4]

        # Convert ypr to rotation matrices
        ypr_thigh = RenderMesh.quat2ypr(thigh_quat)
        ypr_shank = RenderMesh.quat2ypr(shank_quat)
        hipA = ypr_thigh[1]
        shankA = ypr_shank[1]

        if init_ang == None:
            init_ang=[hipA,shankA]
        # Render leg mesh and calculate statistics
        hip = [0,0]
        knee = [-1*sin(hipA*deg2rad), -1*cos(hipA*deg2rad)]
        ankle = [-1*sin(shankA*deg2rad)+knee[0], cos(shankA*deg2rad)+knee[1]]

        ax.plot([hip[0],knee[0]],[hip[1],knee[1]])
        ax.plot([knee[0], ankle[0]], [knee[1], ankle[1]])

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
  #  list_KA =trim(np.append(list_KA, np.array([[index, kneeA]]), axis=0))
  #  ax_KA.plot(list_KA[:,0],list_KA[:,1])

    # Hip Angle
  #  list_HA = trim(np.append(list_KA, np.array([[index, hipA]]), axis=0))


if __name__ == "__main__":
    anim = animation.FuncAnimation(fig, update, interval=1, frames=n_frames)
    #anim.save('filename.mp4',writer=writer)
    plt.show()
