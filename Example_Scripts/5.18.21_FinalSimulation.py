import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sin,cos
from Modules.Utils import DataHandler

###################################################
# START PARAMS - CHANGE THESE VARIABLES
###################################################
plot_import = True      # Plot the imported & processed joint angles
filtered_import = True  # Filter the imported & processed joint angles

save_dir = "C:\\Users\\mason\\Desktop\\Thesis\\Patient_files" # where all saved files will be stored
save_import = False      # Save the imported & processed joint angles as csv
save_animation = False   # Save simulation as mp4
show_animation = True  # Whether to show animation after save

# Simulation Appearance/Speed
leg_thickness = 2
leg_color = 'k'
joint_thickness = 20
joint_color = 'r'
footsize = 0.25
foot_dir = -1

sim_size = 2.5       # How big leg simulation window is
window_range = 1000  # How many indexes in the future to view
sampling_freq = 10   # assign to sample rate (ms) for real time if index_speed=1
index_speed = 1      # Index multiplier to speed up simulation by skipping points

"""
Setting Data Parameters
    -UNCOMMENT/CHANGE THE BELOW LIST (data_param) TO SELECT AND MODIFY PROCESSED DATA
    -Recomended settings are already defined
    -Structure of data parameters:
        data_param = [path,init_indexs,filter_iter]
    
Variables:
    *path*:         path to raw IMU data [quaternion,quaternion]
                    *Need to modify to include the path on your local directory*
                        
    init_idexs:     list of initialization points where knee set to zero flexion
                    Some trials initialization point is not stable so it is skipped
                        
    filter_iter:    iteration for linear filter. Higher value = smoother data
"""
data_param = [
    'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal1.csv',      [200,1431,2562,3601],5
    #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal2.csv',      [268,2966,5349,7535],10
    #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\Personal3_180.csv',  [429,3280,6838,9434],10
    #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\SelfCollectedData\\IMU_MocapTrial.csv',  [3176],10

    #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P3S1.csv', [624],10 #[624, 5390, 9453, 16523, 19910]
    #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P4S2.csv', [0],10
    #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P3S3.csv', [11150, 18580, 40731],10

    #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P4S1.csv', [0],10
    #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P4S2.csv', [2210,10364,13051,20501],10
    #'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files\\RawData\\IMU_P4S3.csv', [560,2451],10
]
###################################################
# END PARAMS - CHANGE THESE VARIABLES
###################################################














##################################################3
# Calculate trajectories
###################################################
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\ffmpeg-20200831-4a11a6f-win64-static\\bin\\ffmpeg.exe'
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

path = 0            # named int, do not change
init_idxs = 1       # named int, do not change
filter_iter = 2     # named int, do not change
kneen = 0           # named int, do not change
hipn = 1            # named int, do not change
shankn = 2          # named int, do not change
deg2rad = 3.14/180.0

DataHandler = DataHandler.Data_Handler(root_dir=save_dir) # Initialize data handler
data_name = data_param[path].split("\\")[-1].split(".")[0]
KHS_filt = DataHandler.quat2KneeHipShank(quat_csv_path=data_param[path],
                                         init_idxs=data_param[init_idxs],
                                         filter_iter=data_param[filter_iter],
                                         name_prefix=data_name,
                                         plot_output=plot_import,
                                         save_output=save_import,
                                         filtered=filtered_import)

###################################################
# SIM INIT
###################################################

# Parameters (CAN PLAY WITH SETTINGS TO FIND RIGHT SIM)
index_start = data_param[init_idxs][0]      # Starting index used as initial/reference frame

n_frames = int((np.shape(KHS_filt)[0]-index_start)/index_speed)  # Number of times the animation updates

# Unpack Data
knee_angles = KHS_filt[:,kneen].tolist()
hip_angles = KHS_filt[:,hipn].tolist()
shank_angles = KHS_filt[:,shankn].tolist()

######################################################################
# SET UP PLOTS #######################################################
######################################################################

# Create Figure and plots
fig = plt.figure(figsize=plt.figaspect(0.5))
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

ax = fig.add_subplot(2, 2, 1)
ax_KA = fig.add_subplot(2, 2, 2)
ax_HA = fig.add_subplot(2, 2, 4)
ax_SA = fig.add_subplot(2, 2, 3)

ax_KA.set_ylim((min(-2,min(knee_angles)),min(40,max(knee_angles))))
ax_HA.set_ylim((min(-20,min(hip_angles)),min(40,max(hip_angles))))
ax_SA.set_ylim((min(-20,min(shank_angles)),min(40,max(shank_angles))))

# Set Titles
ax.set_title("Leg Simulation")
ax_KA.set_title("Knee Displacment vs Time")
ax_HA.set_title("Hip Displacment vs Time")
ax_SA.set_title("(NOT USED IN SIM) Shank Displacment vs Time")

# Plot Joint Angles
ax_KA.plot(knee_angles)
ax_HA.plot(hip_angles)
ax_SA.plot(shank_angles)

# Plot initialization indexes (verticle red lines)
init_color ='r'
init_idxs = data_param[1]
ax_KA.vlines(init_idxs, min(knee_angles), max(knee_angles), colors=init_color, linestyles=':',label = "Init Pt")
ax_HA.vlines(init_idxs, min(hip_angles), max(hip_angles), colors=init_color, linestyles=':')
ax_SA.vlines(init_idxs, min(shank_angles), max(shank_angles), colors=init_color, linestyles=':')

# Plot zero displacement line
ax_KA.plot([0 for pt in range(len(knee_angles))], c='k', linewidth=1,label = "Joint F/E")
ax_HA.plot([0 for pt in range(len(hip_angles))], c='k', linewidth=1)
ax_SA.plot([0 for pt in range(len(shank_angles))], c='k', linewidth=1)

ax_KA.legend()
annote_loc = 0.8*sim_size

def perpendicular_vector(v):
    ax = v[0]
    ay= v[1]
    vperp = [-ay,ax]
    return vperp

def update(i):
    global list_KA
    global init_ang

    # Get index references
    index = index_start + i * index_speed

    ############################################
    ## Simulation ##############################
    ############################################
    ax.clear()
    ax.set_title(f'Leg Simulation {data_name}')
    ax.text(annote_loc, annote_loc, f"Index={index}")
    ax.set_xlim(-sim_size/2, sim_size/2)
    ax.set_ylim(-sim_size, 0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    # Calculate Joint Locations

    hip = [0,0]
    vknee = [sin(hip_angles[index]*deg2rad), cos(hip_angles[index]*deg2rad)]
    uvknee = (-1*np.array(vknee) / np.linalg.norm(np.array(vknee))).tolist()
    knee = [uvknee[0]+hip[0], uvknee[1] + hip[1]]


    vankle = [sin((hip_angles[index]-knee_angles[index])* deg2rad),
              cos((hip_angles[index]-knee_angles[index])* deg2rad)]
    uvankle = (-1*np.array(vankle) / np.linalg.norm(np.array(vankle))).tolist()
    ankle = [uvankle[0]+uvknee[0], uvankle[1]+uvknee[1]]

    vfoot = perpendicular_vector(uvankle)
    vfoot = (foot_dir*footsize * np.array(vfoot) / np.linalg.norm(np.array(vfoot))).tolist()
    foot = [ankle[0]+vfoot[0],ankle[1]+vfoot[1]]

    # Plot Leg in simulation
    ax.scatter([hip[0], knee[0],ankle[0]], [hip[1], knee[1],ankle[1]], s=joint_thickness,c=joint_color)
    ax.plot([hip[0],knee[0]],[hip[1],knee[1]],linewidth = leg_thickness,c=leg_color)
    ax.plot([knee[0], ankle[0]], [knee[1], ankle[1]],linewidth = leg_thickness,c=leg_color)
    ax.plot([ankle[0], foot[0]], [ankle[1], foot[1]],linewidth = leg_thickness,c=leg_color)


    # Change Viewing Window
    ax_KA.set_xlim(index, index+window_range)
    ax_HA.set_xlim(index, index+window_range)
    ax_SA.set_xlim(index, index+window_range)


if __name__ == "__main__":
    anim = animation.FuncAnimation(fig, update, interval=sampling_freq, frames=n_frames)
    if save_animation:
        print('\n\nSaving Animation... Please Wait...')

        from datetime import datetime
        time_begin = datetime.now()  # current date and time
        date = time_begin.strftime("%m_%d_%Y")
        name = f'{save_dir}\\Outputs\\{date}_Simulation_{data_name}'
        anim.save(f'{name}.mp4',writer=writer)
        print(f'Animation Saved to: {name}')
    if show_animation:
        plt.show()
