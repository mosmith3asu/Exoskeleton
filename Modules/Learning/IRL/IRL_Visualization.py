from Modules.Learning.IRL.IRL_Tools import IRL_tools,load_obj
from Modules.Learning.IRL import IRL_maxent as maxent
from Modules.Utils.DataHandler import Data_Handler
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pylab as plt
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
#time_begin = datetime.now() # current date and time

##########################################################################
# TRAINING PARAMETERS ####################################################
# Enables
VERBOSE = True
SAVE_MODEL = True
WEIGHTED = False
#MODE = 'Test'
MODE = 'Train'

# Data
data_names= [
    #'trimmed_train_P3S2'
    'trimmed_train_P3S3'
         ]
test_data_names= [
    #'trimmed_test_P3S2'
    'trimmed_test_P3S3'
         ]

##########################################################################
# Set up Data Handler ####################################################
DataHandler = Data_Handler()
DataHandler.import_KneeAngle=False
#DataHandler.import_HipAngle=False
#DataHandler.import_ShankAngle=False
DataHandler.import_GRF=False

##########################################################################
# IMPORT DATA ############################################################
X_train,y_train = DataHandler.import_custom('trimmed_train_P3S3')
X_test,y_test = DataHandler.import_custom('trimmed_test_P3S3')
#DataHandler.verify_import_with_graph()
data=DataHandler.data_per_gaitphase(X_train,y_train)

def IRL_Visualization(IRL_name):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """
    global ani_fig
    global ani_ax
    global pax
    global IRL
    global reward_map
    global gp_names
    global ani_title
    global n_loops
    IRL = load_obj(IRL_name)
    reward_map = np.reshape(IRL.reward, (IRL.N_STATES, IRL.N_STATES, IRL.N_STATES))


    print(f'\n\n\nReward: {np.shape(IRL.reward)}')
    print(f'Reshaped Reward: {np.shape(np.reshape(IRL.reward,(IRL.N_STATES,IRL.N_STATES,IRL.N_STATES)))}')
    print(f'Policy: {np.shape(IRL.policy)}')
    IRL_2D_rewardmap()
    IRL_max_reward_plot()


    n_loops = 10
    gp_names = []
    for p in range(IRL.N_STATES):
        gp_names.append(f'\n(GP=[{int(p * (100 / IRL.N_STATES))}%,{int((p + 1) * (100 / IRL.N_STATES))}%])')
    ani_fig, ani_ax = plt.subplots()
    pax = ani_ax.pcolor(reward_map[0, :, :])
    ani_fig.colorbar(pax, ax=ani_ax)
    ani_title = ani_ax.text(0.5, 0.9, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                        transform=ani_ax.transAxes, ha="center")
    #pax = plt.pcolor(reward_map[0, :, :])
    IRL_animate_reward()

def IRL_2D_rewardmap(RESCALE_FEATURES=True):

    reward_map = np.reshape(IRL.reward, (IRL.N_STATES, IRL.N_STATES, IRL.N_STATES))
    # 2d heatmap
    plot_cols = 5
    plot_rows = 2
    r = -1

    gp_names = []
    for p in range(IRL.N_STATES):
        gp_names.append(f'GP=[{int(p*(100/IRL.N_STATES))}%,{int((p+1)*(100/IRL.N_STATES))}%]')

    feature_names=["Hip Displacement (Deg)","Shank Displacement (Deg)"]

    for i in range(1, IRL.N_STATES+1):
        c = (i - 1) % plot_cols
        # Set up plots
        ax = plt.subplot(plot_rows, plot_cols, i)
        ax.set_title(gp_names[i-1])
        ax.set_ylabel(feature_names[0])
        ax.set_xlabel(feature_names[1])
        ax.pcolor(reward_map[i-1,:,:])

        if RESCALE_FEATURES:
            scale_x = 1/IRL.state_scales[2]
            scale_y = 1/IRL.state_scales[1]
            ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(round(x / scale_x,0)))
            ax.xaxis.set_major_formatter(ticks_x)
            ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(round(x / scale_y,0)))
            ax.yaxis.set_major_formatter(ticks_y)

    plt.suptitle('Recovered Reward Map (S=[GP,HD,SD])')
    plt.subplots_adjust(hspace=0.3,wspace=0.3)

def IRL_max_reward_plot():
    N_STATES = IRL.N_STATES
    reward_map = np.reshape(IRL.reward, (IRL.N_STATES, IRL.N_STATES, IRL.N_STATES))
    # Maximum reward plot
    print("\n Max Reward vs Time")
    points=np.empty((0,4))
    for p in range(N_STATES):
        hs = np.argmax(reward_map[p])
        hidx = int(hs/N_STATES)
        sidx = hs%N_STATES
        r=np.max(reward_map[p])
        phs=np.array([[p,hidx,sidx,r]])
        points=np.append(points,phs,axis=0)

    print(points)
    y_buffer = 1.1

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title("Maximum State Reward vs Gait Phase")

    # Joint plotting
    color = 'blue'
    ax.plot(points[:,0],points[:,1]*IRL.state_scales[1],label="Hip State",linestyle='-',c=color)
    ax.plot(points[:, 0], points[:, 2]*IRL.state_scales[2],label="Shank State",linestyle=':',c=color)
    gp_names=np.linspace(0,100,N_STATES).astype(int)
    ax.set_xlabel("Interpolated Gait Phase (%)")
    ax.set_ylabel("Joint Displacement (Degrees)",color = color)
    xticks(np.arange(N_STATES + 1), gp_names)
    ax.tick_params(axis='y', labelcolor=color)
    ax.legend(loc='upper left')
    ax.set_ylim((ax.get_ylim()[0],ax.get_ylim()[1]*y_buffer))

    # Reward Value
    color = 'tab:red'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Reward Value', color=color)
    ax2.plot(points[:, 0], points[:, 3], label="Reward Value",c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    ax2.set_ylim((ax2.get_ylim()[0], ax2.get_ylim()[1] * y_buffer))

def update(frame):
    i=frame%(n_loops+3)
    if i<reward_map.shape[0]:
        ani_title.set_text(f'Recovered Reward Map {gp_names[i]}')
        pax = ani_ax.pcolor(reward_map[i, :, :])
    else:
        ani_title.set_text(f'Restarting...')
        pax = ani_ax.pcolor(np.zeros((IRL.N_STATES,IRL.N_STATES)))

    return pax,ani_title

def IRL_animate_reward(RESCALE_FEATURES=True):



    feature_names = ["Hip Displacement (Deg)", "Shank Displacement (Deg)"]
    ani_ax.set_ylabel(feature_names[0])
    ani_ax.set_xlabel(feature_names[1])
    ani_ax.pcolor(reward_map[0, :, :])

    if RESCALE_FEATURES:
        scale_x = 1 / IRL.state_scales[2]
        scale_y = 1 / IRL.state_scales[1]
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(round(x / scale_x, 0)))
        ani_ax.xaxis.set_major_formatter(ticks_x)
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(round(x / scale_y, 0)))
        ani_ax.yaxis.set_major_formatter(ticks_y)

    ani = FuncAnimation(ani_fig, update, frames = n_loops*(IRL.N_STATES+3), interval = 400,blit=True)

if __name__ == '__main__':
    IRL_name = '03_17_2021_IRL_P3S2P3S3_PHS_10sx20a_obj'
    IRL_Visualization(IRL_name)
    plt.show()