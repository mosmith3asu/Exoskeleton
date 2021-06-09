import numpy as np
from datetime import datetime
from Modules.Utils.DataHandler import Data_Handler
import matplotlib.pyplot as plt
from Modules.Plotting import InteractiveLegend
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np
##########################################################################
##########################################################################
# Data:             Trimmed training data
# Weighted Ratings: True
# Features:         Gaitphase, Hip Angle, Shank Angle
# Label:            Torque
##########################################################################
##########################################################################
time_begin = datetime.now() # current date and time

##########################################################################
# GLOBAL CONTROL PARAMETERS###############################################
# Kernel

# Enables
VERBOSE = True
SAVE_MODEL = True

# Data
data_names= [
    #'processed_P3S3L1',
    #'processed_P3S3L2',
    #'processed_P3S3L3',
    'trimmed_train_P3S2',
    'trimmed_test_P3S2',
    'trimmed_train_P3S3',
    'trimmed_test_P3S3'
         ]
print(f'Data Summary:\n {data_names}')

##########################################################################
# IMPORT DATA ############################################################
DataHandler = Data_Handler()



# ARRAY INDICIES
GAITPHASE = 0
KNEE_ANGLE = 1
HIP_ANGLE = 2
SHANK_ANGLE = 3
GRF1 = 4
GRF2 = 5
GRF3 = 6
GRF4= 7
TORQUE = 8

subphases = {'durations':[np.linspace(0,5),
                          np.linspace(5,19),
                     np.linspace(19,50),
                     np.linspace(50,81),
                     np.linspace(81,100)],
             #'colors':  ['blue','orange','green','red','purple']
             'colors':  ['white','lightgrey','white','lightgrey','white'],
             'labels': ['IC','LR','MS','TS','PS']}


features_names = DataHandler.print_features()
features_names.append('Torque')
feature_scales = [1,1,1,1]
var_param = [['Heel Strike',1,False],
             ['Knee Angle',1,True],
             ['Hip Angle',1,True],
             ['Shank Angle',1,True],
             ['GRF1',0.1,False],
             ['GRF2',0.1,False],
             ['GRF3',0.1,False],
             ['GRF4',0.1,False],
             ['Torque',20,True]]
#if VERBOSE:print("\nShape of raw session data:",np.shape(session_data))
def main():
    i_fig = 1
    #session_features()
    gaitphase_features()

    plt.show()
def get_HS_idxs(session_data):
    HS_idxs = []
    for i_session in range(len(session_data)):
        session = session_data[i_session]
        phases = session[:, 0]
        HS = [0]
        for p in range(len(phases) - 1):
            if phases[p + 1] < phases[p]: HS.append(p + 1)
        HS.append(np.size(phases) - 1)
        HS_idxs.append(HS)
    return HS_idxs

##########################################################################
# SESSION PREVIEW ########################################################
def session_features():
    global HS_idxs
    global i_fig
    fig1 = plt.figure(f'Figure {i_fig}: All Features per Session')
    fig1.title = 'Session Preview'
    # var_param = [[Name,Scale Output, Enable]]
    session_data = []
    for name in data_names:
        X, y = DataHandler.import_custom(name)
        new_data = np.append(X, y, axis=1)
        session_data.append(new_data)
    HS_idxs = get_HS_idxs(session_data)

    y_range = (0,60)
    axs1=[]
    legends1 = []

    print(np.shape(session_data))
    for i_session in range(len(session_data)):
        # Add subplot
        n = len(fig1.axes)
        for i in range(n):
            fig1.axes[i].change_geometry(n + 1, 1, i + 1)
        axs1.append(fig1.add_subplot(n + 1, 1, n + 1))




        # Plot with lines at heel strike indexs
        session = session_data[i_session]
        phases = session[:, 0]

        axs1[i_session].vlines(HS_idxs[i_session], min(y_range), max(y_range), colors='r',
                               linestyles=':', linewidth=1, label=features_names[0])

        for var_n in range(1,session.shape[1]-1):
            axs1[i_session].plot(session[:, var_n], label=f'{features_names[var_n]}')
            #print('session shape',session.shape[0])
            #print('session shape/100', )
            #for i in range(1,int(session.shape[0]/100)):
            #    print((i-1)*100,i*100)
            #    t = np.arange((i-1)*100,i*100)
            #    axs1[i_session].plot(t,session[(i-1)*100:i*100,var_n],label=f'{features_names[var_n]}')

        # Add torque on seperate axis
        torques = session[:,-1]
        color = 'tab:red'
        ax2 = axs1[i_session].twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Torque', color=color)
        ax2.plot(torques, label="Reward Value", c=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')
        #ax2.set_ylim((ax2.get_ylim()[0], ax2.get_ylim()[1] * y_buffer))

        axs1[i_session].set_title(data_names[i_session])
        axs1[i_session].set_ylim(y_range)
        leg = axs1[i_session].legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                                ncol=1, borderaxespad=0)
        legends1.append(leg)
        fig1.subplots_adjust(right=0.8)


        print('\n')

    leg = InteractiveLegend.create(legends1)

##########################################################################
# SINGLE GAITPHASE #######################################################
def gaitphase_features(PLOT_ANALYSIS=False,LW=0.2):
    #global HS_indxs
    #global i_fig

    session_data = []
    DataHandler.import_KneeAngle = True
    features_names = ['GP (%)', 'Knee (deg)','Hip (deg)', 'Shank (deg)','Torque (N.m)']
    for name in data_names:
        X, y = DataHandler.import_custom(name)
        new_data = np.append(X, y, axis=1)
        session_data.append(new_data)

    n_sessions = len(session_data)
    print(f'n_sessions {n_sessions} ')
    print(f'Session Data Shape {np.shape(session_data)}')
    statistics_cols = ['Feature']
    statistics_rows = ['mean_peak','mean_peak_time', 'mean_duration','mean_range[0]','mean_range[1]']
    statistics = np.array([statistics_rows]).reshape(-1,1)


    ##############################
    # Plotting Features ##########
    ##############################
    plot_cols = n_sessions
    plot_rows  = session_data[0].shape[1]
    if not(PLOT_ANALYSIS): plot_rows=plot_rows-1
    fig = plt.figure('Per Gaitphase')
    print(f'Plot row,col {plot_rows, plot_cols}')

    for i_session in range(n_sessions):
        session = session_data[i_session]
        print(f'Shape session {session.shape}')
        session_per_gp = DataHandler.data_per_gaitphase(session[:,:-1],session[:,-1])
        print(f'Shape session_per_gp {session_per_gp.shape},{session_per_gp[0].shape}')
        #session_per_gp = np.reshape(session_per_gp,(session_per_gp.shape[0], 100,5))
        #print(f'Shape session_per_gp {session_per_gp.shape}')

        # generate axes for each feature (excluding gait phase = x axis)
        n_features = np.shape(session_per_gp[0][1])[0]
        print(f'N features {n_features}')
        gp_axs = []
        for f in range(1,plot_rows+1):
            ax = plt.subplot(plot_rows, plot_cols, (f-1)*n_sessions+i_session+1)
            gp_axs.append(ax)

        #print(f'AXS Shape {np.shape(gp_axs)}')

        # add data to the  plots
        #print(f'SPGP {np.shape(session_per_gp)}')
        for gp in session_per_gp:
            gp_axs[0].set_title(f"All Features - "
                                f"{data_names[i_session].split('_')[1]}ing (strides={int(np.shape(session)[0]/100)})\n "
                                f"Patient {data_names[i_session].split('P')[-1].split('S')[0]} "
                                f"Session {data_names[i_session].split('S')[-1]}\n\n")
            for f in range(1,plot_rows+1):

                gp_axs[f-1].plot(gp[:,f],color='k',linewidth=LW)
                gp_axs[f-1].set_ylabel(features_names[f])


                for i in range(len(subphases['durations'])):
                    sp = subphases['durations'][i]
                    color = subphases['colors'][i]
                    label = subphases['labels'][i]
                    y_scale = 1.4
                    gp_axs[f-1].fill_between(sp, [0], np.max(session[:,f]*y_scale),
                                            facecolor=color, alpha=0.3,)

                    if f == 1:
                        gp_axs[f-1].annotate(label, xy=(np.mean(sp), np.max(session[:,f])*y_scale*0.95),  #xycoords='data',
                                            horizontalalignment='center', verticalalignment='bottom',
                                             color='grey'
                                            )

                    try:
                        gp_axs[f-1].set_ylim((0,np.max(session[:,f])*y_scale*0.95))
                        gp_axs[f-1].set_xlim((0, 100))
                        if f< plot_rows:gp_axs[f-1].set_xticks([])
                        else: gp_axs[f-1].set_xlabel('Gait Phase (%)')
                    except:
                        print(f'Err Session Sample: {session[:,f].shape}')
                        print(f)


            #plt.subplots_adjust(hspace=0.5)
    ##############################
    # Torque Analysis ############
    ##############################

        # calculate metrics for therapist interaction (torque)
        significant_torque = 0.5
        torque_stats = feature_statistics(session_per_gp[:,:,-1],significant_torque = significant_torque)
        mean_range=torque_stats["mean_range"]
        mean_peak = torque_stats["mean_peak"]
        mean_peak_time = torque_stats["mean_peak_time"]
        mean_duration = torque_stats["mean_duration"]
        new_statistics = np.array([[mean_peak],
                                   [mean_peak_time],
                                   [mean_duration],
                                   [mean_range[0]],
                                   [mean_range[1]]]).round(2)


        statistics = np.append(statistics,new_statistics,axis=1)
        statistics_cols.append(data_names[i_session])
        if PLOT_ANALYSIS:
            # Plot statistics
            line_width = 3
            gp_axs[-1].plot(torque_stats["mean_gp_torques"][:, 0], torque_stats["mean_gp_torques"][:, 1], c='k')
            gp_axs[-1].plot(np.linspace(mean_range[0], mean_range[1], 10),
                     np.ones(10) * mean_peak,
                     color='cornflowerblue', linestyle=':', linewidth=line_width,label="Meant Peak Torque")

            t = np.linspace(0, 100, 100)
            gp_axs[-1].fill_between(t, significant_torque, mean_peak, where=(t > mean_range[0]) & (t < mean_range[1]), alpha=0.5,
                             facecolor='c',label="Sig Torque x Mean Peak")
            gp_axs[-1].vlines(mean_peak_time, 0, mean_peak, colors='r',
                       linestyles=':', linewidth=line_width, label="Mean Peak Time")
            gp_axs[-1].legend(loc='upper left')
    ##############################
    # Display Results in Terminal#
    ##############################
    print(f'\n\nResults...\n')
    df = DataFrame(statistics,columns=statistics_cols)
    print(df)

##########################################################################
# SINGLE GAITPHASE #######################################################

def feature_statistics(torque_per_gp, significant_torque = 1,VERBOSE=False):
    if VERBOSE:
        print(f'feature_statistics...')
        print(f'|\t feature shape {np.shape(torque_per_gp)}')

    peaks = []
    peak_times = []
    actuation_times = []
    durations = []
    intercepts = []
    all_gp_torques = np.empty((0, 2), int)

    for torques in torque_per_gp:
        scale = 100 / torques.size
        # Peak Torque
        peak = max(torques)
        peaks.append(peak)

        # Peak Time
        peak_time = np.mean(np.where(torques == peak)[0]) * scale
        peak_times.append(peak_time)

        # Duration of significant actuation
        actuation_indxs = []
        for i in range(torques.size):
            if torques[i] > significant_torque: actuation_indxs.append(i)
        if len(actuation_indxs) == 0:
            actuation_indxs = [0, 0]
        actuation_time = [min(actuation_indxs) * scale, max(actuation_indxs) * scale]
        actuation_times.append(actuation_time)
        duration = actuation_time[1] - actuation_time[0]
        durations.append(duration)

        # Acutation for session
        gps = np.linspace(0, 100, len(torques)).reshape((-1, 1))
        intercepts.append(torques[0])
        torques = np.reshape(torques, (-1, 1))
        gp_torques = np.append(gps, torques, axis=1)
        all_gp_torques = np.append(all_gp_torques, gp_torques, axis=0)

    # Calculate mean statistics
    mean_peak = np.mean(peaks)
    mean_range = [np.mean([start[0] for start in actuation_times]),
                  np.mean([end[1] for end in actuation_times])]
    mean_duration = np.mean(durations)
    mean_peak_time = np.mean(peak_times)

    # Calculate mean function
    mean_gp_torques = mean_fun(all_gp_torques[:, 0], all_gp_torques[:, 1], intercept=np.mean(intercepts))

    # Package
    stats = {
        "mean_peak": mean_peak,
        "mean_range": mean_range,
        "mean_duration": mean_duration,
        "mean_peak_time": mean_peak_time,
        "mean_gp_torques": mean_gp_torques
    }
    if VERBOSE: print(stats)
    return stats

def mean_fun(X_train,y_train,POLY_ORDERS=[1,2,3,4,5,6,7,8,9,10],VERBOSE=False,intercept=None):
    best_scoring_order = 0
    best_order_score = 0

    X_train = np.reshape(X_train, (-1, 1)).astype('float32')
    #X_train = np.append(X_train, X_train+100, axis=0)

    if intercept!=None:
        y_train=y_train-intercept
        if VERBOSE: print(f'Intercept {intercept}')
    y_train = np.reshape(y_train, (-1, 1)).astype('float32')
    #y_train = np.append(y_train, y_train, axis=0)

    if VERBOSE: print(f'mean_fun(): x.shape {X_train.shape}, y.shape {y_train.shape}')
    for order in POLY_ORDERS:
        # Set polynomial order and fit data
        poly = PolynomialFeatures(degree=order)
        X_train_poly = poly.fit_transform(X_train)

        # Fit linear model
        reg = linear_model.LinearRegression(normalize=True,
            fit_intercept=False).fit(X_train_poly, y_train)
        score = reg.score(X_train_poly,y_train)
        if VERBOSE: print("Regression Score ( Order =", order, "):", score)

        # Save the regression if it is better than current best
        if score > best_order_score:
            best_order_score = score
            best_scoring_order = order
            best_reg = reg

    model_order = best_scoring_order
    X_obs = np.reshape(np.linspace(0, 100, 100), (-1, 1))
    poly = PolynomialFeatures(degree=model_order)
    X_obs_poly = poly.fit_transform(X_obs)
    y_pred = best_reg.predict(X_obs_poly)+intercept


    #gp_Poly = PolynomialFeatures(degree=best_scoring_order)
    #pred_features = reg.predict(np.array([[1]]))
    #print(f'Pred features {pred_features}')

    return np.append(np.reshape(X_obs,(-1,1)),np.reshape(y_pred,(-1,1)),axis=1)

if __name__ == '__main__':
    main()