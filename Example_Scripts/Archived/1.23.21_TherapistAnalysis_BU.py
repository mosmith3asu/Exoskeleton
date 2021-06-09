import numpy as np
from datetime import datetime
from Modules.Utils.DataHandler import Data_Handler
import matplotlib.pyplot as plt
from Modules.Plotting import InteractiveLegend
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
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
    #'trimmed_train_P3S2',
    'trimmed_train_P3S3',
    #'trimmed_test_P3S2',
    'trimmed_test_P3S3'
         ]
print(f'Data Summary:\n {data_names}')

##########################################################################
# IMPORT DATA ############################################################
DataHandler = Data_Handler()
#DataHandler.import_KneeAngle=False
#DataHandler.import_HipAngle=False
#DataHandler.import_ShankAngle=False
#DataHandler.import_GRF=False

session_data = []
X,y = DataHandler.import_custom(data_names)
data_per_gp = DataHandler.data_per_gaitphase(X,y)

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



var_param = [['Heel Strike',1,False],
             ['Knee Angle',1,True],
             ['Hip Angle',1,True],
             ['Shank Angle',1,True],
             ['GRF1',0.1,False],
             ['GRF2',0.1,False],
             ['GRF3',0.1,False],
             ['GRF4',0.1,False],
             ['Torque',20,True]]
if VERBOSE:print("\nShape of raw session data:",np.shape(session_data))

##########################################################################
# SESSION PREVIEW ########################################################
def session_features():
    global HS_idxs
    global i_fig
    fig1 = plt.figure(f'Figure {i_fig}: All Features per Session')
    fig1.title = 'Session Preview'
    # var_param = [[Name,Scale Output, Enable]]

    y_range = (0,60)
    axs1=[]
    legends1 = []
    HS_idxs = []
    print(np.shape(session_data))
    for i_session in range(len(session_data)):
        # Add subplot
        n = len(fig1.axes)
        for i in range(n):
            fig1.axes[i].change_geometry(n + 1, 1, i + 1)
        axs1.append(fig1.add_subplot(n + 1, 1, n + 1))

        # Plot with lines at heel strike indexs
        session = session_data[i_session]
        for var_n in range(session.shape[1]):
            if var_n == 0:
                phases = session[:, var_n]
                HS = []
                for p in range(len(phases)-1):
                    if phases[p+1]<phases[p]: HS.append(p+1)
                axs1[i_session].vlines(HS, min(y_range), max(y_range), colors='r',
                                linestyles=':', linewidth=1, label=var_param[var_n][0])
                HS_idxs.append(HS)
            else:
                axs1[i_session].plot(var_param[var_n][1]*session[:,var_n],label=f'{var_param[var_n][1]}x{var_param[var_n][0]}')

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
def gaitphase_features():
    global HS_indxs
    global i_fig
    gp_fig= []
    gp_axs =[]
    gp_legends = []
    torque_per_gp = []
    n_sessions = len(session_data)
    statistics_cols = ['Feature']
    statistics_rows = ['mean_peak','mean_peak_time', 'mean_duration','mean_range[0]','mean_range[1]']
    #statistics = np.empty((len(statistics_rows),0), int)
    statistics = np.array([statistics_rows]).reshape(-1,1)


    ##############################
    # Plotting Features ##########
    ##############################
    axs_labels = ["Knee","Hip","Shank","Torque"]

    for i_session in range(n_sessions):
        i_fig =i_fig +1
        gp_fig.append(plt.figure(f'Figure {i_fig}: Gait Phase Features: Data: {data_names[i_session]}'))

        session = session_data[i_session]
        session_per_gp = DataHandler.data_per_gaitphase(session[:,:-1],session[:,-1])

        # generate axes for each feature (excluding gait phase = x axis)
        n_features = np.shape(session_per_gp)[2]-1
        for f in range(n_features):
            n = len(gp_fig[i_session].axes)
            for i in range(n):
                gp_fig[i_session].axes[i].change_geometry(n + 1, 1, i + 1)
            axs = gp_fig[i_session].add_subplot(n + 1, 1, n + 1)
            gp_axs.append(axs)

        # add data to the  plots
        print(np.shape(session_per_gp[0]))
        for gp in session_per_gp:
            features = gp[:,1:]
            for f in range(np.shape(features)[1]):
                if f==0:gp_axs[f].set_title("All Features - Patient 3 Session 3")

                gp_axs[f].plot(features[:,f],color='k',linewidth=0.5)
                gp_axs[f].set_ylabel(axs_labels[f])

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
        gp_axs[-1].legend()
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

i_fig = 1
session_features()
gaitphase_features()

plt.show()