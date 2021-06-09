import numpy as np
from datetime import datetime
from Modules.Utils.DataHandler import Data_Handler
import matplotlib.pyplot as plt
from Modules.Plotting import InteractiveLegend
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

def analysis(torque_per_gp, significant_torque = 1,title="Analysis",VERBOSE=False):
    if VERBOSE:
        print(f'\n{title}...')
        #print(f'|\t feature shape {np.shape(torque_per_gp)}')

    peaks = []
    peak_times = []
    actuation_times = []
    durations = []
    intercepts = []
    all_gp_torques = np.empty((0, 2), int)
    torques_array = np.empty((0,np.size(torque_per_gp[0])))

    for torques in torque_per_gp:
        scale = 100 / torques.size
        # Peak Torque
        peak = max(torques)
        peaks.append(peak)
        if np.size(torques)==100:
            torques_array =np.append(torques_array,np.array(torques).reshape(1,-1),axis=0)

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
        #print('SHAPE', np.shape(torques))


    # Calculate mean statistics
    mean_peak = np.mean(peaks)

    mean_peak_time = np.mean(peak_times)
    mean_range = [np.mean([start[0] for start in actuation_times]),
                  np.mean([end[1] for end in actuation_times])]
    mean_duration = np.mean(durations)


    # Calculate variance
    var_peak = np.var(peaks)
    var_total = np.var(torques_array)

    # Calculate mean function
    mean_gp_torques = mean_fun(all_gp_torques[:, 0], all_gp_torques[:, 1], intercept=np.mean(intercepts))

    # Package
    stats = {
        "mean_peak": mean_peak,
        "mean_range[0]": mean_range[0],
        "mean_range[1]": mean_range[1],
        "mean_duration": mean_duration,
        "mean_peak_time": mean_peak_time,
        'var_peak': var_peak,
        'var_total':var_total
    }
    if VERBOSE: print(stats)
    #stats["mean_gp_torques"]= mean_gp_torques
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



plt.show()