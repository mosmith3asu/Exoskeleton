# Analyzing 2 Dimiensions = Knee Angle and Gait Phase

from Modules.Plotting import Simple_Plot
import numpy as np
from numpy import genfromtxt
from math import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from datetime import datetime
time_begin = datetime.now() # current date and time
SAVE_MODEL=True
VERBOSE = True
ORDER = 8

def expected_knee_angle(gait_phase):
    expected_angle = 1e-08*pow(gait_phase,6) - 4e-06*pow(gait_phase,5) + 0.0003*pow(gait_phase,4) - 0.0112*pow(gait_phase,3) + 0.1127*pow(gait_phase,2) + 0.5405*gait_phase + 9.9107
    return expected_angle

data_paths = [
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S2L1.csv",
    #'C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S2L2.csv',
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S2L3.csv",
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S3L1.csv",
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S3L2.csv",
    "C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S3L3.csv"
]

if SAVE_MODEL:
    date = time_begin.strftime("%m_%d_%Y")                                 # get current date
    data_name=""
    for path in data_paths:
        data_name= data_name + path.split("\\")[-1].split(".")[0]   # get name of data used in model trainig
    MODEL_NAME = f"{date}_SearchOrder{ORDER}_{data_name}"                # Construnct final model name
    if VERBOSE: print("Saving Model as: ",MODEL_NAME)

compare2healthy = False
# Import Data
paired_data = genfromtxt(data_paths[0], delimiter=',')
data_S3L1 = paired_data[1:,0:3]
regression_data = data_S3L1

if compare2healthy:
    for i in range(len(regression_data)):
        gaitphse = regression_data[i,0]
        print("Old angle", regression_data[i, 1])
        regression_data[i,1] = expected_knee_angle(gaitphse)-regression_data[i,1]
        print("New angle",regression_data[i,1])

# Set up regressor class and Set up GPR inputs
def linregression_Surface(datapoints, search_up_to_order=1, surf_resolution=20):
    # Input: [nx3] dataset, polynomial order/degree, resolution of plotted surface
    # Output: x,y,z mesh used in ax.plot_surf(x,y,z) to describe regression surface
    # Dependant variable assumed to be in column 3 of datapoints
    # Full list of functions @ https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    print("Regression Scores: \n0=Patient_Data neglected (worst) \n1=Patient_Data perfectly fitted (best)\n")
    # Slice data into independant (x1,x2...) and dependant y variables
    X = datapoints[:, 0:len(datapoints[0]) - 1]
    Y = datapoints[:, -1]

    # Initialize variables to track best regression
    best_scoring_order = 0
    best_order_score = 0

    # Iterate through regressions of different orders and score
    for order in range(1, ORDER + 1):
        # Set polynomial order and fit data
        poly = PolynomialFeatures(degree=order)
        X_ = poly.fit_transform(X)

        # Fit linear model
        reg = linear_model.LinearRegression()
        reg.fit(X_, Y)

        # The test set, or plotting set
        x1_Length = int(np.max(datapoints[:, 0]))
        x2_Length = int(np.max(datapoints[:, 1]))
        predict_x0, predict_x1 = np.meshgrid(np.linspace(0, x1_Length, surf_resolution),
                                             np.linspace(0, x2_Length, surf_resolution))

        predict_x = np.concatenate((predict_x0.reshape(-1, 1),
                                    predict_x1.reshape(-1, 1)),
                                   axis=1)
        predict_x_ = poly.fit_transform(predict_x)
        predict_y = reg.predict(predict_x_)

        # Report intermediate scores
        score = reg.score(X_, Y)
        print("Regression Score ( Order =", order, "):", score)

        # Save the regression if it is better than current best
        if score > best_order_score:
            best_order_score = score
            best_scoring_order = order

            best_predict_x0 = predict_x0
            best_predict_x1 = predict_x1
            best_predict_y = predict_y.reshape(predict_x0.shape)
            surface = (best_predict_x0, best_predict_x1, best_predict_y)

            best_reg = reg

    # Report & Return the best regression score
    print("Best Regression Score @ Order=", best_scoring_order, " with score:", best_order_score)
    return surface, best_scoring_order, best_order_score, best_reg

# Plot
data_clipping = True
max_z = 40
min_z = 0
reg_surf, order, score, lin_reg = linregression_Surface(data_S3L1, search_up_to_order=8)
title = f'Linear Regression: {data_name} Score {round(score,3)}'
plt_components = Simple_Plot.surfplot3D(reg_surf, title)
#plt, fig, ax = plt_components
#ax.clear()
#plt_components=plt, fig, ax
Simple_Plot.scatterplot3D(regression_data, title, plt_components=plt_components)
plt, fig, ax = plt_components
ax.set_xlabel("Gait Phase %")
ax.set_ylabel("Knee Displacment (degrees)")
ax.set_zlabel("Torque (N.m)")
plt.show()

if SAVE_MODEL:
    import pickle
    # save
    name = "GeneratedModels\\Linreg\\"+MODEL_NAME
    with open(name+".pkl", 'wb') as f:
        pickle.dump(lin_reg, f)
    plt.savefig(name+".png")
    if VERBOSE: print("Model Saved")