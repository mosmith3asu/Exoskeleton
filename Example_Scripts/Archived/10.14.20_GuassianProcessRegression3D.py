# Analyzing 2 Dimiensions = Knee Angle and Gait Phase

from Modules.Plotting import Simple_Plot
from Modules.Learning import Regressors
import numpy as np
from numpy import genfromtxt
from math import *

SAVE_MODEL = True
MODEL_NAME = "10_18_20_GPR"
def expected_knee_angle(gait_phase):
    expected_angle = 1e-08*pow(gait_phase,6) - 4e-06*pow(gait_phase,5) + 0.0003*pow(gait_phase,4) - 0.0112*pow(gait_phase,3) + 0.1127*pow(gait_phase,2) + 0.5405*gait_phase + 9.9107
    return expected_angle

data_paths = [
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\Patient3S2L1.csv",
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\Patient3S2L2.csv",
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\Patient3S2L3.csv",
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\Patient3S3L1.csv",
    "C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\Patient3S3L2.csv",
    "C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\Patient3S3L3.csv"
]

compare2healthy = False
# Import Data


regression_data = np.array([[]])
for path in data_paths:
    paired_data = genfromtxt(path, delimiter=',')
    data = paired_data[1:, 0:3]
    if np.size(regression_data)<1:
       regression_data = data
       print("init")
    else:
        regression_data=np.append(regression_data,data,axis=0)

    print(np.shape(regression_data))

print(np.shape(regression_data))


#data_S3L1 = paired_data[1:,0:3]
#data_S3L2 = paired_data[1:,0:3]
#data_S3L3 = paired_data[1:,0:3]
#regression_data = data_S3L1

# Set up regressor class and Set up GPR inputs
regression = Regressors.regression_methods()
kernals = regression.gpr_kernals()
alpha = 1e-5
k = kernals[1]
# Perform Regression and return variables and surface
reg_surf, gp, score = regression.gpr_surface(regression_data, kernel=k,
                                             alpha=alpha, plot_prior=True, debug=True)
r2,MSE = score

# Plot
title = f'Guassian Process Regression:' \
        f'\n Kernal={kernals[1]} ' \
        f'\nR^2 = {r2} MSE = {MSE}' \
        f'\n {len(regression_data)} datapoints'

plt_components = Simple_Plot.surfplot3D(reg_surf, title)
Simple_Plot.scatterplot3D(regression_data, title, plt_components=plt_components)
plt, fig, ax = plt_components
ax.set_xlabel("Gait Phase %")
ax.set_ylabel("Knee Displacment (degrees)")
ax.set_zlabel("Torque (N.m)")

if SAVE_MODEL:
    import pickle
    # save
    name = "GeneratedModels\\"+MODEL_NAME+".pkl"
    with open(name, 'wb') as f:
        pickle.dump(gp, f)



plt.show()