# Analyzing 2 Dimiensions = Knee Angle and Gait Phase

import numpy as np
from numpy import genfromtxt
from sklearn.ensemble import BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from Modules.Learning import Regressors
from Modules.Plotting import Simple_Plot

# GLOBAL CONTROL PARAMETERS###############################################
SAVE_MODEL = True
MODEL_NAME = "10_19_20_GPR_Bagged_K1P3S3L3"
VERBOSE = True
SCALED = True
n_samples = 2000 # samples per estimator
n_estimators = 2

data_paths = [
    # "C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\Patient3S2L1.csv",
    # "C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\Patient3S2L2.csv",
    # "C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\Patient3S2L3.csv",
    # "C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\Patient3S3L1.csv",
    # "C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\Patient3S3L2.csv",
    "C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\Patient3S3L3.csv"
]

# IMPORT DATA ##########################################################
regression_data = np.array([[]])
for path in data_paths:
    paired_data = genfromtxt(path, delimiter=',')
    data = paired_data[1:, 0:3]
    if np.size(regression_data) < 1:
        regression_data = data
        print("init")
    else:
        regression_data = np.append(regression_data, data, axis=0)

#regression_data =  regression_data[:1000,:]

if VERBOSE: print("Shape of training data:", np.shape(regression_data))

# GAUSSIAN PROCESS REGRESSION #############################################
# Set up regressor class and Set up GPR inputs
regression = Regressors.regression_methods()
kernals = regression.gpr_kernals()

# GPR Control Parameters
alpha = 1e-10
kernel = kernals[1]
optimizer_iterations = 6

if VERBOSE: print("Initializing Regressor...")

# Perform Regression and return variables and surface
X = regression_data[:, 0:len(regression_data[0]) - 1]  # Split into inputs
Y = regression_data[:, -1]  # Split into outputs
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=42)

GPR = GaussianProcessRegressor(kernel=kernel,
                               alpha=alpha,
                               copy_X_train=False,
                               n_restarts_optimizer=optimizer_iterations,
                               normalize_y=True
                               )

# Bagging
#n_estimators = int(np.size(X_train[:, 1]) / n_samples)+1

if VERBOSE: print(f'{n_estimators} estimators with {n_samples} samples each')
if VERBOSE: print("Beginning Regression...")

gp = Pipeline([('scaler', StandardScaler()),('bagging', BaggingRegressor(base_estimator=GPR, max_samples=n_samples,
                                                                         n_estimators=n_estimators))])
gp.fit(X_train, y_train)
print("Accuracy:", gp.score(X_test,y_test))

# Plot #################################################
# Define Input Space/Domains
x1 = np.linspace(X_train[:, 0].min(), X_train[:, 0].max())
x2 = np.linspace(X_train[:, 1].min(), X_train[:, 1].max())

# Generate X0p, X1p, normalized inputs and Zp outputs plotting
X0p, X1p = np.meshgrid(x1, x2)
Zp = [gp.predict([(X0p[i, j], X1p[i, j]) for i in range(X0p.shape[0])]) for j in range(X0p.shape[1])]

Zp = np.array(Zp).T
reg_surf = (X0p, X1p, Zp)

# Plot
title = f'Guassian Process Regression Bagged:'

plt_components = Simple_Plot.surfplot3D(reg_surf, title)
Simple_Plot.scatterplot3D(regression_data, title, plt_components=plt_components)
plt, fig, ax = plt_components
ax.set_xlabel("Gait Phase %")
ax.set_ylabel("Knee Displacment (degrees)")
ax.set_zlabel("Torque (N.m)")

if SAVE_MODEL:
    import pickle

    # save
    name = "GeneratedModels\\" + MODEL_NAME + ".pkl"
    with open(name, 'wb') as f:
        pickle.dump(gp, f)

if VERBOSE: print("FINISHED...")

plt.show()
