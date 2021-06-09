# Analyzing 2 Dimiensions = Knee Angle and Gait Phase
import gc
from Modules.Plotting import Simple_Plot
from Modules.Learning import Regressors
import numpy as np
from numpy import genfromtxt,savetxt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from matplotlib.pyplot import savefig
from sklearn.gaussian_process.kernels import WhiteKernel,ConstantKernel
from datetime import datetime
time_begin = datetime.now() # current date and time

# GLOBAL CONTROL PARAMETERS###############################################
kernel_num = 1
regression = Regressors.regression_methods()
kernals = regression.gpr_kernals()
kernel_settings = kernals[kernel_num]+WhiteKernel()+ConstantKernel()

VERBOSE = True
SAVE_MODEL = True
TEST_DATA_PERCENT = 0.1
sample_every_nth = 4

if VERBOSE: print(f"Running GPR:\n"
                  f"Kernel = {1}\n"
                  f"Saveing Model ={SAVE_MODEL}\n"
                  f"Testing Data % = {TEST_DATA_PERCENT*100.}\n"
                  f"Resample Every = {sample_every_nth} points\n")

data_paths = [
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S2L1.csv",
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S2L2.csv",
    "C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S2L3.csv",
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S3L1.csv",
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S3L2.csv",
    #"C:\\Users\\mason\\Desktop\\Patient_files\\PairedData\\P3S3L3.csv"
]

if SAVE_MODEL:
    date = time_begin.strftime("%m_%d_%Y")                                 # get current date
    data_name=""
    for path in data_paths:
        data_name= data_name + path.split("\\")[-1].split(".")[0]   # get name of data used in model trainig
    MODEL_NAME = f"{date}_K{kernel_num}_{data_name}"                # Construnct final model name
    if VERBOSE: print("Saving Model as: ",MODEL_NAME)

# IMPORT DATA ##########################################################
regression_data = np.array([[]])
for path in data_paths:
    paired_data = genfromtxt(path, delimiter=',')
    data = paired_data[1:, 0:3]
    if np.size(regression_data)<1:
       regression_data = data
    else:
        regression_data=np.append(regression_data,data,axis=0)

if VERBOSE:print("Shape of raw data:",np.shape(regression_data))

# RESAMPLE/SPLIT INTO TEST/TRAIN DATA #############################################
# Reduce frequency of data by eliminating ever Nth sample

regression_data = regression_data[::sample_every_nth,:]
if VERBOSE:print("Shape of resampled data:",np.shape(regression_data))

# Split into input/outputs
X= regression_data[:, 0:len(regression_data[0]) - 1]     # Split into inputs
Y = regression_data[:, -1]                     # Split into outputs

# Splint into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=TEST_DATA_PERCENT, random_state=42)
if VERBOSE:print("Shape of training data:",np.shape(X_train))

# GAUSSIAN PROCESS REGRESSION #############################################
# Set up regressor class and Set up GPR inputs


# GPR Control Parameters
alpha = 1e-4
kernel = kernel_settings
optimizer_iterations = 5

if VERBOSE: print("Initializing Regressor...")

#X_train = preprocessing.scale(X)    # Normalize Training Data
# if SCALED:
#     scaler = StandardScaler().fit(X_train)
#     X_train = scaler.transform(X_train)

# gp = GaussianProcessRegressor(kernel=kernel,
#                                       alpha=alpha,
#                                       copy_X_train=True,
#                                       n_restarts_optimizer=optimizer_iterations,
#                                       normalize_y=False,
#                                       random_state=None
#                                       )
#gp.set_params(scaler)

gp = Pipeline([("scaler",StandardScaler()),("GPR",GaussianProcessRegressor(kernel=kernel,alpha=alpha,copy_X_train=True,
                                                                      n_restarts_optimizer=optimizer_iterations,
                                                                      normalize_y=False,random_state=None))
          ])
if VERBOSE: print("Fitting Data...")
gp.fit(X_train, y_train)
score = round(gp.score(X_test, y_test),3)
print("Regression Score:",score ) # Score regression


# Plot #################################################
# Define Input Space/Domains
x1 = np.linspace(X_train[:, 0].min(), X_train[:, 0].max())
x2 = np.linspace(X_train[:, 1].min(), X_train[:, 1].max())

# Generate X0p, X1p, normalized inputs and Zp outputs plotting
X0p, X1p = np.meshgrid(x1, x2)
Zp = [gp.predict([(X0p[i, j], X1p[i, j]) for i in range(X0p.shape[0])]) for j in range(X0p.shape[1])]

# Scale back up X1 and X1 from normalization
# if SCALED:
#     X_test = np.append(x1.reshape(-1,1),x2.reshape(-1,1),axis=1)
#     X_test = scaler.inverse_transform(X_test)
#     X0p, X1p = np.meshgrid(X_test[:,0].reshape(1,-1), X_test[:,1].reshape(1,-1))

Zp = np.array(Zp).T
reg_surf = (X0p, X1p, Zp)

# Print best regression if input specifies
#if debug or print_best: print("GPR with: \tr2=", r2, "\tMSE= ", MSE, "\tAlpha= ", alpha,
#                              "\nkernel = ", kernel)
# Plot
title = f'Guassian Process Regression: Score: {score}'

plt_components = Simple_Plot.surfplot3D(reg_surf, title)
Simple_Plot.scatterplot3D(regression_data, title, plt_components=plt_components)
plt, fig, ax = plt_components
ax.set_xlabel("Gait Phase %")
ax.set_ylabel("Knee Displacment (degrees)")
ax.set_zlabel("Torque (N.m)")

if SAVE_MODEL:
    import pickle
    # save
    name = "GeneratedModels\\"+MODEL_NAME
    with open(name+".pkl", 'wb') as f:
        pickle.dump(gp, f)
    plt.savefig(name+".png")
    if VERBOSE: print("Model Saved")

if VERBOSE: print("\nFINISHED...")
time_end = datetime.now()
time_elapsed = time_end-time_begin
if VERBOSE: print(f"\nElapse time (seconds): {time_elapsed.total_seconds()}")

gc.collect()
plt.show()