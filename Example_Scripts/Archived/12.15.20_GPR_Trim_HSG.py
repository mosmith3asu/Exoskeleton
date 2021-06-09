import gc
from Modules.Plotting import Simple_Plot
from Modules.Learning import Regressors
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process.kernels import WhiteKernel,ConstantKernel
from datetime import datetime
from Modules.Utils.DataHandler import Data_Handler

##########################################################################
##########################################################################
# Data:             Trimmed training data
# Weighted Ratings: True
# Features:         Hip Angle, Shank Angle, Ground Reaction Forces
# Label:            Torque
##########################################################################
##########################################################################
time_begin = datetime.now() # current date and time

##########################################################################
# GLOBAL CONTROL PARAMETERS###############################################
# Kernel
kernel_num = 1
regression = Regressors.regression_methods()
kernals = regression.gpr_kernals()
kernel_settings = kernals[kernel_num]+ConstantKernel()+WhiteKernel()

# Enables
VERBOSE = True
SAVE_MODEL = True
WEIGHTED = False

# Data
data_names= [
    #'trimmed_train_P3S2'
    'trimmed_train_P3S3'
         ]
test_data_names= [
    #'trimmed_test_P3S2'
    'trimmed_test_P3S3'
         ]
max_points= 1000 # helps if less than half of all data
fix_gaitphase = 50 # Have to fix gaitphase to see other feature data
# Larger values indicate more noise in data point
# Rating: [0, 1, 2, 3]
# Rating: [no info, good, neutral, poor]
ALPHA_WEIGHTS = [1.e-4, 1.e-8, 1.e-4,1.e-1]
ALPHA_WEIGHT = 1.e-4

# Report
if VERBOSE: print(f"Running GPR:\n"
                  f"Kernel = {1}\n"
                  f"Saveing Model ={SAVE_MODEL}\n"
                  f"Max Training Points = {max_points} points\n")


##########################################################################
# IMPORT DATA ############################################################
DataHandler = Data_Handler()
DataHandler.return_enable=np.array([
            ['Rating'	,True], ['Gaitphase',False],
            ['Knee Angle',False],['Hip Angle',True],
            ['Shank Angle',True],	['Torque',True],
            ['GFR1',True],	['GRF2',True],
            ['GRF3',True],	['GRF4',True],
            ['HS Index',False],	['SW index',False]])

if VERBOSE: DataHandler.print_features()

if SAVE_MODEL:
    date = time_begin.strftime("%m_%d_%Y")                                 # get current date
    feature_names = 'Trim_PHS'
    data_name = DataHandler.append_names(data_names)
    MODEL_NAME = f"{date}_K{kernel_num}_{data_name}_{feature_names}"                # Construnct final model name
    if VERBOSE: print("\nSaving Model as: ",MODEL_NAME)

ratings_train,X_train,y_train = DataHandler.import_custom(data_names)
ratings_test,X_test,y_test = DataHandler.import_custom(test_data_names)

if VERBOSE:print("\nShape of raw ratings (Train|Test):",np.shape(ratings_train),'|',np.shape(ratings_test))
if VERBOSE:print("Shape of raw feature data (Train|Test):",np.shape(X_train),'|',np.shape(X_test))
if VERBOSE:print("Shape of raw label data (Train|Test):",np.shape(y_train),'|',np.shape(y_test))

##########################################################################
# RESAMPLE/SPLIT INTO TEST/TRAIN DATA ####################################
# Cap maximimum number of data points
ratings_train= DataHandler.resample_cap(ratings_train,max_points)
X_train= DataHandler.resample_cap(X_train,max_points)
y_train= DataHandler.resample_cap(y_train,max_points)

if VERBOSE:print("\nShape of resampled ratings (Train|Test):",np.shape(ratings_train),'|',np.shape(ratings_test))
if VERBOSE:print("Shape of resampled feature data (Train|Test):",np.shape(X_train),'|',np.shape(X_test))
if VERBOSE:print("Shape of resampled label data (Train|Test):",np.shape(y_train),'|',np.shape(y_test))


##########################################################################
# GAUSSIAN PROCESS REGRESSION ############################################
# Set up regressor class and Set up GPR inputs

# GPR Control Parameters
if WEIGHTED: alpha = np.reshape([ALPHA_WEIGHTS[int(rating)] for rating in ratings_train],(1,-1))
else: alpha = ALPHA_WEIGHT
kernel = kernel_settings
optimizer_iterations = 5

if VERBOSE: print("\nInitializing Regressor...")


gp = Pipeline([("scaler",StandardScaler()),
               ("GPR",GaussianProcessRegressor(kernel=kernel,alpha=alpha,copy_X_train=True,
                                               n_restarts_optimizer=optimizer_iterations,
                                               normalize_y=False,random_state=None))
          ])
if VERBOSE: print("Fitting Data...")
gp.fit(X_train, y_train)
score = round(gp.score(X_test, y_test),3)
print("Regression Score:",score ) # Score regression


##########################################################################
# PLOTING ################################################################
# Define Input Space/Domains
# ONLY HIP AND SHANK ANGLE PLOT AT SOME FIXED GAIT PHASE

x1 = np.linspace(X_train[:, 1].min(), X_train[:, 1].max())
x2 = np.linspace(X_train[:, 2].min(), X_train[:, 2].max())

# Generate X0p, X1p, normalized inputs and Zp outputs plotting
X0p,X1p = np.meshgrid(x1, x2)
Zp=np.zeros(X0p.shape)
for j in range(X0p.shape[1]):
    for i in range(X0p.shape[0]):
        x = np.array([[fix_gaitphase ,X0p[i, j], X1p[i, j]]])
        pred = gp.predict(x)
        Zp[i,j] = pred

Zp = np.array(Zp).T
reg_surf = (X0p, X1p, Zp)

# Plot
title = f'Features: {feature_names} |Fixed Gait Phase {fix_gaitphase} | GPR Score: {score}'

plt_components = Simple_Plot.surfplot3D(reg_surf, title)
scatter_data = np.append(X_train[:,1:],y_train,axis=1)
Simple_Plot.scatterplot3D(scatter_data, title, plt_components=plt_components)
plt, fig, ax = plt_components
ax.set_xlabel("Hip Displacement (degrees)")
ax.set_ylabel("Shank Displacment (degrees)")
ax.set_zlabel("Torque (N.m)")

##########################################################################
# EXPORT #################################################################
if SAVE_MODEL:
    DataHandler.save_model(gp,MODEL_NAME,plt)
    if VERBOSE: print("\n\nModel Saved...")

if VERBOSE: print("\nFINISHED...")
time_end = datetime.now()
time_elapsed = time_end-time_begin
if VERBOSE: print(f"\nElapse time (seconds): {time_elapsed.total_seconds()}")

gc.collect()
plt.show()