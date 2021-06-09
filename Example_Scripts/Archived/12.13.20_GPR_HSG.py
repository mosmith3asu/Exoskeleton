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
# Weighted Ratings: True
# Features:         Gaitphase, Hip Angle, Shank Angle
# Label:            Torque
##########################################################################
##########################################################################
time_begin = datetime.now() # current date and time

##########################################################################
# GLOBAL CONTROL PARAMETERS###############################################
kernel_num = 1
regression = Regressors.regression_methods()
kernals = regression.gpr_kernals()
kernel_settings = kernals[kernel_num]+ConstantKernel()+WhiteKernel()

VERBOSE = True
SAVE_MODEL = True
WEIGHTED = False

max_points= 400 # helps if less than half of all data
TEST_DATA_PERCENT = 0.25

fix_gaitphase = 50 # Have to fix gaitphase to see other feature data

# Larger values indicate more noise in data point
# Rating: [0, 1, 2, 3]
# Rating: [no info, good, neutral, poor]
ALPHA_WEIGHTS = [1.e-4, 1.e-8, 1.e-4,1.e-1]
ALPHA_WEIGHT = 1.e-4
if VERBOSE: print(f"Running GPR:\n"
                  f"Kernel = {1}\n"
                  f"Saveing Model ={SAVE_MODEL}\n"
                  f"Testing Data % = {TEST_DATA_PERCENT*100.}\n"
                  f"Max Points = {max_points} points\n")

data_names= [
    #'processed_P3S3L1',
    #'processed_P3S3L2',
    'processed_P3S3L3'
         ]

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

if VERBOSE: # Display training features
    features = []
    for label,enable in DataHandler.return_enable:
        if label != 'Rating' and label != 'Torque' and enable=='True':
            features.append(label+', ')
    print(f'Features ({np.size(features)}):\n', features)

if SAVE_MODEL:
    date = time_begin.strftime("%m_%d_%Y")                                 # get current date
    feature_names = 'PHS'
    data_name = DataHandler.append_names(data_names)
    MODEL_NAME = f"{date}_K{kernel_num}_{data_name}_{feature_names}"                # Construnct final model name
    if VERBOSE: print("\nSaving Model as: ",MODEL_NAME)

ratings,X,Y = DataHandler.import_custom(data_names)
if VERBOSE:print("\nShape of raw ratings:",np.shape(ratings))
if VERBOSE:print("Shape of raw feature data:",np.shape(X))
if VERBOSE:print("Shape of raw label data:",np.shape(Y))

##########################################################################
# RESAMPLE/SPLIT INTO TEST/TRAIN DATA ####################################
# Cap maximimum number of data points
ratings= DataHandler.resample_cap(ratings,max_points)
X= DataHandler.resample_cap(X,max_points)
Y= DataHandler.resample_cap(Y,max_points)


if VERBOSE:print("\nShape of resampled ratings:",np.shape(ratings))
if VERBOSE:print("Shape of resampled feature data:",np.shape(X))
if VERBOSE:print("Shape of resampled label data:",np.shape(Y))
#print(ratings)

# Splint into training and test set
X_train, X_test, y_train, y_test,ratings_train,ratings_test = \
    train_test_split(X,Y,ratings, test_size=TEST_DATA_PERCENT, random_state=42)

if VERBOSE:print("\nShape of training ratings:",np.shape(ratings_train))
if VERBOSE:print("Shape of training feature data:",np.shape(X_train))
if VERBOSE:print("Shape of training label data:",np.shape(y_train))

##########################################################################
# GAUSSIAN PROCESS REGRESSION ############################################
# Set up regressor class and Set up GPR inputs

# GPR Control Parameters
if WEIGHTED: alpha = np.reshape([ALPHA_WEIGHTS[int(rating)] for rating in ratings_train],(1,-1))
else: alpha = ALPHA_WEIGHT
kernel = kernel_settings
optimizer_iterations = 5

if VERBOSE: print("\nInitializing Regressor...")


gp = Pipeline([("scaler",StandardScaler()),("GPR",GaussianProcessRegressor(kernel=kernel,alpha=alpha,copy_X_train=True,
                                                                      n_restarts_optimizer=optimizer_iterations,
                                                                      normalize_y=False,random_state=None))
          ])
if VERBOSE: print("Fitting Data...")
gp.fit(X_train, y_train)
score = round(gp.score(X_test, y_test),3)
print("Regression Score:",score ) # Score regression

##########################################################################
# EXPORT #################################################################
if SAVE_MODEL:
    DataHandler.save_model(gp,MODEL_NAME)
    if VERBOSE: print("Model Saved")

if VERBOSE: print("\nFINISHED...")
time_end = datetime.now()
time_elapsed = time_end-time_begin
if VERBOSE: print(f"\nElapse time (seconds): {time_elapsed.total_seconds()}")

gc.collect()