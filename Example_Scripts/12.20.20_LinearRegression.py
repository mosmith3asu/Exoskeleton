# Analyzing 2 Dimiensions = Knee Angle and Gait Phase

from Modules.Plotting import Simple_Plot
import numpy as np
from numpy import genfromtxt
from math import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from datetime import datetime
from Modules.Utils.DataHandler import Data_Handler
import matplotlib.pyplot as plt
from Modules.Utils.PredictionUtil import ModelPrediction


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
# Enables
VERBOSE = True
SAVE_MODEL = True
WEIGHTED = False
POLY_ORDERS = np.arange(1,8)
surf_resolution=20
TRIMMED = 'Trim_'
# Data
data_names= [
    'trimmed_train_P3S2',
    'trimmed_train_P3S3'
         ]
test_data_names= [
    'trimmed_test_P3S2',
    'trimmed_test_P3S3'
         ]
pred = ModelPrediction(test_data_names)

fix_gaitphase = 50 # Have to fix gaitphase to see other feature data
# Larger values indicate more noise in data point
# Rating: [0, 1, 2, 3]
# Rating: [no info, good, neutral, poor]
ALPHA_WEIGHTS = [1.e-4, 1.e-8, 1.e-4,1.e-1]
ALPHA_WEIGHT = 1.e-4

# Report
if VERBOSE: print(f"Running Linear Regression...\n"
                  f"Searching Poly Orders = {[POLY_ORDERS]}\n"
                  f"Saveing Model ={SAVE_MODEL}\n")


##########################################################################
# IMPORT DATA ############################################################
DataHandler = Data_Handler()
DataHandler.import_KneeAngle=False
#DataHandler.import_HipAngle=False
#DataHandler.import_ShankAngle=False
DataHandler.import_GRF=False


##########################################################################
# IMPORT DATA ############################################################
#X_train,y_train = DataHandler.import_custom('trimmed_train_P3S3')
X_train,y_train = DataHandler.import_custom(data_names)
#X_train_new,y_train_new = DataHandler.import_custom('trimmed_train_P3S2')
#X_train = np.append(X_train,X_train_new,axis=0)
#y_train = np.append(y_train,y_train_new,axis=0)
X_test2,y_test2 = DataHandler.import_custom('trimmed_test_P3S2')
X_test3,y_test3 = DataHandler.import_custom('trimmed_test_P3S3')
#DataHandler.verify_import_with_graph()
data=DataHandler.data_per_gaitphase(X_train,y_train)


if SAVE_MODEL:
    date = time_begin.strftime("%m_%d_%Y")  # get current date
    feature_names = TRIMMED+DataHandler.feature_accronym()#'PHS'
    data_name = DataHandler.append_names(data_names)
    MODEL_NAME = f"{date}_{data_name}_{feature_names}_O#"
    if VERBOSE: print("Saving Model as: ", MODEL_NAME)


##########################################################################
# LINEAR REGRESSION ######################################################
# Initialize variables to track best regression
best_scoring_order = 0
best_order_score = 0

##if VERBOSE:print("\nShape of ratings (Train|Test):",np.shape(ratings_train),'|',np.shape(ratings_test))
if VERBOSE:print("Shape of feature data (Train|Test):",np.shape(X_train),'|',np.shape(X_test2))
if VERBOSE:print("Shape of label data (Train|Test):",np.shape(y_train),'|',np.shape(y_test2),'\n')

fig1 = plt.figure(f'OLS Regression Orders')
fig1.title = 'Session Preview'
axs1=[]

plot_cols = len(data_names)
plot_rows = len(POLY_ORDERS)
plt.figure("OLS Regression (Session)")

for i in range(len(POLY_ORDERS)):
    order = POLY_ORDERS[i]
    # Add subplot
    #n = len(fig1.axes)
    #for i in range(n):
    #    fig1.axes[i].change_geometry(n + 1, 1, i + 1)
    #axs1.append(fig1.add_subplot(n + 1, 1, n + 1))
    plt_num=1+i*2

    # Set polynomial order and fit data
    poly = PolynomialFeatures(degree=order)
    X_train_poly = poly.fit_transform(X_train)

    # Fit linear model
    reg = linear_model.LinearRegression()
    reg.fit(X_train_poly, y_train)

    # Report intermediate scores SESSION 2
    ax = plt.subplot(plot_rows, plot_cols, plt_num)
    X_test_poly2 = poly.fit_transform(X_test2)
    score2 = reg.score(X_test_poly2, y_test2)
    y_pred = reg.predict(X_test_poly2)
    ax.plot(y_test2, label=f'Test')
    ax.plot(y_pred, label=f'O{order} S{round(score2,2)}')
    ax.legend()

    # Report intermediate scores SESSION 3
    ax = plt.subplot(plot_rows, plot_cols, plt_num+1)
    X_test_poly3 = poly.fit_transform(X_test3)
    score3 = reg.score(X_test_poly3 , y_test3)
    y_pred = reg.predict(X_test_poly3)
    ax.plot(y_test3,label=f'Test')
    ax.plot(y_pred,label=f'O{order} S{round(score3,2)}')
    ax.legend()
    score = (score2+score3)/2
    print("Regression Score ( Order =", order, "):", score)

    # Save the regression if it is better than current best
    if score > best_order_score:
        best_order_score = score
        best_scoring_order = order
        best_reg = reg
if VERBOSE: print("Best Regression Score @ Order=", best_scoring_order, " with score:", best_order_score,'\n')
plt.show()
# Report & Return the best regression score


##########################################################################
# EXPORT #################################################################

if SAVE_MODEL:
    import pickle
    # save
    MODEL_NAME = f"{date}_{data_name}_{feature_names}_O{best_scoring_order}"
    name = "GeneratedModels\\"+MODEL_NAME
    with open(name+".pkl", 'wb') as f:
        pickle.dump(best_reg, f)
    if VERBOSE: print("Model Saved: ",MODEL_NAME+".pkl")

