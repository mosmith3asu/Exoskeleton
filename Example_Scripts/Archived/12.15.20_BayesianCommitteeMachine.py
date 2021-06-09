# https://arxiv.org/pdf/1806.00720.pdf
import pickle
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from math import sqrt
from Modules.Utils import Data_Filters
from Modules.Utils.DataHandler import Data_Handler
VERBOSE = True
FILTERED = True
PLOT_TITLE = "GPR Model Trained on Session 2 Lap 1 \n" \
             "Tested on: Lap 2 and 3"

model_dir = '/Example_Scripts/GeneratedModels/GPR\\'

data_names= [
    #'processed_P3S3L1',
    #'processed_P3S3L2',
    #'processed_P3S3L3',
    #'trimmed_train_P3S2',
    #'trimmed_train_P3S3',
    'trimmed_test_P3S2',
    'trimmed_test_P3S3'
         ]

commitee_paths = [
    #'12_16_2020_K1_P3S3L1_PHS.pkl',
    #'12_16_2020_K1_P3S3L2_PHS.pkl',
    #'12_16_2020_K1_P3S3L3_PHS.pkl',
    '02_19_2021_K1_P3S3_Trim_PHS.pkl',
    #'12_16_2020_K1_P3S2_Trim_PHS.pkl',
    #'12_16_2020_K1_P3S3_Trim_PHS.pkl'
]


DataHandler = Data_Handler()
DataHandler.import_KneeAngle=False
DataHandler.import_GRF=False
DataHandler.print_features()
train_name= ""
for model_path in commitee_paths:
    train_name =train_name +"-"+ model_path.split("_")[-2].split(".")[0]


# Beysien Commitee machine ###############################################################
def commitee_predict(test_data, models, has_labels=False, return_std=False):
    print("Commitee Prediction...")
    num_test = np.size(test_data[:, 0])

    M = len(models)  # Number of experts
    print("Num Experts:", M)

    if has_labels:
        y_obs = test_data[:, -1]
        X_obs = test_data[:, 0:2]
    else:
        X_obs = test_data

    # Make predictions with individual models
    mu_experts = []
    s2_experts = []
    print(f'NaN in X_obs {np.isnan(X_obs).any()}')
    print(X_obs[0,:])
    for expert in models:
        mu, s = expert.predict(X_obs, return_std=True)
        s2 = s * s
        mu_experts.append(np.reshape(mu,(-1,1)))
        s2_experts.append(np.reshape(s2,(-1,1)))

    # Initialize aggrogated mu and s2
    mu = np.zeros((num_test, 1))
    s2 = np.zeros((num_test, 1))

    # Begin aggregation
    kss = 1  # weight of expert at X
    for i in range(M):
        s2 = s2 + 1. / s2_experts[i]
    s2 = 1.0 / (s2 + (1 - M) / kss)

    for i in range(M):
        mu = mu + s2 * (mu_experts[i] / s2_experts[i])


    if return_std:
        s = [sqrt(variance) for variance in s2]
        mu = mu.flatten()
        return mu, s
    else:
        mu = mu.flatten()
        return mu


# MAIN ##################################################
errors = np.array([])
# load in all of the models
commitee = []
for model_path in commitee_paths:
    with open(model_dir+model_path, 'rb') as f:
        gp = pickle.load(f)
        commitee.append(gp)

fig, axs = plt.subplots(len(data_names))
plt_num = 0

for name in data_names:
    # Import Data ########################
    print('Test Data: ',name)
    X_obs,y_obs = DataHandler.import_custom(name)
    if VERBOSE: print("Shape of feature data:", np.shape(X_obs))
    if VERBOSE: print("Shape of label data :", np.shape(y_obs))

    # Predictions ########################
    y_pred, sigma = commitee_predict(X_obs,commitee,return_std=True)
    if FILTERED:
        n_iter = 10
        y_pred = Data_Filters.lfilter(y_pred, n_iter)
    axs[plt_num].plot(y_obs, linestyle=":")
    axs[plt_num].plot(y_pred)
    current_data_name = name.split("\\")[-1].split(".")[0]

    # R2 Calculations #####################
    score = r2_score(y_obs,y_pred)
    print(f"R2: {score}")

    # MSE Calculations ####################
    mse = np.mean(abs(np.reshape(y_pred,(1,-1))-np.reshape(y_obs,(1,-1))))#np.mean(abs(y_pred-y_obs))
    print('MSE: ',mse,'\n')
    errors=np.append(errors, abs(np.reshape(y_pred,(1,-1))-np.reshape(y_obs,(1,-1))))

    # Plotting #############################
    axs[plt_num].set_title(f"Model Trained on: {train_name} Compared Against: {current_data_name} (MSE: {round(mse,2)} R2: {round(score,2)})")
    y_upperbound = [y_pred[i] + 1.9600 * sigma[i] for i in range(len(y_pred))]
    y_lowerbound = [y_pred[i] - 1.9600 * sigma[i] for i in range(len(y_pred))]
    x = np.arange(0, len(y_pred))
    axs[plt_num].fill_between(x, y_upperbound, y_lowerbound, alpha=0.5, interpolate=True, facecolor="grey")
    legend_list = [current_data_name, "Predicted Output", "95% Confidense Interval"]
    axs[plt_num].legend(legend_list,loc='upper left',fontsize='x-small')

    plt_num = plt_num + 1

print('Total MSE: ', np.mean(errors))
plt.subplots_adjust(right = 0.9,bottom = 0.1,top = 0.9,wspace=0.2,hspace =0.5 )
plt.show()
