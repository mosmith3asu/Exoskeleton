# https://arxiv.org/pdf/1806.00720.pdf
import pickle
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from math import sqrt
from Modules.Utils import Data_Filters
from Modules.Utils.DataHandler import Data_Handler
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
VERBOSE = True
FILTERED = True
PLOT_TITLE = "GPR Model Trained on Session 2 Lap 1 \n" \
             "Tested on: Lap 2 and 3"

model_dir = 'C:\\Users\\mason\Desktop\\Thesis\\ML_MasonSmithGit\Example_Scripts\\GeneratedModels\\LinearRegression\\'

data_names= [
    'processed_P3S3L1',
    'processed_P3S3L2',
    'processed_P3S3L3',
    #'trimmed_train_P3S2',
    #'trimmed_train_P3S3',
    'trimmed_test_P3S2',
    'trimmed_test_P3S3'
         ]


#model_path = '12_20_2020_P3S2_Trim_PHS_O3.pkl'
model_path = '12_20_2020_P3S2_Trim_PK_O5.pkl'
#model_path =   '12_20_2020_P3S3_Trim_PHS_O2.pkl'
#model_path =  '12_20_2020_P3S3_Trim_PK_O4.pkl'



DataHandler = Data_Handler()
DataHandler.return_enable=np.array([
            ['Rating'	,True], ['Gaitphase',True],
            ['Knee Angle',True],['Hip Angle',False],
            ['Shank Angle',False],	['Torque',True],
            ['GFR1',False],	['GRF2',False],
            ['GRF3',False],	['GRF4',False],
            ['HS Index',False],	['SW index',False]])

train_name= ""

train_name =train_name +"-"+ model_path.split("_")[-4].split(".")[0]


# MAIN ##################################################

# load in all of the models

with open(model_dir+model_path, 'rb') as f:
    linreg = pickle.load(f)
    model_order = int(model_path.split("O")[-1].split(".")[0])

fig, axs = plt.subplots(len(data_names))
plt_num = 0

for name in data_names:
    print('Test Data: ',name)
    ratings_test, X_obs, y_obs = DataHandler.import_custom([name])
    if VERBOSE: print("Shape of feature data:", np.shape(X_obs))
    if VERBOSE: print("Shape of label data :", np.shape(y_obs))

    poly = PolynomialFeatures(degree=model_order)
    X_obs_poly = poly.fit_transform(X_obs)
    y_pred = linreg.predict(X_obs_poly)

    axs[plt_num].plot(y_obs, linestyle=":")
    axs[plt_num].plot(y_pred)

    current_data_name = name.split("\\")[-1].split(".")[0]

    score = r2_score(y_obs,y_pred)
    print(f"Score for {current_data_name}: {score}\n")

    axs[plt_num].set_title(f"Model Trained on: {train_name} Compared Against: {current_data_name} Score: {round(score,2)}")

    legend_list = [current_data_name, "Predicted Output", "95% Confidense Interval"]
    axs[plt_num].legend(legend_list,loc='upper left',fontsize='x-small')

    plt_num = plt_num + 1

plt.subplots_adjust(right = 0.9,bottom = 0.1,top = 0.9,wspace=0.2,hspace =0.5 )
plt.show()
