import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from Modules.Utils.Analysis_Utils import analysis
from Modules.Utils.PredictionUtil import ModelPrediction
import xlsxwriter
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
from Modules.Learning.IRL.IRL_Tools import load_obj
from matplotlib import rcParams
import matplotlib
from matplotlib.animation import FuncAnimation, PillowWriter
#import PythonMagick
#matplotlib.use('Agg')
from Modules.Utils.DataHandler import Data_Handler
from Modules.Plotting import InteractiveLegend

from Modules.Utils.Data_Filters import lfilter
print(matplotlib.matplotlib_fname())
rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick\www\convert.exe'
#rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'

CUSTOM_NAME_MOD = ""
y_max = 7
data_names = [
    #'trimmed_test_V1P3S2',
    #'trimmed_train_P3S2',
    'trimmed_test_P3S2',
    'trimmed_test_P3S3'
]
train_names = [
    #'trimmed_test_V1P3S2',
    #'trimmed_train_P3S2',
    'trimmed_train_P3S2',
    'trimmed_train_P3S3'
]
color_scheme = {'OLS': 'green',
                'GPR': 'purple',
                'IRL': 'orange'}

pred = ModelPrediction(data_names)
pred.y_lim = (0,7)
all_stats = []
n_algorithms = 3
n_datasets = len(data_names)
SAVE_RESULTS = True
if SAVE_RESULTS:
    time_begin = datetime.now()  # current date and time
    date = time_begin.strftime("%m_%d_%Y")  # get current date
    base = 'Results'
    NAME = f"{date}_{base}{CUSTOM_NAME_MOD}"  # Construnct final model name
    print("Saving Results... \n|\t as:", NAME)
    save_dir = "C:\\Users\\mason\\Desktop\\Thesis\\ML_MasonSmithGit\\Example_Scripts\\GeneratedModels\\Results\\" + NAME + "\\"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'|\t Created diretory {save_dir}\n')

all_tables= []
tables_per_session = {}
rows_between_tables = 12

def main():
    # ols_model_name = '03_24_2021_P3S2P3S3_Trim_PHS_O5.pkl'
    ols_model_name = '03_22_2021_P3S2P3S3_Trim_PHS_O5.pkl'
    # ols_model_name ='03_25_2021_P3S2P3S3_Trim_PHS_O4.pkl'
    lr_stats = pred.linear_regression(ols_model_name)
    all_stats.append(lr_stats)

    # PRED Gaussian Process Regression
    gpr_model_paths = ['03_25_2021_K1_V1P3S2P3S3_Trim_PHS.pkl']
    gpr_stats = pred.bayesian_committee(gpr_model_paths)
    all_stats.append(gpr_stats)

    # PRED Inverse Reinforcement Learning
    model_name_irl = '03_25_2021_IRL_V1P3S2P3S3_PHS_10sx20a_obj'
    irl_stats = pred.irl(model_name_irl, filt_iter=10, filter_BWxL="L")
    all_stats.append(irl_stats)

    algorithm_TaskDifficultyAdaption()
    plt.show()



def algorithm_TaskDifficultyAdaption():
    plt.figure("Algorithm Performance (Gait Cycle)")
    algorithm_names = ['OLS Torque', 'GPR Torque', 'IRL Torque', "Therapist Torque"]

    #stat_trend_names = ['T_peak', 't_peak']
    stats = [{'T_peak': [],
            't_peak': []} for i in range(n_datasets)]

    ols_stats =[ {'T_peak': [],
                 't_peak':[]} for i in range(n_datasets)]
    gpr_stats =[ {'T_peak': [],
                 't_peak':[]} for i in range(n_datasets)]
    irl_stats = [{'T_peak': [],
                  't_peak': []} for i in range(n_datasets)]
    ther_stats = [{'T_peak': [],
                  't_peak': []} for i in range(n_datasets)]
    plot_cols = n_datasets
    plot_rows = n_algorithms + 1

    lr_analysis = []
    gpr_analysis = []
    irl_analysis = []
    therapist_analysis = []

    verbose = False
    axs = []
    r = -1
    X_obs, y_obs = pred.get_obs()
    # print(f'AXS Shape {np.shape(gp_axs)}')
    stat_names = ['Peak Torque (N.m)', 'Peak Torque (N.m)']
    for i in range(1, plot_cols * plot_rows + 1):
        # Set up plots
        ax = plt.subplot(plot_rows, plot_cols, i)
        c = (i - 1) % plot_cols


        if c == 0:
            r += 1
            ax.set_ylabel(algorithm_names[r])
        if r == 0:
            ax.set_title(data_names[c].split("_")[-1]+'\n')

        # Plot data
        if r == 0:
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.lr_preds[c])
            for phase in y_pred_gp:
                ols_stats[c]['T_peak'].append(phase.max())
                ols_stats[c]['t_peak'].append(np.where(phase == phase.max())[0][0])
        if r == 1:
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.gpr_preds[c])
            for phase in y_pred_gp:
                gpr_stats[c]['T_peak'].append(phase.max())
                gpr_stats[c]['t_peak'].append(np.where(phase == phase.max())[0][0])
        if r == 2:
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.irl_preds_d[c])
            for phase in y_pred_gp:
                irl_stats[c]['T_peak'].append(phase.max())
                irl_stats[c]['t_peak'].append(np.where(phase == phase.max())[0][0])
        if r == 3:
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.lr_preds[c])
            for phase in y_obs_gp:
                ther_stats[c]['T_peak'].append(phase.max())
                ther_stats[c]['t_peak'].append(np.where(phase == phase.max())[0][0])



    L=1

    plot_rows =len(stat_names)
    r=-1
    for i in range(1, plot_cols * plot_rows + 1):
        # Set up plots
        ax = plt.subplot(plot_rows, plot_cols, i)
        c = (i - 1) % plot_cols

        if c == 0:
            r += 1
            ax.set_ylabel(stat_names[r])
        if r == 0:
            ax.set_title("Adaptive Assistance (Peak Torque)\n" +data_names[c].split("_")[-1] + '\n')
        if r == plot_rows-1:
            ax.set_xlabel('Stride #')
        # Plot data

        if r == 0:
            ax.plot(ols_stats[c]['T_peak'],label='OLS', linewidth=L)
            ax.plot(gpr_stats[c]['T_peak'], label='GPR', linewidth=L)
            ax.plot(irl_stats[c]['T_peak'], label='IRL', linewidth=L)
            ax.plot(ther_stats[c]['T_peak'], label='Ther',color='k',linestyle='--', linewidth=L)
        if r == 1:
            x = np.arange(0, np.size(ols_stats[c]['T_peak']))
            ols_trend = np.poly1d(np.polyfit(x, ols_stats[c]['T_peak'], 1))
            gpr_trend = np.poly1d(np.polyfit(x, gpr_stats[c]['T_peak'], 1))
            irl_trend = np.poly1d(np.polyfit(x, irl_stats[c]['T_peak'], 1))
            ther_trend = np.poly1d(np.polyfit(x, ther_stats[c]['T_peak'], 1))

            ax.plot(x, ols_trend(x), label='OLS Trend', linewidth=L)
            ax.plot(x, gpr_trend(x), label='GPR Trend', linewidth=L)
            ax.plot(x, irl_trend(x), label='IRL Trend', linewidth=L)
            ax.plot(x, ther_trend(x), label='Therapist Trend', linewidth=L, color='k', linestyle='--')

            print(f'######################################\n T_peak{data_names[c].split("_")[-1]}')
            print(f'Slopes')
            slopes = []
            ther_slope = ther_trend(x)[1] - ther_trend(x)[0]
            slopes.append(ols_trend(x)[1] - ols_trend(x)[0])
            slopes.append(gpr_trend(x)[1] - gpr_trend(x)[0])
            slopes.append(irl_trend(x)[1] - irl_trend(x)[0])
            slopes.append(ther_slope)
            slopes = [round(s, 2) for s in slopes]
            print(slopes)

            slopes = []
            ther_slope = ther_trend(x)[1] - ther_trend(x)[0]
            slopes.append(ols_trend(x)[1] - ols_trend(x)[0] - ther_slope)
            slopes.append(gpr_trend(x)[1] - gpr_trend(x)[0] - ther_slope)
            slopes.append(irl_trend(x)[1] - irl_trend(x)[0] - ther_slope)
            slopes = [round(s, 2) for s in slopes]
            print(f'dSlopes')
            print(slopes)

        axs.append(ax)
        ax.legend(loc = 'lower left')
        ax.set_ylim((0, 8))
    plt.subplots_adjust(wspace=0.1,hspace=0.1)

    plt.figure("Adaptive Assistance (Peak Torque Time)")
    stat_names = ['Peak Torque Time (% Stance Phase)', 'Peak Torque Time (% Stance Phase)']
    r = -1
    kw = 't_peak'
    for i in range(1, plot_cols * plot_rows + 1):
        # Set up plots
        ax = plt.subplot(plot_rows, plot_cols, i)
        c = (i - 1) % plot_cols

        if c == 0:
            r += 1
            ax.set_ylabel(stat_names[r])
        if r == 0:
            ax.set_title("Adaptive Assistance (Peak Torque Time)\n" + data_names[c].split("_")[-1] + '\n')
        if r == plot_rows - 1:
            ax.set_xlabel('Stride #')
        if r == 0:
            ax.plot(ols_stats[c]['t_peak'], label='OLS', linewidth=L)
            ax.plot(gpr_stats[c]['t_peak'], label='GPR', linewidth=L)
            ax.plot(irl_stats[c]['t_peak'], label='IRL', linewidth=L)
            ax.plot(ther_stats[c]['t_peak'], label='Ther',color='k',linestyle='--', linewidth=L)
        if r ==1:
            x = np.arange(0, np.size(ols_stats[c]['t_peak']))
            ols_trend = np.poly1d(np.polyfit(x, ols_stats[c]['t_peak'], 1))
            gpr_trend = np.poly1d(np.polyfit(x, gpr_stats[c]['t_peak'], 1))
            irl_trend = np.poly1d(np.polyfit(x, irl_stats[c]['t_peak'], 1))
            ther_trend = np.poly1d(np.polyfit(x, ther_stats[c]['t_peak'], 1))

            ax.plot(x, ols_trend(x), label='OLS Trend', linewidth=L)
            ax.plot(x, gpr_trend(x), label='GPR Trend', linewidth=L)
            ax.plot(x, irl_trend(x), label='IRL Trend', linewidth=L)
            ax.plot(x, ther_trend(x), label='Therapist Trend', linewidth=L,color='k',linestyle='--')

            print(f'######################################\nt_peak{data_names[c].split("_")[-1]}')
            print(f'Slopes')
            slopes = []
            ther_slope = ther_trend(x)[1] - ther_trend(x)[0]
            slopes.append(ols_trend(x)[1] - ols_trend(x)[0])
            slopes.append(gpr_trend(x)[1] - gpr_trend(x)[0])
            slopes.append(irl_trend(x)[1] - irl_trend(x)[0])
            slopes.append(ther_slope)
            slopes = [round(s,2) for s in slopes]
            print(slopes)

            slopes = []
            ther_slope = ther_trend(x)[1] - ther_trend(x)[0]
            slopes.append(ols_trend(x)[1] - ols_trend(x)[0] - ther_slope)
            slopes.append(gpr_trend(x)[1] - gpr_trend(x)[0] - ther_slope)
            slopes.append(irl_trend(x)[1] - irl_trend(x)[0] - ther_slope)
            slopes = [round(s, 2) for s in slopes]
            print(f'dSlopes')
            print(slopes)

        ax.set_ylim((0, 80))
        axs.append(ax)
        ax.legend(loc='lower left')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return lr_analysis, gpr_analysis, irl_analysis, therapist_analysis


if False:
    if r == 0:
        z = (x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--")
        ax.plot(lfilter(ols_stats[c]['T_peak'], n_iter=I, denom=D), label='OLS', linewidth=L)
        ax.plot(lfilter(gpr_stats[c]['T_peak'], n_iter=I, denom=D), label='GPR', linewidth=L)
        ax.plot(lfilter(irl_stats[c]['T_peak'], n_iter=I, denom=D), label='IRL', linewidth=L)
        ax.plot(lfilter(ther_stats[c]['T_peak'], n_iter=I, denom=D), label='Ther', linewidth=L * 2, color='k',
                linestyle='--')
        if r == 1:
            ax.plot(lfilter(ols_stats[c]['t_peak'], n_iter=I), label='OLS', linewidth=L)
        ax.plot(lfilter(gpr_stats[c]['t_peak'], n_iter=I), label='GPR', linewidth=L)
        ax.plot(lfilter(irl_stats[c]['t_peak'], n_iter=I), label='IRL', linewidth=L)
        ax.plot(lfilter(ther_stats[c]['t_peak'], n_iter=I), label='Ther', linewidth=L * 2, color='k', linestyle='--')

main()