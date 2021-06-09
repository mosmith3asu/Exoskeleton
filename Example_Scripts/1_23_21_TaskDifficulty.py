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
    #plt.figure("Algorithm Performance (Gait Cycle)")
    algorithm_names = ['OLS', 'GPR', 'IRL', "Therapist"]

    #stat_trend_names = ['T_peak', 't_peak']
    stats = []

    ols_stats =[ {'T_peak': [],
                 't_peak':[],
                  't_start':[],
                  't_end':[],
                  't_duration':[]} for i in range(n_datasets)]
    gpr_stats =[ {'T_peak': [],
                 't_peak':[],
                  't_start':[],
                  't_end':[],
                  't_duration':[]} for i in range(n_datasets)]
    irl_stats = [ {'T_peak': [],
                 't_peak':[],
                  't_start':[],
                  't_end':[],
                  't_duration':[]} for i in range(n_datasets)]
    ther_stats = [ {'T_peak': [],
                 't_peak':[],
                  't_start':[],
                  't_end':[],
                  't_duration':[]} for i in range(n_datasets)]
    plot_cols = n_datasets
    plot_rows = n_algorithms + 1

    lr_analysis = []
    gpr_analysis = []
    irl_analysis = []
    therapist_analysis = []

    T_sig = 1.2
    # Load Statistics
    verbose = False
    axs = []
    r = -1
    X_obs, y_obs = pred.get_obs()
    # print(f'AXS Shape {np.shape(gp_axs)}')


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
                #ols_stats[c]['T_peak'].append(phase.max())
                #ols_stats[c]['t_peak'].append(np.where(phase == phase.max())[0][0])
                #ols_stats[c]['t_sig'].append()
                S = analysis([phase],significant_torque=T_sig)
                ols_stats[c]['T_peak'].append(S['mean_peak'])
                ols_stats[c]['t_peak'].append(S['mean_peak_time'])
                ols_stats[c]['t_start'].append(S['mean_range[0]'])
                ols_stats[c]['t_end'].append(S['mean_range[1]'])
                ols_stats[c]['t_duration'].append(S['mean_duration'])
        if r == 1:
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.gpr_preds[c])
            for phase in y_pred_gp:
                #gpr_stats[c]['T_peak'].append(phase.max())
                #gpr_stats[c]['t_peak'].append(np.where(phase == phase.max())[0][0])
                #gpr_stats[c]['s2'].append(np.var(gpr_stats[c]['t_peak']))
                S = analysis([phase], significant_torque=T_sig)
                gpr_stats[c]['T_peak'].append(S['mean_peak'])
                gpr_stats[c]['t_peak'].append(S['mean_peak_time'])
                gpr_stats[c]['t_start'].append(S['mean_range[0]'])
                gpr_stats[c]['t_end'].append(S['mean_range[1]'])
                gpr_stats[c]['t_duration'].append(S['mean_duration'])
        if r == 2:
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.irl_preds_d[c])
            for phase in y_pred_gp:
                #irl_stats[c]['T_peak'].append(phase.max())
                #irl_stats[c]['t_peak'].append(np.where(phase == phase.max())[0][0])
                 # irl_stats[c]['s2'].append(np.var(irl_stats[c]['t_peak']))
                S = analysis([phase], significant_torque=T_sig)
                irl_stats[c]['T_peak'].append(S['mean_peak'])
                irl_stats[c]['t_peak'].append(S['mean_peak_time'])
                irl_stats[c]['t_start'].append(S['mean_range[0]'])
                irl_stats[c]['t_end'].append(S['mean_range[1]'])
                irl_stats[c]['t_duration'].append(S['mean_duration'])
        if r == 3:
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.lr_preds[c])
            for phase in y_obs_gp:
                #ther_stats[c]['T_peak'].append(phase.max())
                #ther_stats[c]['t_peak'].append(np.where(phase == phase.max())[0][0])
                #ther_stats[c]['s2'].append(np.var(phase))
                S = analysis([phase], significant_torque=T_sig)
                ther_stats[c]['T_peak'].append(S['mean_peak'])
                ther_stats[c]['t_peak'].append(S['mean_peak_time'])
                ther_stats[c]['t_start'].append(S['mean_range[0]'])
                ther_stats[c]['t_end'].append(S['mean_range[1]'])
                ther_stats[c]['t_duration'].append(S['mean_duration'])


    stats = [ols_stats,gpr_stats,irl_stats,ther_stats]
    stat_names = {'T_peak': 'Peak Torque (N.m)',
      't_peak': 'Peak Torque Time (% p_s)',
      't_start': 'Start of Significant \n Assistance (% p_s)',
      't_end': 'End of Significant \n Assistance (% p_s)',
      't_duration': 'Duration of Significant \n Assistance (% p_s)'}

    L = 1
    plot_rows = 3

    for kw in stats[0][0]:
        plt.figure(f'Statistic: {kw}')
        r=-1

        for i in range(1, plot_cols * plot_rows + 1):
            # Set up plots
            ax = plt.subplot(plot_rows, plot_cols, i)
            c = (i - 1) % plot_cols
            print(kw)

            label = stat_names[kw]
            if c == 0:
                r += 1
                if r==0:y_label = label
                elif r==1:y_label = label.split('(')[0] + f'Trend ({label.split("(")[1]}'
                elif r==2: y_label = label.split('(')[0] + f'Trend Slopes \n({label.split("(")[1][:-1]}/stride)'

            # Plot data
            if r == 0:
                title = f'Therapist {int(data_names[c].split("S")[-1]) - 1}: {kw} Adaptation'
                ax.plot(ols_stats[c][kw],label='OLS', linewidth=L)
                ax.plot(gpr_stats[c][kw], label='GPR', linewidth=L)
                ax.plot(irl_stats[c][kw], label='IRL', linewidth=L)
                ax.plot(ther_stats[c][kw], label='Ther',color='k',linestyle='--', linewidth=L)
                ax.set_ylim((0, ax.get_ylim()[1] * 1.1))
                ax.legend(loc='lower left')
                ax.set_xlabel('Stride')
            if r == 1:
                title = f'Therapist {int(data_names[c].split("S")[-1]) - 1}: {kw} Adaptation Trend'
                x = np.arange(0, np.size(ols_stats[c][kw]))
                ols_trend = np.poly1d(np.polyfit(x, ols_stats[c][kw], 1))
                gpr_trend = np.poly1d(np.polyfit(x, gpr_stats[c][kw], 1))
                irl_trend = np.poly1d(np.polyfit(x, irl_stats[c][kw], 1))
                ther_trend = np.poly1d(np.polyfit(x, ther_stats[c][kw], 1))

                ax.plot(x, ols_trend(x), label='OLS Trend', linewidth=L)
                ax.plot(x, gpr_trend(x), label='GPR Trend', linewidth=L)
                ax.plot(x, irl_trend(x), label='IRL Trend', linewidth=L)
                ax.plot(x, ther_trend(x), label='Therapist Trend', linewidth=L, color='k', linestyle='--')
                ax.set_ylim((0, ax.get_ylim()[1] * 1.1))
                ax.legend(loc='lower left')
                ax.set_xlabel('Stride')
                plt.subplots_adjust(wspace=0.2, hspace=0.2)

            if r == 2:
                title = f'Therapist {int(data_names[c].split("S")[-1]) - 1}: {kw} Adaptation Trend Slopes'
                x_labels= algorithm_names
                x = np.arange(0, np.size(ols_stats[c][kw]))
                ols_trend = np.poly1d(np.polyfit(x, ols_stats[c][kw], 1))
                gpr_trend = np.poly1d(np.polyfit(x, gpr_stats[c][kw], 1))
                irl_trend = np.poly1d(np.polyfit(x, irl_stats[c][kw], 1))
                ther_trend = np.poly1d(np.polyfit(x, ther_stats[c][kw], 1))

                ols_slope = ols_trend(x)[1] - ols_trend(x)[0]
                gpr_slope = gpr_trend(x)[1] - gpr_trend(x)[0]
                irl_slope = irl_trend(x)[1] - irl_trend(x)[0]
                ther_slope = ther_trend(x)[1] - ther_trend(x)[0]

                y = [ols_slope,gpr_slope,irl_slope,ther_slope]
                colors = []
                for trend in y[:-1]:
                    pos_trend = (trend>0)
                    pos_ther =(y[-1]>0)
                    if pos_ther==pos_trend:
                        colors.append('green')
                    else: colors.append('red')
                colors.append('green')

                add_var = False
                if add_var:
                    print(f'\n#### {kw} ####')
                    print(f'colors {colors}')
                    print(ols_stats[c][kw][1:])
                    print(ols_stats[c][kw][:-1])
                    print(np.array(ols_stats[c][kw][1:])-np.array(ols_stats[c][kw][:-1]))
                    print(np.var(np.array(ols_stats[c][kw][1:]) - np.array(ols_stats[c][kw][:-1])))
                    s2 = [np.var(np.array(ols_stats[c][kw][1:])-np.array(ols_stats[c][kw][:-1])),
                          np.var(np.array(gpr_stats[c][kw][1:])-np.array(gpr_stats[c][kw][:-1])),
                          np.var(np.array(irl_stats[c][kw][1:])-np.array(irl_stats[c][kw][:-1])),
                          np.var(np.array(ther_stats[c][kw][1:])-np.array(ther_stats[c][kw][:-1]))]
                    print(f'Slopes Y: {y}')
                    print(f'Algs X: {x_labels}')
                    print(f'Var s2: {s2}')

                    x_pos = [i for i, _ in enumerate(x_labels)]
                    rects1 = ax.bar(x_pos, y, color='green', yerr=s2,label='Trend Slope')
                else:

                    ax.bar(x_labels, np.zeros(np.shape(y)).tolist(), color='green', label='Correct Trends')
                    ax.bar(x_labels, np.zeros(np.shape(y)).tolist(), color='red', label='Incorrect Trends')
                    rects1 = ax.bar(x_labels, y, color=colors)#,label='Trend Slopes')
                #ax.plot(np.zeros(np.size(y)+3,)-1.5,linestyle = '--',color='k')
                ax.plot(np.zeros(np.size(y), ), linestyle='--', color='k')
                autolabel(rects1, ax)
                #ax.bar_label(rect, padding=3)

                buffer = 1.3
                lim = max([abs(yi) for yi in y])*buffer
                print(lim)
                #ax.set_ylabel()
                ax.set_ylim(-lim,lim)
                ax.legend(loc='upper left')
                ax.set_xlabel('Algorithm & Therapist Observation')
                plt.yticks([])
                plt.subplots_adjust(wspace=0.2, hspace=0.5)

            ax.set_title(f"{title}")
            ax.set_ylabel(y_label)
            axs.append(ax)





def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()


        if height <0: offset = -10
        else: offset=3

        if abs(height) < 0.5: offset *= 1.3

        ax.annotate('{}'.format(round(height,2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, offset),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

main()