from Modules.Utils.PredictionUtil import ModelPrediction
from Modules.Utils.Analysis_Utils import analysis

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from Modules.Utils.DataHandler import Data_Handler
from datetime import datetime
import os

time_begin = datetime.now() # current date and time
DataHandler = Data_Handler() # Import Data handler

SAVE_RESULTS = True
if SAVE_RESULTS:
    date = time_begin.strftime("%m_%d_%Y")  # get current date
    base = 'Results'
    NAME = f"{date}_{base}"  # Construnct final model name
    print("Saving Results... \n|\t as:", NAME)
    save_dir = "C:\\Users\\mason\\Desktop\\Thesis\\ML_MasonSmithGit\\Example_Scripts\\GeneratedModels\\Results\\"+NAME+"\\"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'|\t Created diretory {save_dir}\n')


def main():
    ##########################################################################
    # TEST DATA ##############################################################
    ##########################################################################
    y_max = 7
    data_names= [
        'trimmed_test_P3S2',
        'trimmed_test_P3S3'
             ]

    pred = ModelPrediction(data_names)
    all_stats = []


    # PRED Linear Regression #################################################
    ols_model_name = '03_22_2021_P3S2P3S3_Trim_PHS_O5.pkl'
    lr_stats = pred.linear_regression(ols_model_name)
    all_stats.append(lr_stats)

    # PRED Gaussian Process Regression #######################################
    model_paths = ['03_22_2021_K1_P3S3_Trim_PHS.pkl']
    gpr_stats = pred.bayesian_committee(model_paths)
    all_stats.append(gpr_stats)

    # PRED Inverse Reinforcement Learning ####################################
    #model_name_irl = '03_16_2021_IRL_P3S2P3S3_PHS_7sx20a_obj'
    model_name_irl = '03_17_2021_IRL_P3S2P3S3_PHS_10sx20a_obj'
    irl_stats = pred.irl(model_name_irl,filt_iter=10,filter_BWxL="L")
    all_stats.append(irl_stats)

    # PLOTTING SESSION #######################################################
    plt.figure("Algorithm Performance (Session)")
    algorithm_names = ['OLS Torque', 'GPR Torque', 'IRL Torque']
    n_algorithms = 3
    n_datasets = len(data_names)
    plot_cols = n_datasets
    plot_rows = n_algorithms
    axs = []

    r = -1
    for i in range(1, plot_cols * plot_rows + 1):
        # Set up plots
        ax = plt.subplot(plot_rows, plot_cols, i)
        c = (i - 1) % plot_cols
        if c == 0:
            r += 1
        ax.set_ylabel(algorithm_names[r])
        if r == 0:
            ax.set_title(data_names[c].split("_")[-1])

        # Plot data
        if r == 0:
            ax = pred.plot_lr(ax, all_stats[r][c])
        if r == 1:
            ax = pred.plot_gpr(ax, all_stats[r][c])
        if r == 2:
            ax = pred.plot_irl(ax, all_stats[r][c])
        ax.set_ylim((0, y_max))
        axs.append(ax)
    if SAVE_RESULTS: save_figures(plt,title='Preditions')

    ##########################################################################
    # PLOTTING GAIT CYCLE ####################################################
    ##########################################################################
    plt.figure("Algorithm Performance (Gait Cycle)")
    algorithm_names = ['OLS Torque', 'GPR Torque', 'IRL Torque',"Therapist Torque"]

    plot_cols = n_datasets
    plot_rows = n_algorithms+1

    lr_analysis = []
    gpr_analysis = []
    irl_analysis = []
    therapist_analysis = []

    verbose = False
    axs = []
    r = -1
    X_obs,y_obs = pred.get_obs()
    print(np.shape(y_obs))
    for i in range(1, plot_cols * plot_rows + 1):
        # Set up plots
        ax = plt.subplot(plot_rows, plot_cols, i)
        c = (i - 1) % plot_cols
        if c == 0:
            r += 1
        ax.set_ylabel(algorithm_names[r])
        if r == 0:
            ax.set_title(data_names[c].split("_")[-1])

        # Plot data
        if r == 0:
            y_obs_gp,y_pred_gp = pred.per_gaitphase(X_obs[c],y_obs[c],pred.lr_preds[c])
            ax = pred.plot_per_gaitphase(ax,y_pred_gp)
            lr_analysis.append(analysis(y_pred_gp,title=f'OLS Analysis {data_names[c]}',VERBOSE=verbose))
        if r == 1:
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.gpr_preds[c])
            ax = pred.plot_per_gaitphase(ax, y_pred_gp)
            gpr_analysis.append(analysis(y_pred_gp,title=f'GPR Analysis {data_names[c]}',VERBOSE=verbose))
        if r == 2:
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.irl_preds_d[c])
            ax = pred.plot_per_gaitphase(ax, y_pred_gp)
            irl_analysis.append(analysis(y_pred_gp,title=f'IRL Analysis {data_names[c]}',VERBOSE=verbose))
        if r == 3:
            y_obs_gp,y_pred_gp = pred.per_gaitphase(X_obs[c],y_obs[c],pred.lr_preds[c])
            ax = pred.plot_per_gaitphase(ax, y_obs_gp)
            therapist_analysis.append(analysis(y_obs_gp, title=f'Therapist Analysis {data_names[c]}', VERBOSE=verbose))

        ax.set_ylim((0,y_max))
        axs.append(ax)
    if SAVE_RESULTS: save_figures(plt,title='TestData')


    ##########################################################################
    # ANALYSIS ###############################################################
    ##########################################################################
    # Define table format
    col_labels = ['','OLS','GPR','IRL',"Ther"]
    dcol_labels = ['', 'dOLS', 'dGPR', 'dIRL','Best']
    pecol_labels = ['', '%errOLS', '%errGPR', '%errIRL', 'Best']
    #row_labels = ['Peak Torque', 'Peak Torque Time', 'Actuation Duration', 'Begin Actuation', 'End Actuation','Varience']


    # Reformat analysis collected in last step
    all_analysis = []
    for n in range(len(data_names)):
        all_analysis.append([lr_analysis[n], gpr_analysis[n], irl_analysis[n], therapist_analysis[n]])


    for n in range(len(data_names)):
        name=data_names[n]

        this_analysis = all_analysis[n]

        if True:
            row_labels=[]
            ther_stats = np.empty((0, 1), dtype=float)
            for kw in this_analysis[-1]:
                row_labels.append(kw)
                value = this_analysis[-1][kw]
                ther_stats=np.append(ther_stats,np.array([[value]]),axis=0)
            ther_stats=ther_stats.round(2)

        df_array = np.array([row_labels]).reshape(-1, 1)
        ddf_array = np.array([row_labels]).reshape(-1, 1)
        pedf_array = np.array([row_labels]).reshape(-1, 1)

        for anal in this_analysis[:-1]:
            new_statistics =np.empty((0, 1), dtype=float)
            for kw in anal:
                value = anal[kw]
                new_statistics=np.append(new_statistics,np.array([[value]]),axis=0)
            new_statistics=new_statistics.round(2)

            dnew_statistics = (ther_stats-new_statistics).round(2)


            penew_statistics = np.empty((0,1))
            for s in range (ther_stats.size):
                perc_err = round(100*(ther_stats[s,0]-new_statistics[s,0])/(ther_stats[s,0]),2)
                penew_statistics = np.append(penew_statistics,np.array([[perc_err]]),axis=0)

            df_array = np.append(df_array, new_statistics, axis=1)
            ddf_array = np.append(ddf_array, dnew_statistics, axis=1)
            pedf_array = np.append(pedf_array, penew_statistics, axis=1)

        df_array = np.append(df_array, ther_stats, axis=1)

        winners = []
        for r in range(ddf_array.shape[0]):
            row = np.abs(ddf_array[r,1:].astype(float))
            winner_idx = np.argmin(row)
            winners.append(col_labels[winner_idx+1])
        winners=np.array(winners).reshape(-1,1)
        ddf_array = np.append(ddf_array, winners, axis=1)
        pedf_array = np.append(pedf_array, winners, axis=1)



        df = DataFrame(df_array, columns=col_labels)
        ddf = DataFrame(ddf_array, columns=dcol_labels)
        pedf = DataFrame(pedf_array, columns=pecol_labels)



        #print(df_all)


        print('\n#############################')
        print(f'DATA: {name}\n')

        print(f'Raw Stats...')
        print(df)
        print(f'\nDifference...')
        print(ddf)
        print(f'\nPercent Error..')
        print(pedf)
        print('#############################\n\n\n\n')
        save_tables(tables = [df,ddf,pedf],
                    table_titles =["Raw Stats","Differene", "Percent Error"],
                    table_headers = [col_labels,dcol_labels,pecol_labels])


    plt.show()
def save_figures(plot,title):
    print(f'Saving Plot...')
    print(f'|\t Name: {title}')
    bbox = 'tight'
    n = plot.get_fignums()[-1]
    print(f'|\t FIG NUM: {n}')
    plot.figure(n).set_size_inches(12,10,forward=True)
    #@plot.figure()
    plot.savefig(save_dir + f'{date}_{title}.png',bbox_inches=bbox)
    print("\n")


def save_tables(tables,table_titles,table_headers):
    file_header = np.append(np.array([[f'Results for {date}']]), np.full((1, tables[0].shape[1] - 1), ""), axis=1)
    row_space = np.full((1, tables[0].shape[1]), "")
    df_all = np.append(file_header, row_space, axis=0)
    for t in range(len(table_titles)):
        table_title = np.append(np.array([[table_titles[t]]]), np.full((1, tables[t].shape[1] - 1), ""), axis=1)
        df_all = np.append(df_all, table_title, axis=0)
        df_all = np.append(df_all, np.array([table_headers[t]]), axis=0)
        df_all = np.append(df_all, tables[t], axis=0)
        df_all = np.append(df_all, row_space, axis=0)

    np.savetxt(save_dir + f'{date}_tables.csv', df_all, fmt='%s', delimiter=",")


if __name__=="__main__":
    main()