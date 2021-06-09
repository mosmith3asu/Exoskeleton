import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import pandas as pd
from Modules.Utils.Analysis_Utils import analysis
from Modules.Utils.PredictionUtil import ModelPrediction
import xlsxwriter

CUSTOM_NAME_MOD = ""
y_max = 7
data_names = [
    'trimmed_test_P3S2',
    'trimmed_test_P3S3'
]




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
    global all_stats

    # PREDIT WITH MODELS ####################################################
    # PRED Linear Regression
    ols_model_name = '03_22_2021_P3S2P3S3_Trim_PHS_O5.pkl'
    lr_stats = pred.linear_regression(ols_model_name)
    all_stats.append(lr_stats)

    # PRED Gaussian Process Regression
    model_paths = ['03_22_2021_K1_P3S3_Trim_PHS.pkl']
    gpr_stats = pred.bayesian_committee(model_paths)
    all_stats.append(gpr_stats)

    # PRED Inverse Reinforcement Learning
    # model_name_irl = '03_16_2021_IRL_P3S2P3S3_PHS_7sx20a_obj'
    model_name_irl = '03_17_2021_IRL_P3S2P3S3_PHS_10sx20a_obj'
    irl_stats = pred.irl(model_name_irl, filt_iter=10, filter_BWxL="L")
    all_stats.append(irl_stats)

    # Run subfunctions on preditions ##########################################
    session_outputs()
    performance_results = algorithm_performane()
    analyze_results(performance_results)

    if SAVE_RESULTS:
        save_tables('Tables')

    plt.show()


def session_outputs():
    global all_stats
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

        axs.append(ax)
    if SAVE_RESULTS: save_figures(plt, title='Predictions')


def algorithm_performane():
    plt.figure("Algorithm Performance (Gait Cycle)")
    algorithm_names = ['OLS Torque', 'GPR Torque', 'IRL Torque', "Therapist Torque"]

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
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.lr_preds[c])
            ax = pred.plot_per_gaitphase(ax, y_pred_gp)
            lr_analysis.append(analysis(y_pred_gp, title=f'OLS Analysis {data_names[c]}', VERBOSE=verbose))
        if r == 1:
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.gpr_preds[c])
            ax = pred.plot_per_gaitphase(ax, y_pred_gp)
            gpr_analysis.append(analysis(y_pred_gp, title=f'GPR Analysis {data_names[c]}', VERBOSE=verbose))
        if r == 2:
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.irl_preds_d[c])
            ax = pred.plot_per_gaitphase(ax, y_pred_gp)
            irl_analysis.append(analysis(y_pred_gp, title=f'IRL Analysis {data_names[c]}', VERBOSE=verbose))
        if r == 3:
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.lr_preds[c])
            ax = pred.plot_per_gaitphase(ax, y_obs_gp)
            therapist_analysis.append(analysis(y_obs_gp, title=f'Therapist Analysis {data_names[c]}', VERBOSE=verbose))

        axs.append(ax)
    if SAVE_RESULTS: save_figures(plt, title='TestData')
    return lr_analysis, gpr_analysis, irl_analysis, therapist_analysis

def analyze_results(performance_results):
    lr_analysis, gpr_analysis, irl_analysis, therapist_analysis = performance_results

    # Define table format
    col_labels = ['Metric', 'OLS', 'GPR', 'IRL', "Ther"]
    dcol_labels = ['Metric', 'dOLS', 'dGPR', 'dIRL', 'Best']
    pecol_labels = ['Metric', '%errOLS', '%errGPR', '%errIRL', 'Best']
    # row_labels = ['Peak Torque', 'Peak Torque Time', 'Actuation Duration', 'Begin Actuation', 'End Actuation','Varience']

    # Reformat analysis collected in last step
    all_analysis = []
    for n in range(len(data_names)):
        all_analysis.append([lr_analysis[n], gpr_analysis[n], irl_analysis[n], therapist_analysis[n]])

    for n in range(len(data_names)):
        name = data_names[n]

        this_analysis = all_analysis[n]

        if True:
            row_labels = []
            ther_stats = np.empty((0, 1), dtype=float)
            for kw in this_analysis[-1]:
                row_labels.append(kw)
                value = this_analysis[-1][kw]
                ther_stats = np.append(ther_stats, np.array([[value]]), axis=0)
            ther_stats = ther_stats.round(2)

        df_array = np.array([row_labels]).reshape(-1, 1)
        ddf_array = np.array([row_labels]).reshape(-1, 1)
        pedf_array = np.array([row_labels]).reshape(-1, 1)

        for anal in this_analysis[:-1]:
            new_statistics = np.empty((0, 1), dtype=float)
            for kw in anal:
                value = anal[kw]
                new_statistics = np.append(new_statistics, np.array([[value]]), axis=0)
            new_statistics = new_statistics.round(2)

            dnew_statistics = (ther_stats - new_statistics).round(2)

            penew_statistics = np.empty((0, 1))
            for s in range(ther_stats.size):
                perc_err = round(100 * (ther_stats[s, 0] - new_statistics[s, 0]) / (ther_stats[s, 0]), 2)
                penew_statistics = np.append(penew_statistics, np.array([[perc_err]]), axis=0)

            df_array = np.append(df_array, new_statistics, axis=1)
            ddf_array = np.append(ddf_array, dnew_statistics, axis=1)
            pedf_array = np.append(pedf_array, penew_statistics, axis=1)

        df_array = np.append(df_array, ther_stats, axis=1)

        winners = []
        for r in range(ddf_array.shape[0]):
            row = np.abs(ddf_array[r, 1:].astype(float))
            winner_idx = np.argmin(row)
            winners.append(col_labels[winner_idx + 1])
        winners = np.array(winners).reshape(-1, 1)
        ddf_array = np.append(ddf_array, winners, axis=1)
        pedf_array = np.append(pedf_array, winners, axis=1)

        df = DataFrame(df_array, columns=col_labels)
        ddf = DataFrame(ddf_array, columns=dcol_labels)
        pedf = DataFrame(pedf_array, columns=pecol_labels)

        print('\n#############################')
        print(f'DATA: {name}\n')
        print(f'Raw Stats...\n{df}')
        print(f'\nDifference... \n{ddf}')
        print(f'\nPercent Error.. \n{pedf}')
        print('#############################\n\n\n\n')
        row_labels = ['mean_peak', 'mean_range[0]','mean_range[1]','mean_duration','mean_peak_time','var_peak','var_total']

        name = name.split('_')[-1]
        append_tables(tables=[df, ddf, pedf],
                      table_titles=[f'Raw Statatistics:{name}', f"Difference:{name}", f"Percent Error:{name}"],
                      col_labels=[col_labels, dcol_labels, pecol_labels],
                      row_labels=[row_labels,row_labels,row_labels])

def compare():
    col_labels = ['Algorithm', 'OLS', 'GPR', 'IRL','Cumulative Best']
    header = np.array([['Therapist 1','Therapist 1','Therapist 2','Therapist 2','',''],
                       ['# Best','# Worst','# Best','# Worst','Total Best','Total Worst']])
    footer = []


def save_figures(plot, title):
    bbox = 'tight'
    n = plot.get_fignums()[-1]
    plot.figure(n).set_size_inches(12, 10, forward=True)
    plot.savefig(save_dir + f'{date}_{title}.png', bbox_inches=bbox)

    print(f'Saving Plot...')
    print(f'|\t Name: {title}')
    print(f'|\t FIG NUM: {n}')
    print("\n")

def append_tables(tables, table_titles,
                  col_labels,row_labels,
                  notes = []):
    global all_tables
    global tables_per_session

    print('Appending Tables...')
    if len(notes) < 1:
        notes = ['' for i in range(len(table_titles))]


    for t in range(len(table_titles)):
        data_name = table_titles[t].split(':')[-1]
        #if not(data_name in tables_per_session.keys()):
        #    tables_per_session[data_name] = np.empty((0,np.shape(tables[t])[1]))

        table_title = np.append(np.array([['TABLE #:\n '+table_titles[t]]]), np.full((1, tables[t].shape[1] - 1), ""), axis=1)
        table = np.append(table_title,np.array([col_labels[t]]), axis=0)
        table = np.append(table, tables[t], axis=0)
        note = np.append(np.array([['Notes:'+notes[t]]]), np.full((1, tables[t].shape[1] - 1), ""), axis=1)
        table = np.append(table, note, axis=0)
        all_tables.append(table)
        print(f'|\t Adding table: {np.shape(table)}')


def save_tables(name,border_width=1):
    print(f'Saving Tables...')
    print(f'|\n N_tables: {len(all_tables)}')
    workbook = xlsxwriter.Workbook(save_dir + f'{date}_{name}.xlsx')

    header_format = workbook.add_format({
        'bottom': border_width, 'top': 0, 'left': 0, 'right': 0,
        'align': 'center',
        'valign': 'vcenter'})
    content_format = workbook.add_format(
        {'bottom': 0, 'top': 0, 'left': 0, 'right': 0,
         'align': 'center',
         'valign': 'vcenter'
         })
    notes_format = workbook.add_format(
        {'bottom': 1, 'top': 1, 'left': 0, 'right': 0,
         'align': 'left',
         'valign': 'vcenter'
         })


    row_offset = 0
    col = 0
    worksheet = workbook.add_worksheet(data_name)
    for table in all_tables:

        for row in range(table .shape[0]):
            if row <2:
                worksheet.write_row(row+row_offset, col, table[row,:], header_format)
            elif row == table .shape[0]-1:
                worksheet.write_row(row+row_offset, col, table[row, :], notes_format)
            else:
                worksheet.write_row(row+row_offset, col, table[row, :],content_format)
        worksheet.merge_range(0 + row_offset, 0, 0, table.shape[1]-1, table[0,0], header_format) # merge title
        #worksheet.merge_range(table.shape[0], 0, 0, table.shape[1] - 1, table[-1, 0], notes_format)  # merge notes

    workbook.close()

if __name__ == "__main__":
    main()
