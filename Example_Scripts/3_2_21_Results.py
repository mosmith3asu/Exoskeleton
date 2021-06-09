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
DataHandler = Data_Handler()
features_names = DataHandler.print_features()
features_names.append('Torque')
feature_scales = [1, 1, 1, 1]
from Modules.Utils.typeAtools.formatted_outputs import printh

print(matplotlib.matplotlib_fname())
rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick\www\convert.exe'
#rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'

CUSTOM_NAME_MOD = ""
y_max = 7
data_names = [
    'trimmed_test_V1P3S2',
    #'trimmed_train_P3S2',
    #'trimmed_test_P3S2',
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
    global all_stats

    #session_features()
    #gaitphase_features(train_names, 'Training_GaitphaseFeatures')
    #gaitphase_features(data_names,'Testing_GaitphaseFeatures')

    # PREDIT WITH MODELS ####################################################
    # PRED Linear Regression

    #ols_model_name = '03_24_2021_P3S2P3S3_Trim_PHS_O5.pkl'
    ols_model_name = '03_22_2021_P3S2P3S3_Trim_PHS_O5.pkl'
    #ols_model_name ='03_25_2021_P3S2P3S3_Trim_PHS_O4.pkl'
    lr_stats = pred.linear_regression(ols_model_name)
    all_stats.append(lr_stats)

    # PRED Gaussian Process Regression
    #gpr_model_paths = ['03_22_2021_K1_P3S3_Trim_PHS.pkl']
    #gpr_model_paths = ['03_25_2021_K1_P3S3_Trim_PHS.pkl','03_25_2021_K1_P3S2_Trim_PHS.pkl']
    #gpr_model_paths = ['03_25_2021_K1_P3S2P3S3_Trim_PHS.pkl']
    gpr_model_paths = ['03_25_2021_K1_V1P3S2P3S3_Trim_PHS.pkl']
    gpr_stats = pred.bayesian_committee(gpr_model_paths)
    all_stats.append(gpr_stats)

    # PRED Inverse Reinforcement Learning
    # model_name_irl = '03_16_2021_IRL_P3S2P3S3_PHS_7sx20a_obj'
    #model_name_irl = '03_17_2021_IRL_P3S2P3S3_PHS_10sx20a_obj'
    #model_name_irl = '03_24_2021_IRL_P3S2P3S3_PHS_7sx20a_obj'
    model_name_irl = '03_25_2021_IRL_V1P3S2P3S3_PHS_10sx20a_obj'
    irl_stats = pred.irl(model_name_irl, filt_iter=10, filter_BWxL="L")
    all_stats.append(irl_stats)

    # Run subfunctions on preditions ##########################################
    #session_outputs()
    performance_results = algorithm_performane()
    #performance_results = algorithm_TaskDifficultyAdaption()
    analyze_results(performance_results)
    compare()

    IRL_Visualization(model_name_irl)

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
    # print(f'AXS Shape {np.shape(gp_axs)}')

    # add data to the  plots
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
            ax = pred.plot_per_gaitphase(ax, y_pred_gp,PLOT_SUBPHASES=True,SUBPHASE_HEADERS=True)
            lr_analysis.append(analysis(y_pred_gp, title=f'OLS Analysis {data_names[c]}', VERBOSE=verbose))
            ax.set_xticks([])

        if r == 1:
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.gpr_preds[c])
            ax = pred.plot_per_gaitphase(ax, y_pred_gp,PLOT_SUBPHASES=True)
            gpr_analysis.append(analysis(y_pred_gp, title=f'GPR Analysis {data_names[c]}', VERBOSE=verbose))
            ax.set_xticks([])

        if r == 2:
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.irl_preds_d[c])
            ax = pred.plot_per_gaitphase(ax, y_pred_gp,PLOT_SUBPHASES=True)
            irl_analysis.append(analysis(y_pred_gp, title=f'IRL Analysis {data_names[c]}', VERBOSE=verbose))
            ax.set_xticks([])
        if r == 3:
            y_obs_gp, y_pred_gp = pred.per_gaitphase(X_obs[c], y_obs[c], pred.lr_preds[c])
            ax = pred.plot_per_gaitphase(ax, y_obs_gp,PLOT_SUBPHASES=True)
            therapist_analysis.append(analysis(y_obs_gp, title=f'Therapist Analysis {data_names[c]}', VERBOSE=verbose))
            ax.set_xlabel('Stance Phase (%)')


        axs.append(ax)
    if SAVE_RESULTS: save_figures(plt, title='TestData')
    return lr_analysis, gpr_analysis, irl_analysis, therapist_analysis

def analyze_results(performance_results):
    global BW_dict
    lr_analysis, gpr_analysis, irl_analysis, therapist_analysis = performance_results
    BW_dict = {}
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
        BW_alg_tally = np.zeros((3,2))
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

        # CALCULATE BW TALLEYS
        therapist_num = f'T{int(name.split("S")[-1]) - 1}'
        if not (therapist_num in tables_per_session.keys()):
            BW_dict[therapist_num] = np.zeros((3, 2))

        for r in range(ddf_array.shape[0]):
            row = np.abs(ddf_array[r, 1:].astype(float))
            winner_idx = np.argmin(row)
            loser_idx = np.argmax(row)
            BW_dict[therapist_num][winner_idx, 0] += 1  # Best tally
            BW_dict[therapist_num][loser_idx, 1] += 1  # Worst tally

        # CALCULATE WINNERS TO DISPLAY
        winners = []
        for r in range(ddf_array.shape[0]):
            row = np.abs(ddf_array[r, 1:].astype(float))
            winner = col_labels[np.argmin(row) + 1]#np.argmin(row)
            winners.append(winner)
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
    print(f'Comparing...')
    title = 'Best and Worst Algorithm Tally'

    row_labels = ['OLS', 'GPR', 'IRL','aCumulative Best']
    col_labels = ['Algorithm','T1 # Best','T1 # Worst','T2 # Best','T2 # Worst','Total Best','Total Worst']
    notes = 'Notes: a Cumulative Best refers to either the most tallies for the “Best” category or the least number of tallies for the “Worst” category '
    tables = np.empty((3,0))
    for key in BW_dict:
        therapist_num = key
        table = BW_dict[therapist_num]
        tables = np.append(tables, table, axis=1)

    sum_best = (tables[:,0]+tables[:,2]).reshape(3,1)
    sum_worst = (tables[:,1]+tables[:,3]).reshape(3,1)
    tables = np.append(tables, sum_best,axis=1)
    tables = np.append(tables, sum_worst, axis=1)
    best_per_col=[]
    for c in range(tables.shape[1]):

        this_col = tables[:, c]
        if c%2==0:idxs = np.where(this_col == this_col.max())[0]
        else: idxs = np.where(this_col == this_col.min())[0]

        string = ''
        for i in idxs:
            string = string + row_labels[i] + "; "
        string = string[:-2]
        print(f'|\t col {this_col} idx {idxs} str {string}')
        best_per_col.append(string)

    best_per_col =np.array([best_per_col])
    tables = np.append(tables, best_per_col, axis=0)
    tables = np.append(np.reshape(row_labels,(-1,1)), tables, axis=1)
    tables = np.append(np.array([col_labels]), tables, axis=0)
    title_arr = np.append(np.array([['TABLE #:\n ' + title]]),
                            np.full((1, tables.shape[1] - 1), ""),
                            axis=1)
    tables = np.append(title_arr, tables, axis=0)
    notes_arr = np.append(np.array([[notes]]),np.full((1, tables.shape[1] - 1), ""),axis=1)
    tables = np.append(tables,notes_arr, axis=0)

    append_tables(complete_tables = [tables],table_titles=['BW Tally'])

    print(f'|\t Shape: {tables.shape}')
    print(f'|\t Shape: {tables}')

    # PLOT ###########################################################


    zero_height = 0.5e-1
    n_groups = (len(col_labels)-1)/2
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 1

    # Split best and worst tallys
    Data_Best,Data_Worst =[], []
    this_data = [float(item) for item in tables[2, 1:]]
    for i in range(len(this_data)):
        if i%2==0:Data_Best.append(this_data[i])
        else:Data_Worst.append(this_data[i])
    OLS_Data_Best = Data_Best
    OLS_Data_Worst = Data_Worst
    OLS_Data_Diff = Data_Best[-1]-Data_Worst[-1]
    if OLS_Data_Diff == 0: OLS_Data_Diff = zero_height


    Data_Best, Data_Worst = [], []
    this_data = [float(item) for item in tables[3, 1:]]
    for i in range(len(this_data)):
        if i % 2 == 0: Data_Best.append(this_data[i])
        else:Data_Worst.append(this_data[i])
    GPR_Data_Best = Data_Best
    GPR_Data_Worst = Data_Worst
    GPR_Data_Diff = Data_Best[-1]-Data_Worst[-1]
    if GPR_Data_Diff==0: GPR_Data_Diff =zero_height


    Data_Best, Data_Worst = [], []
    this_data = [float(item) for item in tables[4, 1:]]
    for i in range(len(this_data)):
        if i % 2 == 0:Data_Best.append(this_data[i])
        else:Data_Worst.append(this_data[i])
    IRL_Data_Best = Data_Best
    IRL_Data_Worst = Data_Worst
    IRL_Data_Diff = Data_Best[-1]-Data_Worst[-1]
    if IRL_Data_Diff == 0: IRL_Data_Diff = zero_height


    print(f'|\t N Groups: {n_groups} Size {np.shape(OLS_Data_Best)}')
    print(f'|\t OLS Data: {OLS_Data_Best}')
    print(f'|\t GPR Data: {GPR_Data_Best}')
    print(f'|\t IRL Data: {IRL_Data_Best}')

    # plt.figure("Best and Worst Performance Tallies")
    plt_rows = 2
    plt_cols = 3

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(plt_rows, plt_cols)


    # BEST TALLY
    #ax = plt.subplot(plt_rows,plt_cols, 1)
    ax = fig.add_subplot(gs[0, :-1])
    rects1 = ax.bar(index-bar_width, OLS_Data_Best, bar_width,
                     alpha=opacity,
                     #color=color_scheme['OLS'],
                     label='OLS')
    rects2 = ax.bar(index, GPR_Data_Best, bar_width,
                     alpha=opacity,
                     #color=color_scheme['GPR'],
                     label='GPR')
    rects3 = ax.bar(index+bar_width, IRL_Data_Best, bar_width,
                    alpha=opacity,
                    #color=color_scheme['IRL'],
                    label='IRL')
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    #plt.xlabel('')
    plt.ylabel('# Tallied')
    plt.title('Best Performance Tally')
    plt.xticks(index, ['Therapist 1 # Best','Therapist 2 # Best','Total Best'])
    plt.legend(loc='upper left')

    # WORST TALLY
    ax = fig.add_subplot(gs[1, :-1])
    #ax = plt.subplot(plt_rows,plt_cols, 2)
    rects1 = ax.bar(index - bar_width, OLS_Data_Worst, bar_width,
                    alpha=opacity,
                    #color=color_scheme['OLS'],
                    label='OLS')
    rects2 = ax.bar(index, GPR_Data_Worst, bar_width,
                    alpha=opacity,
                    #color=color_scheme['GPR'],
                    label='GPR')
    rects3 = ax.bar(index + bar_width, IRL_Data_Worst, bar_width,
                    alpha=opacity,
                    #color=color_scheme['IRL'],
                    label='IRL')
    autolabel(rects1,ax)
    autolabel(rects2,ax)
    autolabel(rects3,ax)
    # plt.xlabel('')
    plt.ylabel('# Tallied')
    plt.title('Worst Performance Tally')
    plt.xticks(index, ['Therapist 1 # Worst', 'Therapist 2 # Worst', 'Total Worst'])
    plt.legend(loc='upper left')

    # Difference TALLY
    ax = fig.add_subplot(gs[:, -1])
    # ax = plt.subplot(plt_rows,plt_cols, 2)
    rects1 = ax.bar(1 - bar_width, OLS_Data_Diff, bar_width,
                    alpha=opacity,
                    # color=color_scheme['OLS'],
                    label='OLS')
    rects2 = ax.bar(1, GPR_Data_Diff, bar_width,
                    alpha=opacity,
                    # color=color_scheme['GPR'],
                    label='GPR')
    rects3 = ax.bar(1 + bar_width, IRL_Data_Diff, bar_width,
                    alpha=opacity,
                    # color=color_scheme['IRL'],
                    label='IRL')
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)

    plt.ylabel('# Tallied')
    plt.title('Total Difference in Tallies (Best-Worst)')
    ax.set_xlim((-bar_width,bar_width))
    plt.xticks(index, [''])
    plt.xlabel('Total Difference')
    plt.legend(loc='upper left')


    if SAVE_RESULTS: save_figures(plt, title='BestWorstTallies')

def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height <0: offset = -3
        else: offset=3
        ax.annotate('{}'.format(int(height)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, offset),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def save_figures(plot, title,w=12,h=10):
    bbox = 'tight'
    n = plot.get_fignums()[-1]
    plot.figure(n).set_size_inches(w, h, forward=True)
    plot.savefig(save_dir + f'{date}_{title}.png', bbox_inches=bbox)

    print(f'Saving Plot...')
    print(f'|\t Name: {title}')
    print(f'|\t FIG NUM: {n}')
    print("\n")

def append_tables(tables = [], table_titles = [],
                  col_labels = [],row_labels = [],
                  notes = [],complete_tables = []):
    global all_tables
    global tables_per_session

    print('Appending Tables...')
    if len(notes) < 1:
        notes = ['' for i in range(len(table_titles))]

    if len(tables) < 1:
        for t in range(len(table_titles)):
            table = complete_tables[t]
            all_tables.append(table)
            row_space = np.full((1, np.shape(table)[1]), "")
            spacing_diff = rows_between_tables - np.shape(table)[0]

            for i in range(spacing_diff):
                table = np.append(table, row_space, axis=0)

            data_name = table_titles[t].split(':')[-1]
            if not (data_name in tables_per_session.keys()):
                tables_per_session[data_name] = [table]
            else:
                tables_per_session[data_name].append(table)
    else:



        for t in range(len(table_titles)):

            table_title = np.append(np.array([['TABLE #:\n '+table_titles[t]]]), np.full((1, tables[t].shape[1] - 1), ""), axis=1)
            table = np.append(table_title,np.array([col_labels[t]]), axis=0)
            table = np.append(table, tables[t], axis=0)
            note = np.append(np.array([['Notes:'+notes[t]]]), np.full((1, tables[t].shape[1] - 1), ""), axis=1)
            table = np.append(table, note, axis=0)

            all_tables.append(table)


            row_space = np.full((1, np.shape(tables[t])[1]), "")
            spacing_diff = rows_between_tables-np.shape(table)[0]

            for i in range(spacing_diff):
                table = np.append(table,row_space,axis=0)

            data_name = table_titles[t].split(':')[-1]
            if not (data_name in tables_per_session.keys()):
                tables_per_session[data_name] = [table]
            else:
                tables_per_session[data_name].append(table)


            print(f'|\t Adding table: {np.shape(table)}')


def save_tables(name,border_width=1):
    print(f'Saving Tables...')
    print(f'|\n N_tables: {len(all_tables)}')
    workbook = xlsxwriter.Workbook(save_dir + f'{date}_{name}.xlsx')

    col_label_width = 16
    title_height = 35
    header_format = workbook.add_format({
        'bottom': border_width, 'top': 0, 'left': 0, 'right': 0,
        'align': 'center', 'valign': 'vcenter',
        'font_name': 'Times New Roman','font_size':12})

    content_format = workbook.add_format(
        {'bottom': 0, 'top': 0, 'left': 0, 'right': 0,
         'align': 'center','valign': 'vcenter',
         'font_name': 'Times New Roman', 'font_size': 12
         })
    notes_format = workbook.add_format(
        {'bottom': 1, 'top': 1, 'left': 0, 'right': 0,
         'align': 'left','valign': 'vcenter',
         'font_name': 'Times New Roman', 'font_size': 12
         })

    for data_name in tables_per_session:
        print(f'|\t ')
        print(f'|\t Data name: {data_name}')
        worksheet = workbook.add_worksheet(data_name)
        tables = tables_per_session[data_name]
        table = np.reshape(tables,(-1,tables[0].shape[1]))
        col = 0

        for row in range(table.shape[0]):
            if row%rows_between_tables==0:
                print(f'|\t Merger Row {row}')
                top, left, bottom, right,text = row, 0, row, table.shape[1] - 1,table[row, 0]
                worksheet.merge_range(top, left, bottom, right, text, header_format)
                worksheet.set_row(row, title_height)
            elif row%rows_between_tables <2:
                worksheet.write_row(row , col, table[row, :], header_format)
            elif 'Notes:' in table[row,0]:
                worksheet.write_row(row, col, table[row, :], notes_format)
            else:
                worksheet.write_row(row, col, table[row, :], content_format)

        worksheet.set_column(0, col_label_width)
    workbook.close()
def save_gif(anim,name,fps=2):
    print(f'Saving Animation...')
    anim.save(save_dir + f'{date}_{name}.gif', writer='imagemagick', fps=fps)
def IRL_Visualization(IRL_name):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """
    global ani_fig
    global ani_ax
    global pax
    global maxax
    global IRL
    global reward_map
    global gp_names
    global ani_title
    global n_loops
    IRL = load_obj(IRL_name)
    IRL_reshape = (IRL.states_per_feature[0],IRL.states_per_feature[1],IRL.states_per_feature[2])
    reward_map = np.reshape(IRL.reward, (IRL_reshape))


    print(f'\n\n\nReward: {np.shape(IRL.reward)}')
    print(f'Reshaped Reward: {np.shape(np.reshape(IRL.reward,IRL_reshape))}')
    print(f'Policy: {np.shape(IRL.policy)}')
    IRL_2D_rewardmap()
    IRL_max_reward_plot()

    imagemagick_path = 'C:\Program Files\ImageMagick'
    n_loops = 5
    gp_names = []
    for p in range(IRL.N_STATES):
        gp_names.append(f'\n(GP=[{int(p * (100 / IRL.N_STATES))}%,{int((p + 1) * (100 / IRL.N_STATES))}%])')
    ani_fig, ani_ax = plt.subplots()

    pax = ani_ax.pcolor(reward_map[0, :, :])

    maxs = [np.where(reward_map[m, :, :] == reward_map[m, :, :].max()) for m in range(1)]
    maxs = np.reshape(maxs, (-1, 2))
    maxax = ani_ax.plot(maxs[:, 0], maxs[:, 1], color='r')

    ani_fig.colorbar(pax, ax=ani_ax)
    ani_title = ani_ax.text(0.5, 0.9, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                        transform=ani_ax.transAxes, ha="center")
    #pax = plt.pcolor(reward_map[0, :, :])
    IRL_animate_reward()

def IRL_2D_rewardmap(RESCALE_FEATURES=True):
    IRL_reshape = (IRL.states_per_feature[0], IRL.states_per_feature[1], IRL.states_per_feature[2])
    reward_map = np.reshape(IRL.reward, IRL_reshape)
    # 2d heatmap
    plot_cols = 5
    plot_rows = 2
    r = -1

    gp_names = []
    if IRL.USE_SUBPHASES:
        gp_names=['Initial Contact', 'Loading Response', 'Mid Stance', 'Terminal Stance', 'Pre-Swing']
    else:
        for p in range(IRL.N_STATES):
            gp_names.append(f'GP=[{int(p*(100/IRL.states_per_feature[0]))}%,{int((p+1)*(100/IRL.states_per_feature[0]))}%]')

    feature_names=["Hip Displacement (Deg)","Shank Displacement (Deg)"]
    fig = plt.figure('2D Reward Map')

    for i in range(1, IRL.states_per_feature[0]+1):
        c = (i - 1) % plot_cols
        # Set up plots
        #ax= fig.subplot(plot_rows, plot_cols, i)
        ax = plt.subplot(plot_rows, plot_cols, i)
        ax.set_title(gp_names[i-1])
        ax.set_ylabel(feature_names[0])
        ax.set_xlabel(feature_names[1])
        ax.pcolor(reward_map[i-1,:,:])

        if RESCALE_FEATURES:
            scale_x = 1/IRL.state_scales[2]
            scale_y = 1/IRL.state_scales[1]
            ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(round(x / scale_x,0)))
            ax.xaxis.set_major_formatter(ticks_x)
            ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(round(x / scale_y,0)))
            ax.yaxis.set_major_formatter(ticks_y)

    plt.suptitle('Recovered Reward Map (S=[GP,HD,SD])')
    plt.subplots_adjust(hspace=0.3,wspace=0.3)

    save_figures(plt, 'IRL_2D_RewardMaps', h=8, w=22)

def IRL_max_reward_plot():
    N_STATES = IRL.N_STATES
    IRL_reshape = (IRL.states_per_feature[0], IRL.states_per_feature[1], IRL.states_per_feature[2])
    reward_map = np.reshape(IRL.reward, IRL_reshape)
    # Maximum reward plot
    print("\n Max Reward vs Time")
    points=np.empty((0,4))
    for p in range(IRL.states_per_feature[0]):
        hs = np.argmax(reward_map[p])
        hidx = int(hs/N_STATES)
        sidx = hs%N_STATES
        r=np.max(reward_map[p])
        phs=np.array([[p,hidx,sidx,r]])
        points=np.append(points,phs,axis=0)

    print(points)
    y_buffer = 1.1

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title("Maximum State Reward vs Stance Phase")


    gp_names = []
    if IRL.USE_SUBPHASES:

        t = []
        for phase in IRL.subphases['durations']:
            t.append(phase[1])
        gp_names = [f'IC {t[0]}%', f'LR {t[1]}%', f'MS {t[2]}%', f'TS {t[3]}%',
                    f'PS {t[4]}%']
        plt.xticks(t, gp_names)
    else:
        gp_names=np.linspace(0,100,N_STATES).astype(int)
        plt.xticks(np.arange(N_STATES + 1), gp_names)
        t= points[:,0]

    # Joint plotting
    color = 'blue'
    ax.plot(t,points[:,1]*IRL.state_scales[1],label="Hip State",linestyle='-',c=color)
    ax.plot(t, points[:, 2]*IRL.state_scales[2],label="Shank State",linestyle=':',c=color)


    ax.set_xlabel("Stance Phase (%)")
    ax.set_ylabel("Joint Displacement (Degrees)",color = color)
    #plt.xticks(np.arange(N_STATES + 1), gp_names)

    ax.tick_params(axis='y', labelcolor=color)
    ax.legend(loc='upper left')
    ax.set_ylim((ax.get_ylim()[0],ax.get_ylim()[1]*y_buffer))

    # Reward Value
    color = 'tab:red'
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Reward Value', color=color)
    ax2.plot(t, points[:, 3], label="Reward Value",c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    ax2.set_ylim((ax2.get_ylim()[0], ax2.get_ylim()[1] * y_buffer))

    save_figures(plt, 'IRL_2D_MaxReward', h=8, w=10)

def update(frame):

    #gp_names=['Initial Contact\n [0%,5%]', 'Loading Response\n (5%,19%]',
    #          'Mid Stance\n (19%,50%]', 'Terminal Stance\n (50%,81%]', 'Pre-Swing\n (81,100%]']
    gp_names = ['Initial Contact', 'Loading Response',
                'Mid Stance', 'Terminal Stance', 'Pre-Swing']
    i=int(frame/5)
    #print(f'Frame: {frame}')
    size = 30
    new_data = {'color': 'r',
                'facecolors': 'r',
                'size': size}

    old_data = {'color': 'k',
                'facecolors': 'none',
                'size': int(size*2)}

    if i<reward_map.shape[0]:
        maxs = [np.where(reward_map[m, :, :] == reward_map[m, :, :].max()) for m in range(i+1)]
        maxs = np.reshape(maxs,(-1,2))+0.5

        colors = [old_data['color'] for _ in range(i)]
        colors.append(new_data['color'])
        facecolors = [old_data['facecolors'] for _ in range(i)]
        facecolors.append(new_data['facecolors'])
        sizes = [old_data['size'] for _ in range(i)]
        sizes.append(new_data['size'])

        #print(f'C{colors}\nF{facecolors}')

        ani_title.set_text(f'Recovered Reward Map\n {gp_names[i]}')
        pax = ani_ax.pcolor(reward_map[i, :, :],zorder=-1)
        maxax, = ani_ax.plot(maxs[:, 1], maxs[:, 0],
                             color='k', zorder=1)
        maxax_scatter = ani_ax.scatter(maxs[:,1],maxs[:,0],
                                       color=facecolors,edgecolors=colors,
                                       s=sizes,zorder=1
                                       )

    elif i<reward_map.shape[0]+2:
        i = reward_map.shape[0]-1
        maxs = [np.where(reward_map[m, :, :] == reward_map[m, :, :].max()) for m in range(i + 1)]
        maxs = np.reshape(maxs, (-1, 2)) + 0.5

        colors = [old_data['color'] for _ in range(i)]
        colors.append(new_data['color'])

        facecolors = [old_data['facecolors'] for _ in range(i)]
        facecolors.append(new_data['facecolors'])

        sizes = [old_data['size'] for _ in range(i)]
        sizes.append(new_data['size'])

        #print(f'C{colors}\nF{facecolors}')

        ani_title.set_text(f'Recovered Reward Map\n {gp_names[i]}')
        pax = ani_ax.pcolor(reward_map[i, :, :], zorder=-1)
        maxax, = ani_ax.plot(maxs[:, 1], maxs[:, 0],
                             color='k', zorder=1)
        maxax_scatter = ani_ax.scatter(maxs[:, 1], maxs[:, 0],
                                       color=facecolors, edgecolors=colors,
                                       s=sizes, zorder=1
                                       )
    else:
        ani_title.set_text(f'Restarting...')
        pax = ani_ax.pcolor(np.zeros((IRL.N_STATES,IRL.N_STATES)))
        maxs =np.array([[0,0]])+0.5
        maxax, = ani_ax.plot(maxs[:, 1], maxs[:, 0], color='r',zorder=-1)
        colors = np.full((1, i), 'k').tolist()
        colors[-1] = 'r'
        maxax_scatter = ani_ax.scatter(maxs[:, 1], maxs[:, 0], color=colors, zorder=-1)

    return pax,maxax,maxax_scatter,ani_title

def IRL_animate_reward(RESCALE_FEATURES=True):
    feature_names = ["Hip Displacement (Deg)", "Shank Displacement (Deg)"]
    ani_ax.set_ylabel(feature_names[0])
    ani_ax.set_xlabel(feature_names[1])
    ani_ax.pcolor(reward_map[0, :, :])
    maxs = np.where(reward_map[0, :, :] == reward_map[0, :, :].max())
    maxs = np.reshape(maxs, (-1, 2))
    ani_ax.plot(maxs[:, 0], maxs[:, 1], color='r')

    if RESCALE_FEATURES:
        scale_x = 1 / IRL.state_scales[2]
        scale_y = 1 / IRL.state_scales[1]
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(round(x / scale_x, 0)))
        ani_ax.xaxis.set_major_formatter(ticks_x)
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(round(x / scale_y, 0)))
        ani_ax.yaxis.set_major_formatter(ticks_y)

    ani = FuncAnimation(ani_fig, update, frames = (IRL.N_STATES+1)*5, interval = 500,blit=True)
    writer = PillowWriter(fps=2)
    ani.save(save_dir + f'{date}_IRL_AnimatedReward.gif', writer=writer)
    print('Saved Animation')

def get_HS_idxs(session_data):
    HS_idxs = []
    for i_session in range(len(session_data)):
        session = session_data[i_session]
        phases = session[:, 0]
        HS = [0]
        for p in range(len(phases) - 1):
            if phases[p + 1] < phases[p]: HS.append(p + 1)
        HS.append(np.size(phases) - 1)
        HS_idxs.append(HS)
    return HS_idxs

##########################################################################
# SESSION PREVIEW ########################################################
def session_features():
    global HS_idxs
    global i_fig

    fig1 = plt.figure(f'Figure: All Features per Session')
    fig1.title = 'Session Preview'
    # var_param = [[Name,Scale Output, Enable]]
    session_data = []
    for name in data_names:
        X, y = DataHandler.import_custom(name)
        new_data = np.append(X, y, axis=1)
        session_data.append(new_data)
    HS_idxs = get_HS_idxs(session_data)

    y_range = (0,60)
    axs1=[]
    legends1 = []

    print(np.shape(session_data))
    for i_session in range(len(session_data)):
        # Add subplot
        n = len(fig1.axes)
        for i in range(n):
            fig1.axes[i].change_geometry(n + 1, 1, i + 1)
        axs1.append(fig1.add_subplot(n + 1, 1, n + 1))




        # Plot with lines at heel strike indexs
        session = session_data[i_session]
        phases = session[:, 0]

        axs1[i_session].vlines(HS_idxs[i_session], min(y_range), max(y_range), colors='r',
                               linestyles=':', linewidth=1, label=features_names[0])

        for var_n in range(1,session.shape[1]-1):
            axs1[i_session].plot(session[:, var_n], label=f'{features_names[var_n]}')
            #print('session shape',session.shape[0])
            #print('session shape/100', )
            #for i in range(1,int(session.shape[0]/100)):
            #    print((i-1)*100,i*100)
            #    t = np.arange((i-1)*100,i*100)
            #    axs1[i_session].plot(t,session[(i-1)*100:i*100,var_n],label=f'{features_names[var_n]}')

        # Add torque on seperate axis
        torques = session[:,-1]
        color = 'tab:red'
        ax2 = axs1[i_session].twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Torque', color=color)
        ax2.plot(torques, label="Reward Value", c=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')
        #ax2.set_ylim((ax2.get_ylim()[0], ax2.get_ylim()[1] * y_buffer))

        axs1[i_session].set_title(data_names[i_session])
        axs1[i_session].set_ylim(y_range)
        leg = axs1[i_session].legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                                ncol=1, borderaxespad=0)
        legends1.append(leg)
        fig1.subplots_adjust(right=0.8)


        print('\n')

    leg = InteractiveLegend.create(legends1)

    save_figures(plt, 'SessionFeatures', w=12, h=10)

##########################################################################
# SINGLE GAITPHASE #######################################################
def gaitphase_features(Data_Names,plot_title,Stats=True,PLOT_ANALYSIS=False,LW=0.2):
    #global HS_indxs
    #global i_fig

    session_data = []
    DataHandler.import_KneeAngle = True
    features_names = ['GP (%)', 'Knee (deg)','Hip (deg)', 'Shank (deg)','Torque (N.m)']
    for name in Data_Names:
        X, y = DataHandler.import_custom(name)
        new_data = np.append(X, y, axis=1)
        session_data.append(new_data)

    n_sessions = len(session_data)
    print(f'n_sessions {n_sessions} ')
    print(f'Session Data Shape {np.shape(session_data)}')
    statistics_cols = ['Feature']
    statistics_rows = ['mean_peak','mean_peak_time', 'mean_duration','mean_range[0]','mean_range[1]']
    statistics = np.array([statistics_rows]).reshape(-1,1)


    ##############################
    # Plotting Features ##########
    ##############################
    plot_cols = n_sessions
    plot_rows  = session_data[0].shape[1]
    if not(PLOT_ANALYSIS): plot_rows=plot_rows-1
    fig = plt.figure(plot_title)
    print(f'Plot row,col {plot_rows, plot_cols}')

    for i_session in range(n_sessions):
        session = session_data[i_session]
        print(f'Shape session {session.shape}')
        session_per_gp = DataHandler.data_per_gaitphase(session[:,:-1],session[:,-1])
        print(f'Shape session_per_gp {session_per_gp.shape},{session_per_gp[0].shape}')
        #session_per_gp = np.reshape(session_per_gp,(session_per_gp.shape[0], 100,5))
        #print(f'Shape session_per_gp {session_per_gp.shape}')

        # generate axes for each feature (excluding gait phase = x axis)
        n_features = np.shape(session_per_gp[0][1])[0]
        print(f'N features {n_features}')
        gp_axs = []
        for f in range(1,plot_rows+1):
            ax = plt.subplot(plot_rows, plot_cols, (f-1)*n_sessions+i_session+1)
            gp_axs.append(ax)

        #print(f'AXS Shape {np.shape(gp_axs)}')
        subphases = {'durations': [np.linspace(0, 5),
                                   np.linspace(5, 19),
                                   np.linspace(19, 50),
                                   np.linspace(50, 81),
                                   np.linspace(81, 100)],
                     # 'colors':  ['blue','orange','green','red','purple']
                     'colors': ['white', 'lightgrey', 'white', 'lightgrey', 'white'],
                     'labels': ['IC', 'LR', 'MS', 'TS', 'PS']}

        # add data to the  plots
        #print(f'SPGP {np.shape(session_per_gp)}')
        for gp in session_per_gp:
            gp_axs[0].set_title(f"All Features - "
                                f"{Data_Names[i_session].split('_')[1]}ing (strides={int(np.shape(session)[0]/100)})\n "
                                f"Patient {Data_Names[i_session].split('P')[-1].split('S')[0]} "
                                f"Session {Data_Names[i_session].split('S')[-1]}\n\n")
            for f in range(1,plot_rows+1):

                gp_axs[f-1].plot(gp[:,f],color='k',linewidth=LW)
                gp_axs[f-1].set_ylabel(features_names[f])


                for i in range(len(subphases['durations'])):
                    sp = subphases['durations'][i]
                    color = subphases['colors'][i]
                    label = subphases['labels'][i]
                    y_scale = 1.4
                    gp_axs[f-1].fill_between(sp, [0], np.max(session[:,f]*y_scale),
                                            facecolor=color, alpha=0.3,)

                    if f == 1:
                        gp_axs[f-1].annotate(label, xy=(np.mean(sp), np.max(session[:,f])*y_scale*0.95),  #xycoords='data',
                                            horizontalalignment='center', verticalalignment='bottom',
                                             color='grey'
                                            )

                    try:
                        gp_axs[f-1].set_ylim((0,np.max(session[:,f])*y_scale*0.95))
                        gp_axs[f-1].set_xlim((0, 100))
                        if f< plot_rows:gp_axs[f-1].set_xticks([])
                        else: gp_axs[f-1].set_xlabel('Stance Phase (%)')
                    except:
                        print(f'Err Session Sample: {session[:,f].shape}')
                        print(f)


            #plt.subplots_adjust(hspace=0.5)
    ##############################
    # Torque Analysis ############
    ##############################

        # calculate metrics for therapist interaction (torque)
        significant_torque = 0.5
        torque_stats = feature_statistics(session_per_gp[:,:,-1],significant_torque = significant_torque)
        mean_range=torque_stats["mean_range"]
        mean_peak = torque_stats["mean_peak"]
        mean_peak_time = torque_stats["mean_peak_time"]
        mean_duration = torque_stats["mean_duration"]
        new_statistics = np.array([[mean_peak],
                                   [mean_peak_time],
                                   [mean_duration],
                                   [mean_range[0]],
                                   [mean_range[1]]]).round(2)


        statistics = np.append(statistics,new_statistics,axis=1)
        statistics_cols.append(Data_Names[i_session])
        if PLOT_ANALYSIS:
            # Plot statistics
            line_width = 3
            gp_axs[-1].plot(torque_stats["mean_gp_torques"][:, 0], torque_stats["mean_gp_torques"][:, 1], c='k')
            gp_axs[-1].plot(np.linspace(mean_range[0], mean_range[1], 10),
                     np.ones(10) * mean_peak,
                     color='cornflowerblue', linestyle=':', linewidth=line_width,label="Meant Peak Torque")

            t = np.linspace(0, 100, 100)
            gp_axs[-1].fill_between(t, significant_torque, mean_peak, where=(t > mean_range[0]) & (t < mean_range[1]), alpha=0.5,
                             facecolor='c',label="Sig Torque x Mean Peak")
            gp_axs[-1].vlines(mean_peak_time, 0, mean_peak, colors='r',
                       linestyles=':', linewidth=line_width, label="Mean Peak Time")
            gp_axs[-1].legend(loc='upper left')
    ##############################
    # Display Results in Terminal#
    ##############################
    print(f'\n\nResults...\n')
    df = DataFrame(statistics,columns=statistics_cols)
    print(df)
    save_figures(plt, plot_title, w=12, h=10)

##########################################################################
# SINGLE GAITPHASE #######################################################

def feature_statistics(torque_per_gp, significant_torque = 1,VERBOSE=False):
    if VERBOSE:
        print(f'feature_statistics...')
        print(f'|\t feature shape {np.shape(torque_per_gp)}')

    peaks = []
    peak_times = []
    actuation_times = []
    durations = []
    intercepts = []
    all_gp_torques = np.empty((0, 2), int)

    for torques in torque_per_gp:
        scale = 100 / torques.size
        # Peak Torque
        peak = max(torques)
        peaks.append(peak)

        # Peak Time
        peak_time = np.mean(np.where(torques == peak)[0]) * scale
        peak_times.append(peak_time)

        # Duration of significant actuation
        actuation_indxs = []
        for i in range(torques.size):
            if torques[i] > significant_torque: actuation_indxs.append(i)
        if len(actuation_indxs) == 0:
            actuation_indxs = [0, 0]
        actuation_time = [min(actuation_indxs) * scale, max(actuation_indxs) * scale]
        actuation_times.append(actuation_time)
        duration = actuation_time[1] - actuation_time[0]
        durations.append(duration)

        # Acutation for session
        gps = np.linspace(0, 100, len(torques)).reshape((-1, 1))
        intercepts.append(torques[0])
        torques = np.reshape(torques, (-1, 1))
        gp_torques = np.append(gps, torques, axis=1)
        all_gp_torques = np.append(all_gp_torques, gp_torques, axis=0)

    # Calculate mean statistics
    mean_peak = np.mean(peaks)
    mean_range = [np.mean([start[0] for start in actuation_times]),
                  np.mean([end[1] for end in actuation_times])]
    mean_duration = np.mean(durations)
    mean_peak_time = np.mean(peak_times)

    # Calculate mean function
    #mean_gp_torques = mean_fun(all_gp_torques[:, 0], all_gp_torques[:, 1], intercept=np.mean(intercepts))

    # Package
    stats = {
        "mean_peak": mean_peak,
        "mean_range": mean_range,
        "mean_duration": mean_duration,
        "mean_peak_time": mean_peak_time,
        #"mean_gp_torques": mean_gp_torques
    }
    if VERBOSE: print(stats)
    return stats



if __name__ == "__main__":
    main()
