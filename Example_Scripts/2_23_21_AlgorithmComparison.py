from Modules.Utils.PredictionUtil import ModelPrediction
import matplotlib.pyplot as plt
import numpy as np


def main():
    ##########################################################################
    # TEST DATA ##############################################################
    ##########################################################################
    data_names= [
        #'processed_P3S3L1',
        #'processed_P3S3L2',
        #'processed_P3S3L3',
        #'trimmed_train_P3S2',
        #'trimmed_train_P3S3',
        'trimmed_test_P3S2',
        'trimmed_test_P3S3'
             ]

    pred = ModelPrediction(data_names)
    #pred.import_GaitPhase = False
    pred.import_KneeAngle = False
    #pred.import_HipAngle = False
    #pred.import_ShankAngle = False
    pred.import_GRF = False
    pred.update_import_enables()

    all_stats = []

    ##########################################################################
    # PRED Linear Regression #################################################
    ##########################################################################
    model_path = '12_20_2020_P3S2_Trim_PHS_O3.pkl'
    #model_path = '12_20_2020_P3S2_Trim_PK_O5.pkl'
    # model_path =   '12_20_2020_P3S3_Trim_PHS_O2.pkl'
    # model_path =  '12_20_2020_P3S3_Trim_PK_O4.pkl'

    lr_stats = pred.linear_regression(model_path)
    all_stats.append(lr_stats)

    ##########################################################################
    # PRED Gaussian Process Regression #######################################
    ##########################################################################
    model_paths = [
        # '12_16_2020_K1_P3S3L1_PHS.pkl',
        # '12_16_2020_K1_P3S3L2_PHS.pkl',
        # '12_16_2020_K1_P3S3L3_PHS.pkl',
        '02_19_2021_K1_P3S3_Trim_PHS.pkl',
        # '12_16_2020_K1_P3S2_Trim_PHS.pkl',
        # '12_16_2020_K1_P3S3_Trim_PHS.pkl'
    ]
    gpr_stats = pred.bayesian_committee(model_paths)
    all_stats.append(gpr_stats)

    ##########################################################################
    # PRED Inverse Reinforcement Learning ####################################
    ##########################################################################
    model_name_irl= '03_02_2021_IRL_P3S3_PHS_2sx2a_obj'
    irl_stats = pred.irl(model_name_irl)
    all_stats.append(irl_stats)

    ##########################################################################
    # PLOTTING ###############################################################
    ##########################################################################
    algorithm_names = ['LR','GPR','IRL']
    n_algorithms = 3
    n_datasets = len(data_names)

    plot_cols = n_algorithms
    plot_rows = n_datasets
    axs = []

    r = -1
    for i in range(1, plot_cols*plot_rows+1):

        # Set up plots
        ax = plt.subplot(plot_rows, plot_cols, i)
        c=(i-1)%plot_cols
        if c==0:
            r += 1
            ax.set_ylabel(data_names[r])
        if r==0:
            ax.set_title(algorithm_names[c])

        # Plot data
        if c==0:
            ax=pred.plot_lr(ax,all_stats[c][r])
        if c==1:
            ax=pred.plot_gpr(ax,all_stats[c][r])
        if c==2:
            ax=pred.plot_irl(ax,all_stats[c][r],PLOT_WEIGHTED=True)

        axs.append(ax)


    plt.show()

if __name__=="__main__":
    main()