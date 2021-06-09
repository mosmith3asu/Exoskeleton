import pickle
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from math import sqrt
from Modules.Utils import Data_Filters
from Modules.Utils.DataHandler import Data_Handler
from Modules.Utils.Data_Filters import lfilter,butterworth
class ModelPrediction:
    def __init__(self,data_names, root_dir='C:\\Users\\mason\\Desktop\\Thesis\\Patient_files'):
        self.import_GaitPhase = True
        self.import_KneeAngle = False
        self.import_HipAngle = True
        self.import_ShankAngle = True
        self.import_GRF = False

        self.DataHandler = Data_Handler()

        self.data_names = data_names
        self.X_obs = []
        self.y_obs =[]

        self.lr_preds = []
        self.gpr_preds = []
        self.irl_preds_d = []
        self.irl_preds_w = []

        self.y_lim = (0,3.5)

    def update_import_enables(self):
        self.DataHandler.import_GaitPhase = self.import_GaitPhase
        self.DataHandler.import_KneeAngle = self.import_KneeAngle
        self.DataHandler.import_HipAngle =  self.import_HipAngle
        self.DataHandler.import_ShankAngle = self.import_ShankAngle
        self.DataHandler.import_GRF = self.import_GRF

    def get_obs(self):
        for name in self.data_names:
            # Import Data ########################
            print('Test Data: ', name)
            X_obs, y_obs = self.DataHandler.import_custom(name)
            self.X_obs.append(X_obs)
            self.y_obs.append(y_obs)

        return self.X_obs,self.y_obs

    def per_gaitphase(self, features, labels,pred):
        # Get heel strike indexes
        pred = np.reshape(pred,(-1,1))

        if np.shape(labels[1]) == ():
            labels = np.reshape(labels, (-1, 1))

        HS = [0]
        # gait_phase = self.train_X[:, 0]
        gait_phase = features[:, 0]
        for phase in range(len(gait_phase) - 1):
            if gait_phase[phase + 1] < gait_phase[phase]: HS.append(phase + 1)
        HS.append(np.size(labels))

        obs_per_gaitphase = []
        pred_per_gaitphase = []
        for hs in range(len(HS) - 1):
            phase_labels = labels[HS[hs]:HS[hs + 1], :]
            obs_per_gaitphase.append(phase_labels)

            pred_labels = pred[HS[hs]:HS[hs + 1], :]
            pred_per_gaitphase.append(pred_labels)

        obs_per_gaitphase = np.array(obs_per_gaitphase)
        pred_per_gaitphase = np.array(pred_per_gaitphase)

        return obs_per_gaitphase, pred_per_gaitphase

    def plot_per_gaitphase(self,ax,y_gp,PLOT_SUBPHASES=False,SUBPHASE_HEADERS=False):
        for gp in y_gp:
            ax.plot(gp, color='k', linewidth=0.5)
            ax.set_ylim(self.y_lim)

            if PLOT_SUBPHASES:
                #print(f'\n\n\n\n\n\nPRINTING SUBPHSAES')
                #print(self.y_lim)
                subphases = {'durations': [np.linspace(0, 5),
                                           np.linspace(5, 19),
                                           np.linspace(19, 50),
                                           np.linspace(50, 81),
                                           np.linspace(81, 100)],
                             # 'colors':  ['blue','orange','green','red','purple']
                             'colors': ['white', 'lightgrey', 'white', 'lightgrey', 'white'],
                             'labels': ['IC', 'LR', 'MS', 'TS', 'PS']}
                for i in range(len(subphases['durations'])):
                    sp = subphases['durations'][i]
                    color = subphases['colors'][i]
                    label = subphases['labels'][i]

                    ax.fill_between(sp, [0], self.y_lim[1],
                                    facecolor=color, alpha=0.5, zorder=1)
                    if SUBPHASE_HEADERS:
                        ax.annotate(label, xy=(np.mean(sp),self.y_lim[1]),
                                    # xycoords='data',
                                    horizontalalignment='center', verticalalignment='bottom',
                                    color='grey'
                                    )
        return ax
    ##########################################################################
    # LINEAR REGRESSION ######################################################
    ##########################################################################
    def linear_regression(self,model_path,
                    model_dir = 'C:\\Users\\mason\Desktop\\Thesis\\ML_MasonSmithGit\Example_Scripts\\GeneratedModels\\LinearRegression\\',
                    VERBOSE=True):
        from sklearn.preprocessing import PolynomialFeatures

        train_name = ""
        train_name = train_name + "-" + model_path.split("_")[-4].split(".")[0]

        with open(model_dir + model_path, 'rb') as f:
            linreg = pickle.load(f)
            model_order = int(model_path.split("O")[-1].split(".")[0])

        all_stats=[]

        for name in self.data_names:
            print('Test Data: ', name)
            X_obs, y_obs = self.DataHandler.import_custom(name)
            if VERBOSE: print("Shape of feature data:", np.shape(X_obs))
            if VERBOSE: print("Shape of label data :", np.shape(y_obs))

            poly = PolynomialFeatures(degree=model_order)
            X_obs_poly = poly.fit_transform(X_obs)
            y_pred = linreg.predict(X_obs_poly)
            self.lr_preds.append(y_pred)

            mse = np.mean([(y_pred[i] - y_obs[i])**2 for i in range(len(y_pred))])
            r2 = r2_score(y_obs, y_pred)

            statistics = {"r2": r2,
                          "mse": mse,
                          "prediction": y_pred,
                          "observation": y_obs,
                          "name": name}

            all_stats.append(statistics)

        return all_stats

    def plot_lr(self, ax, lr_stats):
        # Import prediction statistics
        r2 = lr_stats["r2"]
        mse = lr_stats["mse"]
        y_pred = lr_stats["prediction"]
        y_obs = lr_stats["observation"]
        name = lr_stats["name"]

        # Plot result
        ax.plot(y_obs, linestyle=":")
        ax.plot(y_pred)
        legend_list = ['Obs', "Pred"]
        ax.legend(legend_list, loc='upper right', fontsize='x-small')
        ax.set_ylim(self.y_lim)

        # Add scores to plot
        y_lim = ax.get_ylim()[1]
        ax.text(0, y_lim - y_lim * 0.05, f'r2={round(r2, 2)} \nMSE = {round(mse, 2)}',
                fontsize=8, ha='left', va='top')
        return ax

    ##########################################################################
    # BAYESIAN COMMITTEE #####################################################
    ##########################################################################
    def bayesian_committee(self,commitee_paths,
                           model_dir = 'C:\\Users\\mason\Desktop\\Thesis\\ML_MasonSmithGit\Example_Scripts\\GeneratedModels\\GPR\\',
                                   VERBOSE=True, FILTERED = True):

        print('\n\n##############################')
        print('Predict Bayesian Committee')
        print('##############################')
        self.DataHandler = Data_Handler()
        self.DataHandler.import_GaitPhase = self.import_GaitPhase
        self.DataHandler.import_KneeAngle = self.import_KneeAngle
        self.DataHandler.import_HipAngle = self.import_HipAngle
        self.DataHandler.import_ShankAngle = self.import_ShankAngle
        self.DataHandler.import_GRF = self.import_GRF



        errors = np.array([])
        # load in all of the models
        commitee = []
        for model_path in commitee_paths:
            with open(model_dir + model_path, 'rb') as f:
                gp = pickle.load(f)
                commitee.append(gp)


        all_stats=[]

        for name in self.data_names:
            # Import Data ########################
            print('Test Data: ', name)
            X_obs, y_obs = self.DataHandler.import_custom(name)
            if VERBOSE: print("Shape of feature data:", np.shape(X_obs))
            if VERBOSE: print("Shape of label data :", np.shape(y_obs))

            # Predictions ########################
            if len(commitee)==1:
                y_pred, sigma = self.single_predict(X_obs,commitee,return_std=True)
            else:
                y_pred, sigma = self.commitee_predict(X_obs, commitee, return_std=True)

            if FILTERED:
                n_iter = 10
                y_pred = Data_Filters.lfilter(y_pred, n_iter)
            self.gpr_preds.append(y_pred)

            # R2 Calculations #####################
            score = r2_score(y_obs, y_pred)
            print(f"R2: {score}")

            # MSE Calculations ####################
            mse = np.mean(abs(np.reshape(y_pred, (1, -1)) - np.reshape(y_obs, (1, -1))))  # np.mean(abs(y_pred-y_obs))
            print('MSE: ', mse, '\n')
            errors = np.append(errors, abs(np.reshape(y_pred, (1, -1)) - np.reshape(y_obs, (1, -1))))

            # Package for export ###################
            statistics = {"r2": score,
                          "mse": np.mean(errors),
                          "sigma": sigma,
                          "prediction": y_pred,
                          "observation": y_obs,
                          "name": name}
            all_stats.append(statistics)

        return all_stats

    def single_predict(self,test_data, models, has_labels=False, return_std=False):
        print("Single GPR Prediction...")


        if has_labels:
            y_obs = test_data[:, -1]
            X_obs = test_data[:, 0:2]
        else:
            X_obs = test_data

        for expert in models:
            mu, s = expert.predict(X_obs, return_std=True)

        if return_std:
            mu = mu.flatten()
            return mu, s
        else:
            mu = mu.flatten()
            return mu

    def commitee_predict(self,test_data, models, has_labels=False, return_std=False):
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
        print(X_obs[0, :])
        for expert in models:
            mu, s = expert.predict(X_obs, return_std=True)
            s2 = s * s
            mu_experts.append(np.reshape(mu, (-1, 1)))
            s2_experts.append(np.reshape(s2, (-1, 1)))

        # Initialize aggrogated mu and s2
        mu = np.zeros((num_test, 1))
        s2 = np.zeros((num_test, 1))

        # Begin aggregation
        kss = 1  # weight of expert at X
        for i in range(M):
            s2 = s2 + 1. / s2_experts[i]
        print('S2',np.shape(s2))
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

    def plot_gpr(self,ax, gpr_stats):

        # Import prediction statistics
        r2 = gpr_stats["r2"]
        mse = gpr_stats["mse"]
        sigma = gpr_stats["sigma"]
        y_pred = gpr_stats["prediction"]
        y_obs = gpr_stats["observation"]
        name = gpr_stats["name"]
        current_data_name = name.split("\\")[-1].split(".")[0]

        # Plot result
        # axs[plt_num].set_title(
        #    f"Model Trained on: {train_name} Compared Against: {current_data_name} (MSE: {round(mse, 2)} R2: {round(r2, 2)})")
        y_upperbound = [y_pred[i] + 1.9600 * sigma[i] for i in range(len(y_pred))]
        y_lowerbound = [y_pred[i] - 1.9600 * sigma[i] for i in range(len(y_pred))]
        x = np.arange(0, len(y_pred))
        ax.plot(y_obs, linestyle=":")
        ax.plot(y_pred)
        ax.fill_between(x, y_upperbound, y_lowerbound, alpha=0.5, interpolate=True, facecolor="grey")
        legend_list = ['Obs', "Pred", "95% Conf"]
        ax.legend(legend_list, loc='upper right', fontsize='x-small')
        ax.set_ylim(self.y_lim)

        # Add scores to plot
        y_lim = ax.get_ylim()[1]
        ax.text(0, y_lim - y_lim * 0.05, f'r2={round(r2, 2)} \nMSE = {round(mse, 2)}',
                fontsize=8, ha='left', va='top')
        return ax

    ##########################################################################
    # INVERSE REINFORCMENT LEARNING ##########################################
    ##########################################################################
    def irl(self,model_name,
                    model_dir = 'C:\\Users\\mason\Desktop\\Thesis\\ML_MasonSmithGit\Example_Scripts\\GeneratedModels\\IRL\\',
                    filter_BWxL = None, filt_iter=2,VERBOSE=True):

        # Import IRL object
        path = model_dir+model_name
        with open(path, 'rb') as IRL_obj:
            IRL=pickle.load(IRL_obj)

        try:
            print(IRL.USE_SUBPHASES)
        except:
            IRL.USE_SUBPHASES = False
            IRL.subphases = {'durations': [[0, 5],
                                           [5, 19],
                                           [19, 50],
                                           [50, 81],
                                           [81, 100]],
                             # 'colors':  ['blue','orange','green','red','purple']
                             'colors': ['white', 'lightgrey', 'white', 'lightgrey', 'white'],
                             'labels': ['IC', 'LR', 'MS', 'TS', 'PS']}
            IRL.unique_states = [IRL.N_STATES,IRL.N_STATES,IRL.N_STATES]

        # Compute statistics
        all_stats=[]
        for name in self.data_names:
            # Import observed data
            X_obs, y_obs = self.DataHandler.import_custom(name)
            # Convert to state indexes
            idx_test = IRL.get_idx_demo(X_obs, y_obs)

            obs_states = []
            pred_action_d = []
            for trajectory in idx_test:
                for i in range(np.shape(trajectory)[0]):
                    state = trajectory[i, 0]
                    # if VERBOSE: print(f'|\t State: {state}')
                    obs_states.append(state)
            for obs in obs_states:
                # pred_action.append(np.argmax(policy[obs, :]))
                prob_of_action = IRL.policy[obs, :]
                pred_action_discrete = np.argmax(prob_of_action)

                pred_action_weighted = 0
                for a in range(len(prob_of_action)):
                    pred_action_weighted += prob_of_action[a] * a

                pred_action_d.append(pred_action_discrete)

            scale = 1.01 * (IRL.max_action - IRL.min_action) / IRL.n_actions
            pred_torque_d = np.array(pred_action_d) * scale + IRL.min_action


            if filter_BWxL=="L": pred_torque_d=lfilter(pred_torque_d,n_iter=filt_iter)
            elif filter_BWxL=="BW": pred_torque_d= butterworth(pred_torque_d,order=10, w_n=100)

            self.irl_preds_d.append(pred_torque_d)

            mse_d = np.mean([(pred_torque_d[i] - y_obs[i])**2 for i in range(len(pred_torque_d))])
            r2_d = r2_score(y_obs, pred_torque_d)


            statistics = {"r2_discrete": r2_d,
                          "mse_discrete": mse_d,
                          "prediction_discrete": pred_torque_d,
                          "observation": y_obs,
                          "name": name}

            all_stats.append(statistics)

        return all_stats

    def plot_irl(self, ax, irl_stats,PLOT_WEIGHTED=False):

        # Import prediction statistics
        r2_d = irl_stats["r2_discrete"]
        mse_d = irl_stats["mse_discrete"]
        y_pred_d = irl_stats["prediction_discrete"]

        y_obs = irl_stats["observation"]
        name = irl_stats["name"]

        # Plot result
        ax.plot(y_obs, linestyle=":")
        ax.plot(y_pred_d)
        ax.set_ylim(self.y_lim)
        legend_list = ['Obs', "Pred"]


        ax.legend(legend_list, loc='upper right', fontsize='x-small')

        # Add scores to plot
        y_lim = ax.get_ylim()[1]
        ax.text(0, y_lim - y_lim * 0.05, f'r2_d={round(r2_d, 2)} \nMSE_d = {round(mse_d, 2)}',
                fontsize=8, ha='left', va='top')
        return ax