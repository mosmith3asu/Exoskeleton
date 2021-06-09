import os
import numpy as np
from numpy import genfromtxt,savetxt
from pyquaternion import Quaternion
from math import degrees
from Modules.Utils.Data_Filters import lfilter
from time import sleep
class Data_Handler:
    def __init__(self,root_dir='C:\\Users\\mason\\Desktop\\Thesis\\Patient_files'):
        self.root_dir = root_dir
        self.processed_dir=self.root_dir+'\\ProcessedData'
        self.raw_dir = self.root_dir + '\\RawData'
        self.output_dir = self.root_dir+'\\Outputs'
        self.return_enable=np.array([
            ['Rating'	,True], ['Gaitphase',True],
            ['Knee Angle',False],['Hip Angle',True],
            ['Shank Angle',True],	['Torque',True],
            ['GFR1',False],	['GRF2',False],
            ['GRF3',False],	['GRF4',False],
            ['HS Index',False],	['SW index',False]])

        self.name_lst = np.array([['Gait Phase','Knee Angle','Hip Angle','Shank Angle',
                                   'GRF1','GRF2','GRF3','GRF4']])

        self.filter_torque = True

        self.import_GaitPhase = True
        self.import_KneeAngle = False
        self.import_HipAngle = True
        self.import_ShankAngle = True
        self.import_GRF = False

        self.import_enables = [self.import_GaitPhase,self.import_KneeAngle, self.import_HipAngle,
                               self.import_ShankAngle,self.import_GRF]

        self.index_GaitPhase = 0
        self.index_KneeAngle = 1
        self.index_HipAngle = 2
        self.index_ShankAngle = 3
        self.index_GRF = np.arange(4,8)

        self.train_X = np.array([[]])
        self.test_X = np.array([[]])

        self.train_Y = np.array([[]])
        self.test_Y = np.array([[]])

        self.tracker = np.empty((0,3))

    def data_per_gaitphase(self,features,labels):
        # Get heel strike indexes
        if np.shape(labels[1])==():
            labels = np.reshape(labels,(-1,1))

        HS = [0]
        #gait_phase = self.train_X[:, 0]
        gait_phase = features[:, 0]
        for phase in range(len(gait_phase) - 1):
            if gait_phase[phase + 1] < gait_phase[phase]: HS.append(phase + 1)
        HS.append(np.size(gait_phase))
        features_per_gaitphase=[]
        for hs in range(len(HS) - 1):
            phase_features = features[HS[hs]:HS[hs + 1],:]
            phase_labels = labels[HS[hs]:HS[hs + 1], :]
            current_phase= np.append(phase_features,phase_labels,axis=1)
            features_per_gaitphase.append(current_phase)

        features_per_gaitphase=np.array(features_per_gaitphase)
        return features_per_gaitphase

    def verify_import_with_graph(self,x=[],show_plot=True,plot_title="All Features per Session"):
        import matplotlib.pyplot as plt
        if np.size(x)>0: features,_ = self.filter_excluded_features(self.train_X)
        else: features,_ = self.filter_excluded_features(x)

        figs = []
        axs_lst = []
        y_range = (0, 60)

        # Get heel strike indexes
        HS = []
        gait_phase = self.train_X[:,0]
        for phase in range(len(gait_phase) - 1):
            if gait_phase[phase + 1] < gait_phase[phase]: HS.append(phase + 1)

        # Set up figure
        fig = plt.figure(plot_title)
        fig.title = 'Session Preview'

        ##########################
        # Add Input Features
        #########################
        # Plot each feature by gait phase in a new axs
        for i_feature in range(1,features.shape[1]):

            # extract feature from training data
            feature = features[:,i_feature]

            # Add subplot
            if i_feature<self.index_GRF[0]:
                n = len(fig.axes)
                for i in range(n):
                    fig.axes[i].change_geometry(n + 1, 1, i + 1)
                axs = fig.add_subplot(n + 1, 1, n + 1)
                axs_lst.append(axs)

                axs.set_ylabel(f'{self.name_lst[0, i_feature]}')

            if i_feature==1: axs.set_title('Imported Data. Close to Continue')

            # Plot feautre per phase
            for hs in range(len(HS) - 1):
                current_phase = feature[HS[hs]:HS[hs + 1]]
                t = np.linspace(0, 100, len(current_phase))
                axs.plot(t, current_phase,linewidth=0.5,color='k')

                # extract feature from training data

        ##########################
        # Add Torque
        #########################
        feature = self.train_Y[:]

        # Add subplot
        n = len(fig.axes)
        for i in range(n):
                fig.axes[i].change_geometry(n + 1, 1, i + 1)
        axs = fig.add_subplot(n + 1, 1, n + 1)
        axs_lst.append(axs)
        axs.set_ylabel(f'Torque')

        for hs in range(len(HS) - 1):
            current_phase = feature[HS[hs]:HS[hs + 1]]
            t = np.linspace(0, 100, len(current_phase))
            axs.plot(t, current_phase,linewidth=0.5,color='k')

        if show_plot:plt.show()
        else: return plt

    def check_processed(self):
        print("Importing patient data files:")
        self.csv_names = []
        self.csv_dirs = []
        for root, dirs, files in os.walk(self.processed_dir):
            for file in files:
                if file.endswith('.csv'):
                    print(file)
                    self.csv_names.append(file)
                    self.csv_dirs.append(self.processed_dir + '\\' + file)

        print("importing data...")
        # self.patient_data = []

    def import_PSL_processed(self,patient,session,lap):
        path = self.processed_dir +'\\'+'processed_'+'P'+str(patient)+'S'+str(session)+'L'+str(lap)+'.csv'
        import_data = self.format_data(genfromtxt(path, delimiter=','))
        return import_data

    def import_PSL_raw(self,patient,session,lap):
        path = self.raw_dir +'\\'+'P'+str(patient)+'S'+str(session)+'L'+str(lap)+'.csv'
        import_data = self.format_data(genfromtxt(path, delimiter=','))
        return import_data

    def filter_excluded_features(self,data):

        del_lst = []
        headers = self.name_lst

        if self.import_GaitPhase == False:
            del_lst.append(self.index_GaitPhase)
        if self.import_KneeAngle == False:
            del_lst.append(self.index_KneeAngle)
        if self.import_HipAngle == False:
            del_lst.append(self.index_HipAngle)
        if self.import_ShankAngle == False:
            del_lst.append(self.index_ShankAngle)
        if self.import_GRF == False:
            [del_lst.append(i) for i in self.index_GRF]

        if np.size(del_lst)>0:
            data= np.delete(data, del_lst, 1)
            headers = np.delete(headers,del_lst,1)

        self.import_enables = [self.import_GaitPhase, self.import_KneeAngle, self.import_HipAngle,
                               self.import_ShankAngle, self.import_GRF]

        return data,headers

    def import_custom(self,names,VERBOSE=True):
        if type(names) != list: names = [names]

        if VERBOSE:
            print(f'|\tDataHandler.import_custom Report... ')
            print(f'|\t|\t File(s): \t {names}')

        X_data= np.array([[]])
        Y_data = np.array([[]])
        for name in names:
            path = self.processed_dir + '\\' + name + '.csv'
            import_data = genfromtxt(path, delimiter=',')[:,0:np.size(self.name_lst)+1]
            import_data = import_data[1:,:]
            X = import_data[:, 1:]
            Y = import_data[:, 0].reshape(-1,1)

            if np.size(X_data) < 1:
                X_data = X
                Y_data = Y
            else:
                X_data = np.append(X_data, X,axis=0)
                Y_data = np.append(Y_data, Y,axis=0)

            if 'train' in name: import_type = 'Train'
            elif 'test' in name: import_type= 'Test'
            else: import_type = 'Unspecified'

        X_data,headers = self.filter_excluded_features(X_data)

        if VERBOSE:
            print(f'|\t|\t Imported {import_type} Data: \t {headers}')
            print("|\t|\t Shape of X|Y: \t", np.shape(X_data), '|', np.shape(Y_data))


        return X_data,Y_data

        # if 'train' in name:
        #     import_type = 'Train'
        #     if np.size(self.train_X)<1:
        #         self.train_X = X
        #         self.train_Y = Y
        #
        #     else:
        #         self.train_X = np.append(self.train_X,X)
        #         self.train_Y = np.append(self.train_Y,Y)
        # elif 'test' in name:
        #     import_type= 'Test'
        #     if np.size(self.test_X) < 1:
        #         self.test_X = X
        #         self.test_Y = Y
        #     else:
        #         self.test_X = np.append(self.train_X, X)
        #         self.test_Y = np.append(self.train_Y, Y)
        # else:
        #     import_type = 'Unspecified'

    def interpolate_gaitphase(self,data_name,output_name='None'):
        path = self.processed_dir + '\\' + data_name
        if output_name== 'None':
            output_name = self.processed_dir + '\\interpolated_' + data_name

        all_data = genfromtxt(path, delimiter=",")
        N_rows = np.size(all_data[:,0])
        print('Rows:',N_rows)


        HS_indexs = np.array([0])
        all_HS = all_data[1:, -2]
        all_HS = all_HS[~np.isnan(all_HS)]
        mean_between = np.mean(all_HS[1:] - all_HS[:-1])
        HS_indexs = np.append(HS_indexs, all_HS)
        #HS_indexs = np.append(HS_indexs, len(all_data))
        HS_indexs = np.append(HS_indexs, HS_indexs[-1]+mean_between)
        print("HS_Indexes:", HS_indexs)

        gait_phase = []


        for i in range(len(HS_indexs) - 1):
            val = np.linspace(0, 100, num=int(HS_indexs[i + 1] - HS_indexs[i])).tolist()
            gait_phase = np.append(gait_phase, val)
        while np.size(gait_phase)<N_rows:
            HS_indexs = np.append(HS_indexs, HS_indexs[-1] + mean_between)
            val = np.linspace(0, 100, num=int(HS_indexs[i + 1] - HS_indexs[i])).tolist()
            gait_phase = np.append(gait_phase, val)

        print('Size GP:',np.size(gait_phase))
        gait_phase=gait_phase[:N_rows]
        print('Size GP Trimmed:', np.size(gait_phase))
        gait_phase = np.reshape(gait_phase, (-1, 1))
        all_data[:,1] = gait_phase.transpose()
        #all_data.astype(np.str)
        #np.concatenate((self.header,all_data),axis=0)
        #all_data[0,:] = self.header
        export = all_data
        #savetxt("interpolated_gaitphase_data.csv" ,export, delimiter=',')
        savetxt(output_name, export, delimiter=',')

    def append_names(self,names):
        data_name=""
        for name in names:
            name = name.split("\\")[-1].split(".")[0]
            name = name.split("_")[-1].split(".")[-1]
            data_name = data_name + name
        return data_name

    def feature_prefix(self):
        prefix = ""
        if self.import_GaitPhase: prefix += "P"
        if self.import_KneeAngle: prefix+="K"
        if self.import_HipAngle: prefix += "H"
        if self.import_ShankAngle: prefix += "S"
        if self.import_GRF: prefix +="G"
        return prefix

    def feature_accronym(self):
        features_acc = ''
        for label, enable in self.return_enable:
            if label != 'Rating' and label != 'Torque' and enable == 'True':
                accronym=''
                if label=='Gaitphase': accronym = 'P'
                elif label == 'Knee Angle': accronym = 'K'
                elif label == 'Hip Angle': accronym = 'H'
                elif label == 'Shank Angle': accronym = 'S'
                elif label == 'GFR1': accronym = 'G'
                features_acc = features_acc + accronym

        return features_acc

    def resample_nth(self,data,sample_every_nth):
        data = data[::sample_every_nth, :]
        return data

    def resample_cap(self,data,max_data_points):
        if np.shape(data[1])==():
            print(f'\nDataHandler.resample_cap... \n|\t reshaped data \n|\t from {np.shape(data)} ')
            data = np.reshape(data,(-1,1))
            print(f'|\t to {np.shape(data)} ')


        n_pts = np.size(data[:,0])
        if n_pts>=max_data_points and n_pts>1:
            diff= n_pts-max_data_points
            del_indexs= np.linspace(1,n_pts,diff).astype(np.int)-1
            new_data = np.delete(data,del_indexs,0)
            #sample_every_nth = int(n_pts/max_data_points)
            #data = data[::sample_every_nth, :]
        return new_data

    def save_model(self,model,MODEL_NAME,plt=None):
        import pickle
        # save
        name = "GeneratedModels\\" + MODEL_NAME
        with open(name + ".pkl", 'wb') as f:
            pickle.dump(model, f)
        if plt!=None:
            plt.savefig(name + ".png")

    def print_features(self):
        features = []
        for label, enable in self.return_enable:
            if label != 'Rating' and label != 'Torque' and enable == 'True':
                features.append(label + ', ')
        print(f'Features ({np.size(features)}):\n', features,'\n')
        return features
    def quat2KneeHipShank_BU(self, quat_csv_path, init_idxs, first_steps,
                          name_prefix='', plot_output=True, save_output=True,
                          filtered=True, filter_iter=7,verbose=True):
        """
        Input Variables:
            init_idxs: list of length n_lap describing indexs where knee was set to zero displacement
            first_steps: list of length n_lap of either <'flexion' or 'extension'> defining dynamcis of first step
        Where
            n_lap: the number of laps walked by the patient
        """


        from Modules.Utils import Data_Filters
        import matplotlib.pyplot as plt
        from Modules.Utils.typeAtools.formatted_outputs import printh

        # Import Quat data
        data = genfromtxt(quat_csv_path, delimiter=',')
        init_idxs.append(len(data))
        last_maxmin = 0

        # Define Reference Vectors
        null_vec = [1, 0, 0]
        dir_vec = [1, 0, 0]  # Direction specific reference/null vector on transverse plane

        # Initialize arrays
        printh(0,'Begining [Quat] to [Knee,Hip,Shank] transformation...')
        printh(1, f'init_idxs (n={len(init_idxs)-1}): {init_idxs}')
        printh(1, f'first_steps (n={len(first_steps)}): {first_steps}')
        #print('\nBegining [Quat] to [Knee,Hip,Shank] transformation...')
        ##############################################
        # ITERATE THROUGH ALL DATA BEFORE WALKING LAPS
        uninit_data = np.zeros(init_idxs[0]).tolist()
        knee_angles = uninit_data
        hip_angles = uninit_data
        shank_angles = uninit_data


        ##################################
        # ITERATE THROUGH ALL WALKING LAPS
        printh(0, 'Lap Outputs')
        for idx in range(len(init_idxs[:-1])):

            # Define new reference rotation
            init_quats = data[init_idxs[idx], :]

            # Define walking direction
            first_step = first_steps[idx]
            if first_step == 'e': in_extension = True
            elif first_step == 'f': in_extension = False
            else: print(f'ERROR: "first_step={first_step}" is not a valid entry\n please enter "e" for extension or "f" for flexion')



            # ITERATE THROUGH ALL SAMPLES IN THIS WALKING LAP
            for i in range(init_idxs[idx], init_idxs[idx+1]):

                # Get this quaternion rotation and vectors
                quats = data[i, :]
                thigh_quat, femur_quat = self.quat2KneeHipShank_initialize_quats(quats, init_quats)
                thigh_vec = thigh_quat.rotate(null_vec)
                shank_vec = femur_quat.rotate(null_vec)

                knee = self.quat2KneeHipShank_angBetweenV(thigh_vec, shank_vec)  # - init_angles[0]# + start_displace
                hip = self.quat2KneeHipShank_angBetweenV_hipang(thigh_vec, shank_vec, -1 * null_vec)  # - init_angles[1]
                shank = self.quat2KneeHipShank_angBetweenV_hipang(shank_vec, thigh_vec,
                                                                  -1 * null_vec)  # - init_angles[2]
                knee_angles.append(knee)
                hip_angles.append(hip)
                shank_angles.append(shank)

                # If crossing midplane with thigh then move to opposite extension/flexion

            printh(1, f'Lap {idx}')
            printh(2, f'Idx Range = [{init_idxs[idx]},{init_idxs[idx + 1]}]')
            printh(2, f'Lap Size = {len(hip_angles)}')
        if filtered:
            knee_angles = Data_Filters.lfilter(knee_angles, filter_iter)
            hip_angles = Data_Filters.lfilter(hip_angles, filter_iter)
            shank_angles = Data_Filters.lfilter(shank_angles, filter_iter)

        if plot_output:
            # Plotting figures
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax_kneeAngles = fig.add_subplot(3, 1, 1)
            ax_hipAngles = fig.add_subplot(3, 1, 2)
            ax_shankAngles = fig.add_subplot(3, 1, 3)

            ax_kneeAngles.set_title(f'Knee Angles | Filtered:{filtered}')
            ax_hipAngles.set_title(f'Hip Angles | Filtered:{filtered}')
            ax_shankAngles.set_title(f'Shank Angles | Filtered:{filtered}')

            ax_kneeAngles.vlines(init_idxs, min(knee_angles), max(knee_angles), colors='r', linestyles=':')
            ax_hipAngles.vlines(init_idxs, min(hip_angles), max(hip_angles), colors='r', linestyles=':')
            ax_shankAngles.vlines(init_idxs, min(shank_angles), max(shank_angles), colors='r', linestyles=':')

            #ax_hipAngles.plot(hip_angles)
            #ax_shankAngles.plot(shank_angles)
            #ax_kneeAngles.plot(knee_angles)
            ax_kneeAngles.scatter(np.arange(0, len(knee_angles)), knee_angles, s=1)
            ax_hipAngles.scatter(np.arange(0,len(hip_angles)),hip_angles,s=1)
            ax_hipAngles.plot([0 for pt in range(len(hip_angles))],c='k',linewidth=1)
            ax_shankAngles.scatter(np.arange(0,len(shank_angles)),shank_angles,s=1)


            plt.subplots_adjust(right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.5)

        knee_angles = np.reshape(knee_angles, (-1, 1))
        hip_angles = np.reshape(hip_angles, (-1, 1))
        shank_angles = np.reshape(shank_angles, (-1, 1))

        KH = np.append(knee_angles, hip_angles, axis=1)
        KHP = np.append(KH, shank_angles, axis=1)

        if save_output:
            from datetime import datetime
            time_begin = datetime.now()  # current date and time
            date = time_begin.strftime("%m_%d_%Y")
            if name_prefix != '': name_prefix = name_prefix + '_'
            name = f'{self.output_dir}\\{date}_{name_prefix}Quat2KneeHipShank'
            savetxt(name + '.csv', KHP, delimiter=',')
            if plot_output: plt.savefig(name + ".png", dpi=600)
            print('Saved Output:\n', name)

        if plot_output:
            print('Plotting output. Exit plot to continue...')
            plt.show()

        return KHP

    def quat2KneeHipShank(self, quat_csv_path, init_idxs,name_prefix='',
                          plot_type='line',plot_output=True,save_output=True,
                          filtered=True, filter_iter=7,filt_denom = 1):
        """
        Input Variables:
            init_idxs: list of length n_lap describing indexs where knee was set to zero displacement
            first_steps: list of length n_lap of either <'flexion' or 'extension'> defining dynamcis of first step
        Where
            n_lap: the number of laps walked by the patient
        """

        from Modules.Utils import Data_Filters
        import matplotlib.pyplot as plt
        from Modules.Utils.typeAtools.formatted_outputs import printh

        # Import Quat data
        data = genfromtxt(quat_csv_path, delimiter=',')
        init_idxs.append(len(data))

        printh(0, 'Begining [Quat] to [Knee,Hip,Shank] transformation...')
        printh(1, f'init_idxs (n={len(init_idxs) - 1} + 1): {init_idxs}')

        # Initialize arrays
        null_vec = [1, 0, 0]
        uninit_data = np.zeros(init_idxs[0]).tolist()
        knee_angles = [] + uninit_data
        hip_angles = [] + uninit_data
        shank_angles = [] + uninit_data

        ##################################
        # ITERATE THROUGH ALL WALKING LAPS
        ##################################
        printh(0, 'Lap Outputs')
        printh(1, f'Lap 0 (Before Walking)')
        printh(2, f'Idx Range = [0,{init_idxs[0]}]')
        printh(2, f'Lap Size = {len(hip_angles)}')

        for idx in range(len(init_idxs[:-1])):

            # Define new reference rotation
            init_quats = data[init_idxs[idx], :]
            #if is_inverted: init_quats[4:] = Quaternion(init_quats[4:]).normalised.inverse.elements
            # ITERATE THROUGH ALL SAMPLES IN THIS WALKING LAP
            for i in range(init_idxs[idx], init_idxs[idx + 1]):

                # Get this quaternion rotation and vectors
                quats = data[i, :]
                #if is_inverted: quats[4:] = Quaternion(data[i, 4:]).normalised.inverse.elements

                thigh_quat, femur_quat = self.quat2KneeHipShank_initialize_quats(quats, init_quats)
                thigh_vec = thigh_quat.rotate(null_vec)
                shank_vec = femur_quat.rotate(null_vec)

                #knee = self.quat2KneeHipShank_AcuteAngBetweenV(thigh_vec, shank_vec)
                knee = self.quat2KneeHipShank_angBetweenV(thigh_vec, shank_vec,shank_vec)
                hip = self.quat2KneeHipShank_angBetweenV(thigh_vec, shank_vec,null_vec)
                shank = self.quat2KneeHipShank_angBetweenV(shank_vec,thigh_vec,null_vec)
                #knee,hip,shank = self.quat2KneeHipShank_vec2KHSangle(thigh_vec, shank_vec)

                hip_angles.append(hip)
                shank_angles.append(shank)
                knee_angles.append(knee)

                #printh(3, f'Iteration Size = {len(hip_angles)}\t Hip = {hip}')
                #sleep(0.5)

            printh(1, f'Lap {idx+1}')
            printh(2, f'Idx Range = [{init_idxs[idx]},{init_idxs[idx + 1]}]')
            printh(2, f'Expected Lap Size = {init_idxs[idx + 1]-init_idxs[idx]}')
            printh(2, f'Cumulative Size = {len(hip_angles)}')

        printh(0, 'Final Results')
        printh(1, f'Num Processed:\t n={len(knee_angles)}')
        printh(1, f'Num Samples:\t n={len(data)}')
        printh(1, f'Sample Angle:\t {hip}')

        ##################################
        # FILTER
        ##################################
        if filtered:

            knee_angles = Data_Filters.lfilter(knee_angles, filter_iter,denom=filt_denom)
            hip_angles = Data_Filters.lfilter(hip_angles, filter_iter,denom=filt_denom)
            shank_angles = Data_Filters.lfilter(shank_angles, filter_iter,denom=filt_denom)

        ##################################
        # PLOT
        ##################################
        if plot_output:
            # Plotting figures
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax_kneeAngles = fig.add_subplot(3, 1, 1)
            ax_hipAngles = fig.add_subplot(3, 1, 2)
            ax_shankAngles = fig.add_subplot(3, 1, 3)

            ax_kneeAngles.set_title(f'Knee Angles | Filtered:{filtered}')
            ax_hipAngles.set_title(f'Hip Angles | Filtered:{filtered}')
            ax_shankAngles.set_title(f'Shank Angles | Filtered:{filtered}')

            ax_kneeAngles.vlines(init_idxs, min(knee_angles), max(knee_angles), colors='r', linestyles=':')
            ax_hipAngles.vlines(init_idxs, min(hip_angles), max(hip_angles), colors='r', linestyles=':')
            ax_shankAngles.vlines(init_idxs, min(shank_angles), max(shank_angles), colors='r', linestyles=':')

            if plot_type=='line':
                ax_hipAngles.plot(hip_angles,linewidth=1)
                ax_shankAngles.plot(shank_angles,linewidth=1)
                ax_kneeAngles.plot(knee_angles,linewidth=1)
            else:
                ax_kneeAngles.scatter(np.arange(0, len(knee_angles)), knee_angles, s=1)
                ax_hipAngles.scatter(np.arange(0, len(hip_angles)), hip_angles, s=1)
                ax_shankAngles.scatter(np.arange(0, len(shank_angles)), shank_angles, s=1)

            ax_kneeAngles.plot([0 for pt in range(len(knee_angles))], c='k', linewidth=1)
            ax_hipAngles.plot([0 for pt in range(len(hip_angles))], c='k', linewidth=1)
            ax_shankAngles.plot([0 for pt in range(len(shank_angles))], c='k', linewidth=1)

            plt.subplots_adjust(right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.5)

        ##################################
        # PACKAGE
        ##################################
        knee_angles = np.reshape(knee_angles, (-1, 1))
        hip_angles = np.reshape(hip_angles, (-1, 1))
        shank_angles = np.reshape(shank_angles, (-1, 1))

        KH = np.append(knee_angles, hip_angles, axis=1)
        KHP = np.append(KH, shank_angles, axis=1)

        if save_output:
            from datetime import datetime
            time_begin = datetime.now()  # current date and time
            date = time_begin.strftime("%m_%d_%Y")
            if name_prefix != '': name_prefix = name_prefix + '_'
            name = f'{self.output_dir}\\{date}_{name_prefix}Quat2KneeHipShank'
            savetxt(name + '.csv', KHP, delimiter=',')
            if plot_output: plt.savefig(name + ".png", dpi=600)
            print('Saved Output:\n', name)

        if plot_output:
            print('Plotting output. Exit plot to continue...')
            plt.show()

        return KHP

    def quat2KneeHipShank_initialize_quats(self,quats, init_quats, start_displace=0):

        thigh_quat = quats[0:4]
        femur_quat = quats[4:8]
        thigh_quat = Quaternion(thigh_quat).normalised
        femur_quat = Quaternion(femur_quat).normalised

        thigh_init_quat = init_quats[0:4]
        femur_init_quat = init_quats[4:8]
        thigh_init = self.quat2KneeHipShank_rel_rot(Quaternion(thigh_init_quat))
        femur_init = self.quat2KneeHipShank_rel_rot(Quaternion(femur_init_quat))

        init_displace_quat_thigh = Quaternion(axis=thigh_quat.axis, degrees=start_displace / 2.)
        init_displace_quat_femur = Quaternion(axis=femur_quat.axis, degrees=start_displace / 2.)

        thigh_quat = thigh_init * thigh_quat
        thigh_quat = init_displace_quat_thigh * thigh_quat

        femur_quat = femur_init * femur_quat
        thigh_quat = -init_displace_quat_femur * thigh_quat

        return thigh_quat, femur_quat

    def quat2KneeHipShank_vec2KHSangle(self,V_thigh, V_shank):
        """
        Calculate the angle between vectors about the axis normal to the plane formed by leg
        (except for knee which is minimum angle)
        """

        uv_thigh = V_thigh / np.linalg.norm(np.array(V_thigh))
        uv_shank = V_shank / np.linalg.norm(np.array(V_shank))
        uv_null = np.array([1, 0, 0])

        # Calculate normal vector formed by leg (ei assumed saggital plane)
        hip = np.array([0, 0, 0])
        knee = hip + uv_thigh
        ankle = uv_thigh + uv_shank
        cross =np.cross((knee - hip), (ankle - hip))
        uv_norm = cross / np.linalg.norm(np.array(cross))

        # calculate hip angle
        vecs = [uv_thigh, uv_null]
        dot_product = np.dot(vecs[0], vecs[1])
        det = np.linalg.det(np.append(np.append(vecs[0], vecs[1],axis=0),uv_norm,axis=0).reshape(3,3))
        hip_angle = degrees(np.arctan2(det, dot_product))

        # calculate shank angle
        vecs = [uv_shank, uv_null]
        dot_product = np.dot(vecs[0], vecs[1])
        det = np.linalg.det(np.append(np.append(vecs[0], vecs[1],axis=0),uv_norm,axis=0).reshape(3,3))
        shank_angle = degrees(np.arctan2(det, dot_product))

        # calculate knee angle
        vecs = [uv_thigh, uv_shank]
        dot_product = np.dot(vecs[0], vecs[1])
        det = np.linalg.det(np.append(np.append(vecs[0], vecs[1], axis=0), uv_norm, axis=0).reshape(3, 3))
        knee_angle = degrees(np.arctan2(det, dot_product))

        return knee_angle,hip_angle,shank_angle

    def quat2KneeHipShank_angBetweenV(self,V1, V2, Vref):
        # error handling
        uv1 = V1 / np.linalg.norm(np.array(V1))
        uv2 = V2 / np.linalg.norm(np.array(V2))
        uvref = Vref / np.linalg.norm(np.array(Vref))

        #v1= np.array(v1)
        #v2 = np.array(v2)
        #vref = np.array(vref)

        # Calculate normal vector formed by leg (ei assumed saggital plane)
        hip = np.array([0, 0, 0])
        knee = hip + uv1
        ankle = uv1 + uv2
        cross =np.cross((knee - hip), (ankle - hip))
        uv_norm = cross / np.linalg.norm(np.array(cross))

        # calculate angel
        uv1 = uv1/ np.linalg.norm(uv1)
        uv2 = uvref/ np.linalg.norm(uvref)
        dot_product = np.dot(uv1, uv2)
        det = np.linalg.det(np.append(np.append(uv1,uv2,axis=0),uv_norm,axis=0).reshape(3,3))
        angle = degrees(np.arctan2(det, dot_product))

        return angle


    def quat2KneeHipShank_angBetweenV_old(self,v1, v2):
        # check if there is an array of zeros at hip

        # calculate angel
        unit_vector_1 = v1 / np.linalg.norm(v1)
        unit_vector_2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = degrees(np.arccos(dot_product))
        return angle

    def quat2KneeHipShank_rel_rot(self,quat, target_quat=Quaternion([1., 0., 0., 0.])):
        quat_init = target_quat * quat.inverse
        quat_init = quat_init.normalised
        return quat_init

