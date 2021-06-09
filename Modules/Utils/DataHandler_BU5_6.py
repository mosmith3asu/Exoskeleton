import os
import numpy as np
from numpy import genfromtxt,savetxt
from pyquaternion import Quaternion
from math import degrees
from Modules.Utils.Data_Filters import lfilter
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

    def quat2KneeHipShank(self, quat_csv_path, init_idxs=[0], first_quat='Thigh',
                          name_prefix='', plot_output=True, save_output=True,
                          filtered=True, filter_iter=7):

        from Modules.Utils import Data_Filters
        import matplotlib.pyplot as plt

        print('\nBegining [Quat] to [Knee,Hip,Shank] transformation...')
        knee_angles = []
        hip_angles = []
        shank_angles = []
        null_vec = [1, 0, 0]
        # Import Quat data
        data = genfromtxt(quat_csv_path, delimiter=',')

        null_vec = [1, 0, 0]
        knee_angles_raw = []
        hip_angles_raw = []
        shank_angles_raw = []
        for i in range(len(data[:, 0])):
            quats = data[i, :]

            thigh_quat_raw, femur_quat_raw = Quaternion(quats[:4]), Quaternion(quats[4:])
            v1_raw = thigh_quat_raw.rotate(null_vec)
            v2_raw = femur_quat_raw.rotate(null_vec)

            angle_raw = self.quat2KneeHipShank_angBetweenV(v1_raw, v2_raw)  # + start_displace
            hip_raw = self.quat2KneeHipShank_angBetweenV(v1_raw, null_vec)
            shank_raw = self.quat2KneeHipShank_angBetweenV(v2_raw, null_vec)

            knee_angles_raw.append(angle_raw)
            hip_angles_raw.append(hip_raw)
            shank_angles_raw.append(shank_raw)

        # Package RAW data
        knee_angles_raw = np.reshape(knee_angles_raw, (-1, 1))
        hip_angles_raw = np.reshape(hip_angles_raw, (-1, 1))
        shank_angles_raw = np.reshape(shank_angles_raw, (-1, 1))
        KH_raw = np.append(knee_angles_raw, hip_angles_raw, axis=1)
        KHP_raw = np.append(KH_raw, shank_angles_raw, axis=1)

        # Quaternion for 0 knee displacement
        init_quats = np.array([[]])
        for idx in init_idxs:
            if first_quat == 'Thigh' or first_quat == 'thigh':
                thigh_init = data[idx, :4]
                femur_init = data[idx, 4:]
            elif first_quat == 'Shank' or first_quat == 'shank':
                thigh_init = data[idx, 4:]
                femur_init = data[idx, :4]
            else:
                print(f'ERROR: "first_quat={first_quat}" is not a valid entry\n please enter Thigh or Shank ')

            init_quat = np.append(thigh_init, femur_init)
            init_quat = np.append(idx, init_quat).reshape((1, 9))

            if init_quats.size < 1:
                init_quats = init_quat
            else:
                init_quats = np.append(init_quats, init_quat, axis=0)

        num_idxs = len(init_idxs)
        for i in range(len(data[:, 0])):
            idx = 0
            if num_idxs > 1:
                while i >= init_idxs[idx + 1] and idx < num_idxs - 1:
                    if idx == num_idxs - 2:
                        idx = idx + 1
                        break
                    else:
                        idx = idx + 1

            init_quat = init_quats[idx, 1:]
            quats = data[i, :]
            thigh_quat, femur_quat = self.quat2KneeHipShank_initialize_quats(quats, init_quat)
            v1 = thigh_quat.rotate(null_vec)
            v2 = femur_quat.rotate(null_vec)

            angle = self.quat2KneeHipShank_angBetweenV(v1, v2)  # + start_displace
            hip = self.quat2KneeHipShank_angBetweenV(v1, null_vec)
            shank = self.quat2KneeHipShank_angBetweenV(v2, null_vec)

            hip_angles.append(hip)
            shank_angles.append(shank)
            knee_angles.append(angle)

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

            ax_hipAngles.plot(hip_angles)
            ax_shankAngles.plot(shank_angles)
            ax_kneeAngles.plot(knee_angles)

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

        return KHP, KHP_raw

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

    def quat2KneeHipShank_angBetweenV(self,v1, v2):
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

