import numpy as np
from numpy import genfromtxt,savetxt

def main():
    data_names1 = [
        # 'trimmed_train_P3S2'
        'trimmed_train_P3S3'
    ]
    data_names2 = [
        # 'trimmed_test_P3S2'
        'trimmed_test_P3S3'
    ]

    n_states = 7
    n_actions = 20

    ETP = EstimateTransitionalProbabilities(data_names1,n_states,n_actions)
    print('Initial')
    print(f'Shape: {np.shape(ETP.transition_probability)}')
    print(f'Sample: {ETP.transition_probability[0,0,0]}')

    ETP.update_estimate(data_names2)
    print('Updated')
    print(f'Shape: {np.shape(ETP.transition_probability)}')
    print(f'Sample: {ETP.transition_probability[0, 0, 0]}')

def load(model_name,model_dir = 'C:\\Users\\mason\Desktop\\Thesis\\ML_MasonSmithGit\Example_Scripts\\GeneratedModels\\IRL\\'):
    import pickle
    path = model_dir + model_name
    with open(path, 'rb') as IRL_obj:
        IRL = pickle.load(IRL_obj)
    return IRL

class EstimateTransitionalProbabilities:
    def __init__(self,data_names,N_STATES,N_ACTIONS):
        self.transition_probability = []
        self.transition_count = []
        self.n_features = 3

        # Initialize directories
        self.save_dir = "C:\\Users\\mason\\Desktop\\Thesis\\ML_MasonSmithGit\\Example_Scripts\\GeneratedModels\\"
        self.data_root_dir = 'C:\\Users\\mason\\Desktop\\Thesis\\Patient_files'
        self.data_processed_dir = self.data_root_dir + '\\ProcessedData'

        # Initialize Import Vars


        self.datanames_in_use = []
        self.X_train_seperated = []
        self.y_train_seperated = []

        self.import_GaitPhase = True
        self.import_KneeAngle = False
        self.import_HipAngle = True
        self.import_ShankAngle = True
        self.import_GRF = False
        self.index_GaitPhase = 0
        self.index_KneeAngle = 1
        self.index_HipAngle = 2
        self.index_ShankAngle = 3
        self.index_GRF = np.arange(4, 8)
        self.name_lst = np.array([['Gait Phase', 'Knee Angle', 'Hip Angle', 'Shank Angle',
                                   'GRF1', 'GRF2', 'GRF3', 'GRF4']])
        if type(data_names) != list:
            data_names = [data_names]
        X, y = self.import_data(data_names[0])
        self.X_train = np.empty((0, self.n_features))
        self.y_train = np.empty((0, 1))

        # Import initial training data
        X_train,y_train = self.update_training_data(data_names)

        # Initialize actions and states
        self.N_ACTIONS = N_ACTIONS  # constant user input
        self.n_actions = N_ACTIONS  # programming var
        self.N_STATES = N_STATES  # constant user input
        self.states_per_feature = [N_STATES for s in range(np.shape(self.X_train)[1])]  # programming var
        self.n_states = 1  # programming var
        for n in self.states_per_feature:  # programming var
            self.n_states = self.n_states * n
        self.feature_matrix = np.eye((self.n_states))  # (400, 400)

        # Calculate initial state
        self.data = self.data_per_gaitphase(self.X_train, self.y_train)

        # Calculate max/mins of feature space
        self.max_states = []
        self.min_states = []
        for f in range(len(self.data[0, 0, :]) - 1):
            self.max_states.append(self.data[:, :, f].max())
            self.min_states.append(self.data[:, :, f].min())
        self.max_action = self.data[:, :, -1].max()
        self.min_action = self.data[:, :, -1].min()
        self.actions = []
        for a in range(self.n_actions):
            self.actions.append(self.min_action + a * (self.max_action - self.min_action) / (self.n_actions - 1))

        # Calculate scale for discretization
        self.eps = 1.00000000001
        self.state_scales = self.eps * (np.array(self.max_states) - np.array(self.min_states)) / self.N_STATES
        self.action_scale = self.eps * (np.array(self.max_action) - np.array(self.min_action)) / self.n_actions

        # Get discrete action and state space
        self.idx_demo = self.get_idx_demo(self.X_train, self.y_train)

        # Calculate transitional probability and count matrix
        self.transition_count = np.ones((self.n_states, self.n_actions, self.n_states))
        count,prob= self.transition_estimate()
        self.transition_count = count
        self.transition_probability = prob


    def update_training_data(self,data_names):
        print("Transitional Probabilities...")
        print("Importing New Data...")
        print(f"|\t Current Data Names {self.datanames_in_use}")

        # Type check
        if type(data_names) != list:
            data_names = [data_names]

        # Import data
        for name in data_names:
            print(f'|\t New Data Name: {name}')
            X, y = self.import_data(name)
            self.datanames_in_use.append(name)
            self.X_train = np.append(self.X_train, X, axis=0)
            self.y_train = np.append(self.y_train, y, axis=0)

        return self.X_train,self.y_train

    def update_estimate(self, data_names):
        """Update Training Data and recalculate everything from scratch"""
        print("Updating Estimate Count...")
        self.update_training_data(data_names)
        self.idx_demo = self.get_idx_demo(self.X_train, self.y_train)
        count, prob = self.estimate(self.X_train, self.y_train)
        self.transition_count = count
        self.transition_probability = prob
        return self.transition_count,self.transition_probability


    def transition_estimate(self,VERBOSE=False):
        """ Tranistional Prob/Counts have shape (State_i,Action_j,State_k)"""

        print(f'\nCalculating Transitional Probabilities...')
        print(f'|\t Observing transitional counts...')
        print(f'|\t idx Demo Shape {np.shape(self.idx_demo)}')
        print(f'|\t Trans Probs Shape {np.shape(self.transition_count)}')
        print(f'|\t Trajectory {np.shape(self.idx_demo[0])}')

        # Calculate counts for next state_k given each action_j and state_i
        transition_count = np.ones((self.n_states, self.n_actions, self.n_states))
        for trajectory in self.idx_demo:
            for i in range(np.shape(trajectory)[0] - 1):
                state_i = trajectory[i, 0]
                action_j = trajectory[i, 1]
                state_k = trajectory[i + 1, 0]
                if VERBOSE: print(f'state_i" {state_i} action_j" {action_j} state_k" {state_k}')
                transition_count[state_i, action_j, state_k] += 1

        # Calculate probabilities for next state_k given each action_j and state_i
        trans_probs = self.transition_count
        for i in np.shape(trans_probs,0):
            for j in np.shape(trans_probs, 1):
                trans_probs[i,j,:] = trans_probs[i,j,:] / trans_probs[i,j,:].sum()

        return transition_count,trans_probs

    def import_data(self, name):

        path = self.data_processed_dir + '\\' + name + '.csv'
        import_data = genfromtxt(path, delimiter=',')[:, 0:np.size(self.name_lst) + 1]
        import_data = import_data[1:, :]
        X = import_data[:, 1:]
        Y = import_data[:, 0].reshape(-1, 1)

        if 'train' in name: import_type = 'Train'
        elif 'test' in name: import_type = 'Test'
        else: import_type = 'Unspecified'

        X, headers = self.filter_excluded_features(X)

        print(f'DataHandler.import_custom Report... ')
        print(f'|\t File: \t {name}')
        print(f'|\t Imported {import_type} Data: \t {headers}')
        print("|\t Shape of X|Y: \t", np.shape(X), '|', np.shape(Y))
        return X, Y

    def filter_excluded_features(self, data):

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

        if np.size(del_lst) > 0:
            data = np.delete(data, del_lst, 1)
            headers = np.delete(headers, del_lst, 1)

        self.import_enables = [self.import_GaitPhase, self.import_KneeAngle, self.import_HipAngle,
                               self.import_ShankAngle, self.import_GRF]

        return data, headers

    def data_per_gaitphase(self,features,labels):
        # Get heel strike indexes
        if np.shape(labels[1])==():
            labels = np.reshape(labels,(-1,1))

        HS = [0]
        #gait_phase = self.train_X[:, 0]
        gait_phase = features[:, 0]
        for phase in range(len(gait_phase) - 1):
            if gait_phase[phase + 1] < gait_phase[phase]: HS.append(phase + 1)
        HS.append(np.size(labels))

        features_per_gaitphase=[]
        for hs in range(len(HS) - 1):
            phase_features = features[HS[hs]:HS[hs + 1],:]
            phase_labels = labels[HS[hs]:HS[hs + 1], :]
            current_phase= np.append(phase_features,phase_labels,axis=1)
            features_per_gaitphase.append(current_phase)

        features_per_gaitphase=np.array(features_per_gaitphase)
        return features_per_gaitphase

    def get_idx_demo(self,features,labels,VERBOSE=True): ################### State indexs might be bugged
        if VERBOSE: print(f'DataHandler.idx_demo...')
        # Get heel strike indexes
        if np.shape(labels[1]) == ():
            labels = np.reshape(labels, (-1, 1))

        HS = [0]
        # gait_phase = self.train_X[:, 0]
        gait_phase = features[:, 0]
        for phase in range(len(gait_phase) - 1):
            if gait_phase[phase + 1] < gait_phase[phase]: HS.append(phase + 1)
        HS.append(np.size(labels))

        features_per_gaitphase = []
        for hs in range(len(HS) - 1):

            # Convert features to integer state index
            phase_features_idx = []
            phase_features = features[HS[hs]:HS[hs + 1], :]
            for feature in phase_features:
                idx = self.state2int(feature)
                phase_features_idx.append(idx)
            phase_features_idx = np.reshape(phase_features_idx, (-1, 1))

            # Convert action to integer index
            phase_labels_idx = []
            phase_labels = labels[HS[hs]:HS[hs + 1], :]
            for label in phase_labels:
                idx = self.action2int(label)
                phase_labels_idx.append(idx)
            phase_labels_idx = np.reshape(phase_labels_idx, (-1, 1))


            if VERBOSE:
                print(f'phase idx features: {np.shape(phase_features_idx)}')
                print(f'phase idx labels: {np.shape(phase_labels_idx)}')
            current_phase = np.append(phase_features_idx, phase_labels_idx, axis=1)
            features_per_gaitphase.append(current_phase)

        features_per_gaitphase = np.array(features_per_gaitphase)

        if VERBOSE: print(f'|\t Trajectory sample = {features_per_gaitphase[0,0,:]}')
        return features_per_gaitphase

    def state2int(self, state, VERBOSE=False):

        if VERBOSE: print('\nIRL_TOOLS.IDX_STATE...')
        rel_idxs = []
        # scale = 1.01*(np.array(max_states) - np.array(min_states)) / (states_per_feature)
        scale = self.state_scales
        if VERBOSE:
            print(f'|\t Len(state) = {len(state)}')
            print(f'|\t Scale={scale}')
            print(f'|\t N_STATES = {self.N_STATES}')
            print(f'|\t Min_states = {self.min_states}')
            print(f'|\t Max_states = {self.max_states}')
        for s in range(len(state)):
            if VERBOSE:
                print(f'|\t Calc = {(state[s] - self.min_states[s]) / scale[s]}')
            rel_idxs.append(int((state[s] - self.min_states[s]) / scale[s]))

        spf = [self.N_STATES ** (self.n_features - s - 1) for s in range(self.n_features)]

        state_idx = 0
        for i in range(self.n_features):
            multiplier = spf[i]
            state_idx += (rel_idxs[i]) * multiplier
        if VERBOSE:
            print(f'|\t rel_idxs = {rel_idxs}')
            print(f'|\t Spf = {spf}')
            # print(f'|\t max_states={self.max_states}')
            # print(f'|\t min_states={self.min_states}')
            print('|\t State,Index', state, state_idx, '\n')
        return int(state_idx)

    def action2int(self,action,VERBOSE=False):

        if VERBOSE: print('IRL_TOOLS.action2int...')

        scale = self.action_scale
        if VERBOSE:
            print(f'|\t Action = {action}')
            print(f'|\t Scale={scale}')
            print(f'|\t N_ACTIONS = {self.N_ACTIONS}')
            print(f'|\t Min_action = {self.min_action}')
            print(f'|\t Max_action = {self.max_action}')

        action_idx=(int((action-self.min_action) / scale))
        if VERBOSE:
            print(f'|\t Action {action},Index {action_idx}')
        return int(action_idx)

    def save(self,ENABLE=True,VERBOSE=True):
        model_name = self.OBJ_NAME
        if ENABLE:
            import pickle
            if VERBOSE: print("Saving Model as:\n|\t ", model_name)
            path = self.save_dir +model_name+'_obj'
            with open(path, 'wb') as obj_file:
                pickle.dump(self, obj_file)
            return '|\t File Saved'
        else:
            return 'Saving not enabled'

if __name__ == "__main__":
    main()