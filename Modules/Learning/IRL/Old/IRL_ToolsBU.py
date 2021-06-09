import numpy as np

class IRL_tools:
    def __init__(self,X_train,y_train,rand_move,N_STATES=5,N_ACTIONS=3):
        self.X_train = X_train
        self.y_train = y_train
        self.wind = rand_move
        self.n_features = np.shape(X_train)[1]

        self.N_STATES = N_STATES    # constant user input
        self.N_ACTIONS = N_ACTIONS  # constant user input

        self.n_actions = N_ACTIONS                       # programming var
        self.states_per_feature =[N_STATES for s in range(np.shape(X_train)[1])]     # programming var
        self.n_states = 1                                # programming var
        for n in self.states_per_feature:                # programming var
            self.n_states = self.n_states * n
        self.feature_matrix = np.eye((self.n_states))  # (400, 400)

        # Calculate initial state
        self.data = self.data_per_gaitphase(X_train, y_train)

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

        self.idx_demo = self.get_idx_demo(X_train, y_train)
        self.transition_probability=self.transitional_probabilities()

        self.reward = 'Not Saved'
        self.policy = 'Not Saved'
        self.save_dir = "/Example_Scripts/GeneratedModels\\"

        self.state_scales = 1.01*(np.array(self.max_states) - np.array(self.min_states)) / (self.states_per_feature)
        self.action_scale = 1.01 * (np.array(self.max_action) - np.array(self.min_action)) /self.n_states

        #print(self.idx_demo.shape)
        print(f'\nInitializing Feature Space...')
        print(f"|\t Shape of raw feature data (Train|Test):", np.shape(X_train))
        print(f"|\t Shape of raw label data (Train|Test):", np.shape(y_train))
        print(f"|\t ")
        print(f'|\t n_states:{self.n_states}')
        print(f'|\t max_states:{self.max_states}')
        print(f'|\t min_states:{self.min_states}')
        print(f'|\t n_actions:{self.n_actions}')
        print(f'|\t max_action:{self.max_action}')
        print(f'|\t min_action:{self.min_action}')
        print(f'|\t torque_actions:{self.actions}')
        print(f"|\t Shape of demonstrations:", np.shape(self.data))
        print(f'|\t Trajectories Shape {np.shape(self.idx_demo)}')
        print(f"|\t ")
        print(f'|\t Trajectories Samples')
        print(f'||\t (demosntration, length of demo, [state index,action])')
        print(f'||\t Trajectories[0,0,:] {self.idx_demo[0,0,:]}')
        print(f'||\t Trajectories[0,1,:] {self.idx_demo[0, 1, :]}')
        print(f'||\t Trajectories[1,0,:] {self.idx_demo[1, 0, :]}')
        print(f'||\t Trajectories[1,1,:] {self.idx_demo[1, 1, :]}')
        print(f'||\t Max State {self.idx_demo[:, :, 0].max()+1}')
        print(f'||\t Max Action {self.idx_demo[:, :, 1].max()+1}')



    def get_idx_demo(self,features,labels,VERBOSE=False): ################### State indexs might be bugged
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
                idx = self.state2int(feature,self.max_states,self.min_states,self.states_per_feature)
                phase_features_idx.append(idx)
            phase_features_idx = np.reshape(phase_features_idx, (-1, 1))

            # Convert action to integer index
            phase_labels_idx = []
            phase_labels = labels[HS[hs]:HS[hs + 1], :]
            for label in phase_labels:
                idx = self.state2int(label, self.max_action, self.min_action, [self.n_actions])
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

    def state2int(self,state,max_states,min_states,states_per_feature,VERBOSE=False):
        if np.size(min_states)==1: min_states=[min_states]
        if np.size(max_states) == 1: max_states = [max_states]
        if VERBOSE: print('IRL_TOOLS.IDX_STATE...')
        rel_idxs = []
        #scale = 1.01*(np.array(max_states) - np.array(min_states)) / (states_per_feature)
        scale =self.state_scales
        for s in range(len(state)):
            if VERBOSE:
                print(f'Len(state) = {len(state)}')
                print(f'Scale={scale}')
                print(f'Calc = {state[s]}')
                print(f'Calc = {min_states[s]}')
                print(f'Calc = {scale[s]}')
                print(f'Calc = {(state[s]-min_states[s]) / scale[s]}')
            rel_idxs.append(int((state[s]-min_states[s]) / scale[s]))

        #spf=[s for s in states_per_feature]
        spf = np.ones((self.n_features,))*self.N_STATES
        spf[-1]= 1
        state_idx = 0
        for i in range(len(rel_idxs)):
            state_idx += (rel_idxs[i])*spf[i]
        if VERBOSE:
            print(f'|\t rel_idxs = {rel_idxs}')
            print(f'|\t Spf = {states_per_feature}')
            print(f'|\t max_states={max_states}\n min_states={min_states}')
            print('|\t State,Index',state,state_idx,'\n')
        return int(state_idx)

    def int2state(self,state_int,max_states,min_states,states_per_feature):
        scale = 1.01 * (np.array(max_states) - np.array(min_states)) / (states_per_feature)

    def action2int(self,action,VERBOSE=False):

        if VERBOSE: print('IRL_TOOLS.IDX_STATE...')
        rel_idxs = []
        scale = self.action_scale

        for s in range(len(action)):
            if VERBOSE:
                print(f'Len(state) = {len(action)}')
                print(f'Scale={scale}')
                print(f'Calc = {action}')
                print(f'Calc = {self.min_action}')
                print(f'Calc = {scale[s]}')
                print(f'Calc = {(action-self.min_action) / scale}')
            rel_idxs.append(int(action-self.min_action) / scale)

        action_idx = 0
        for i in range(len(rel_idxs)):
            action_idx += (rel_idxs[i])*self.n_actions
        if VERBOSE:
            print(f'|\t rel_idxs = {rel_idxs}')
            print(f'|\t N_ACTIONS = {self.N_ACTIONS}')
            print(f'|\t max_states={self.max_action}\n min_states={self.min_action}')
            print('|\t State,Index',action,action_idx,'\n')
        return int(action_idx)
    def int2action(self,action_int,VERBOSE=False):

        if VERBOSE: print('IRL_TOOLS.IDX_STATE...')
        rel_idxs = []
        scale = self.action_scale

        return int(action_idx)

    def data_per_gaitphase(self,features,labels):
        # Get heel strike indexes
        if np.shape(labels[1])==():
            labels = np.reshape(labels,(-1,1))

        HS = []
        #gait_phase = self.train_X[:, 0]
        gait_phase = features[:, 0]
        for phase in range(len(gait_phase) - 1):
            if gait_phase[phase + 1] < gait_phase[phase]: HS.append(phase + 1)

        features_per_gaitphase=[]
        for hs in range(len(HS) - 1):
            phase_features = features[HS[hs]:HS[hs + 1],:]
            phase_labels = labels[HS[hs]:HS[hs + 1], :]
            current_phase= np.append(phase_features,phase_labels,axis=1)
            features_per_gaitphase.append(current_phase)

        features_per_gaitphase=np.array(features_per_gaitphase)
        return features_per_gaitphase

    def resample_nbins(self,data,n_bins):
        for j in range(len(data[0, :])):
            bins = np.linspace(np.min(data[:,j]), np.max(data[:,j]), n_bins)
            for i in range(len(data[:,0])):
                K = data[i,j]
                closest = bins[min(range(len(bins)), key=lambda i: abs(bins[i] - K))]
                data[i,j]=closest

        return data


    def transitional_probabilities(self,VERBOSE=False):

        print(f'\nCalculating Transitional Probabilities...')

        print(f'|\t Observing transitional probabilities')
        trans_probs = np.zeros((self.n_states, self.n_actions, self.n_states))
        print(f'Demo {np.shape(self.idx_demo)}')
        print(f'Trajectory {np.shape(self.idx_demo[0])[0]}')
        for trajectory in self.idx_demo:
            for i in range(np.shape(trajectory)[0]-1):

                state_i = trajectory[i,0]
                action_j = trajectory[i,1]
                state_k = trajectory[i+1,0]
                if VERBOSE:
                    print(f'state_i" {state_i}')
                    print(f'action_j" {action_j}')
                    print(f'state_k" {state_k}')

                trans_probs[state_i,action_j,state_k] +=1

        trans_probs=trans_probs/trans_probs.sum()

        return trans_probs


    def feature2index(self,search_array,feature):
        feature = np.reshape(feature,(1,-1))
        #return np.where(np.all(search_array==feature,axis=1))[0]
        return np.where((search_array == feature).all(axis=1))[0]


    def reward_output(self,reward):
        #np.array(reward).reshape(self.N_STATES,self.N_STATES)
        print(f'\nDiplaying recovered reward...')
        print(f'|\t Reward shape = {np.shape(reward)}')
        print(f'|\t Max reward = {reward.max()}')
        print(f'|\t Max reward = {reward.min()}')
        print(f'|\t Reward Values =\n {reward.min()}')
        print(reward.reshape((self.N_STATES, self.N_STATES)))

    def reward_plot(self,reward):
        import matplotlib.pyplot as plt
        print(f'Plotting...')
        print(f'|\t exit to save reward')

        plt.subplot(1, 2, 1)
        reward_map = reward.reshape((self.N_STATES, self.N_STATES))
        plt.pcolor(reward_map)
        plt.colorbar()
        plt.title("Recovered reward")
        plt.show()

    def reward_save(self,reward,path,SAVE=True):
        if SAVE:
            print(f'Saving reward map...')
            print(f'|\t Path: {path}')
            np.save(path, reward)
        else:
            print(f'Model was not saved...')

    def plot_test(self,policy,X_test,y_test,VERBOSE=True):
        import matplotlib.pyplot as plt
        print(f'Plot Test Data...')
        idx_test = self.get_idx_demo(X_test, y_test)

        obs_states = []
        pred_action_d = []
        pred_action_w =[]
        for trajectory in idx_test:
            for i in range(np.shape(trajectory)[0]):
                state = trajectory[i,0]
                #if VERBOSE: print(f'|\t State: {state}')
                obs_states.append(state)
        for obs in obs_states:
            #pred_action.append(np.argmax(policy[obs, :]))
            prob_of_action = policy[obs, :]

            pred_action_discrete = np.argmax(prob_of_action)

            pred_action_weighted = 0
            for a in range(len(prob_of_action)):
                pred_action_weighted += prob_of_action[a]*a

            pred_action_d.append(pred_action_discrete)
            pred_action_w.append(pred_action_weighted)

        scale = 1.01*(self.max_action - self.min_action) / self.n_actions
        pred_torque_d =np.array(pred_action_d)*scale+self.min_action
        pred_torque_w = np.array(pred_action_w) * scale + self.min_action

        if VERBOSE:
            print(f'|\t Data Shape {np.shape(y_test)}')
            print(f'|\t idx Shape {np.shape(idx_test)}')
            print(f'|\t pred Shape {np.shape(pred_torque_d)}')
            print(f'|\t max action {self.max_action}')
            print(f'|\t min action {self.min_action}')
            print(f'|\t Scale {scale}')


        plt.subplot()
        plt.plot(y_test,label='Observed')
        plt.plot(pred_torque_d,label='Discrete')
        plt.plot(pred_torque_w, label='Weighted')
        plt.title("Testing Policy")
        plt.show()

    def save_obj(self,model_name,reward,policy,ENABLE=True,VERBOSE=True):
        if ENABLE:
            import pickle
            if VERBOSE: print("Saving Model as:\n|\t ", model_name)
            self.reward=reward
            self.policy=policy

            path = self.save_dir +model_name+'_obj'
            with open(path, 'wb') as obj_file:
                pickle.dump(self, obj_file)

            return 'File Saved'
        else:
            return 'Saving not enabled'


def load_obj(model_name,model_dir = 'C:\\Users\\mason\Desktop\\Thesis\\ML_MasonSmithGit\Example_Scripts\\GeneratedModels\\IRL\\'):
    import pickle
    path = model_dir + model_name
    with open(path, 'rb') as IRL_obj:
        IRL = pickle.load(IRL_obj)
    return IRL