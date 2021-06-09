import numpy as np

class IRL_tools:
    def __init__(self,X_train,y_train,trans_prob_names = "" ,
                 N_STATES=5,N_ACTIONS=3,
                 notes = {},USE_SUBPHASES=True):

        self.USE_SUBPHASES = USE_SUBPHASES
        self.subphases = {'durations': [[0, 5],
                                       [5, 19],
                                        [19, 50],
                                        [50, 81],
                                        [81, 100]],
                          # 'colors':  ['blue','orange','green','red','purple']
                          'colors': ['white', 'lightgrey', 'white', 'lightgrey', 'white'],
                          'labels': ['IC', 'LR', 'MS', 'TS', 'PS']}

       # if self.USE_SUBPHASES:
            #N_STATES_OLD = N_STATES
            #print(f'\n########## CAUTION ###########')
            #print(f'########## CAUTION ###########')
            #print(f'Using Gait Phase Subphases...')
            #print(f'|\t Changing N_STATES to {len(self.subphases["durations"])} instead of {N_STATES_OLD}')
            #N_STATES=len(self.subphases["durations"])
            #print(f'########## CAUTION ###########')
            #print(f'########## CAUTION ###########\n')

        self.header_level  = 0
        self.unique_states={'GP':[],'Hip':[],'Shank':[]}
        self.notes= notes # used for specific notes used during training
        SAMPLE_TRAJECTORIES = False
        self.reward = 'Not Saved'
        self.policy = 'Not Saved'
        self.save_dir = "C:\\Users\\mason\\Desktop\\Thesis\\ML_MasonSmithGit\\Example_Scripts\\GeneratedModels\\"
        self.trans_probs_names = trans_prob_names


        self.X_train = X_train
        self.y_train = y_train
        self.n_features = np.shape(X_train)[1]

        self.N_STATES = N_STATES    # constant user input
        self.N_ACTIONS = N_ACTIONS  # constant user input

        self.n_actions = N_ACTIONS                       # programming var
        self.states_per_feature =[N_STATES for s in range(np.shape(X_train)[1])]     # programming var
        if self.USE_SUBPHASES: self.states_per_feature[0] = len(self.subphases['durations'])
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

        self.eps = 1.00000000001
        self.state_scales = self.eps * (np.array(self.max_states) - np.array(self.min_states)) / self.N_STATES
        self.action_scale =self.eps* (np.array(self.max_action) - np.array(self.min_action)) / self.n_actions

        self.idx_demo = self.get_idx_demo(X_train, y_train)
        self.transition_probability=self.transitional_probabilities()


        #print(self.idx_demo.shape)
        printh(0,f'Initializing Feature Space...')
        printh(1, "")
        printh(1, f"Using Subphases:", self.USE_SUBPHASES)
        printh(1, "")
        printh(1, f'Data...')
        printh(2,f"Shape of raw feature data (Train|Test):", np.shape(X_train))
        printh(2,f"Shape of raw label data (Train|Test):", np.shape(y_train))

        printh(1,"")
        printh(1, f'States...')
        printh(2,f'n_states:{self.n_states}')
        printh(2,f'# GP States {np.size(np.unique(self.unique_states["GP"]))}')
        printh(2,f'# Hip States {np.size(np.unique(self.unique_states["Hip"]))}')
        printh(2,f'# Shank States {np.size(np.unique(self.unique_states["Shank"]))}')
        printh(2,f'max_states:{self.max_states}')
        printh(2,f'min_states:{self.min_states}')

        printh(1,"")
        printh(1, f'Actions...')
        printh(2,f'n_actions:{self.n_actions}')
        printh(2, f'max_action:{self.max_action}')
        printh(2,f'min_action:{self.min_action}')
        printh(2,f'torque_actions:{self.actions}')
        printh(1,f'Shape of demonstrations:", np.shape(self.data)')
        printh(1,f'Trajectories Shape {np.shape(self.idx_demo)}')
        if SAMPLE_TRAJECTORIES:
            print(f"|\t ")
            print(f'|\t Trajectories Samples')
            print(f'||\t (demosntration, length of demo, [state index,action])')
            print(f'||\t Trajectories[0,0,:] {self.idx_demo[0,0,:]}')
            print(f'||\t Trajectories[0,1,:] {self.idx_demo[0, 1, :]}')
            print(f'||\t Trajectories[1,0,:] {self.idx_demo[1, 0, :]}')
            print(f'||\t Trajectories[1,1,:] {self.idx_demo[1, 1, :]}')
            print(f'||\t Max Demo State_idx {self.idx_demo[:, :, 0].max()+1}')
            print(f'||\t Max Demo Action_idx {self.idx_demo[:, :, 1].max()+1}')



    def get_idx_demo(self,features,labels,VERBOSE=False):
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

    def in_what_range(self,value,ranges,VERBOSE=False):
        i=0
        if VERBOSE:
            print(f'|\t in_what_range...')

        for r in ranges:
            if VERBOSE: print(f'|\t|\t {r[0]}<{value}<{r[1]} \t {r[0] <= value <= r[1]}')
            if r[0]<=value <=r[1]: return int(i)
            i += 1
        print('ERROR.in_what_range...')
        print("|\t NOT IN RANGE")
        print(f'|\t Value: {value}')
        print(f'|\t Value: {ranges}')

    def state2int(self,state,VERBOSE=False):

        if VERBOSE: print('\nIRL_TOOLS.IDX_STATE...')
        rel_idxs = []
        #scale = 1.01*(np.array(max_states) - np.array(min_states)) / (states_per_feature)
        scale = self.state_scales
        if VERBOSE:
            print(f'|\t Len(state) = {len(state)}')
            print(f'|\t Scale={scale}')
            print(f'|\t N_STATES = {self.N_STATES}')
            print(f'|\t Min_states = {self.min_states}')
            print(f'|\t Max_states = {self.max_states}')

        if self.USE_SUBPHASES:

            gp_state = state[0]
            gp_idx = self.in_what_range(gp_state, self.subphases['durations'])
            if VERBOSE:
                print(f'|\t Subphase state0 {state[0]}')
                print(f'|\t Subphase state0 index {gp_idx}')


        for s in range(len(state)):
            if VERBOSE:
                print(f'|\t Calc = {(state[s]-self.min_states[s]) / scale[s]}')
            rel_idxs.append(int((state[s]-self.min_states[s]) / scale[s]))

        if self.USE_SUBPHASES: rel_idxs[0]=gp_idx

        if self.USE_SUBPHASES:
            self.unique_states['GP'].append(rel_idxs[0])
            self.unique_states['Hip'].append(rel_idxs[1])
            self.unique_states['Shank'].append(rel_idxs[2])

        spf=[self.N_STATES**(self.n_features-s-1) for s in range(self.n_features)]

        state_idx = 0
        for i in range(self.n_features):
            multiplier = spf[i]
            state_idx += (rel_idxs[i])*multiplier
        if VERBOSE:
            print(f'|\t rel_idxs = {rel_idxs}')
            print(f'|\t Spf = {spf}')
            #print(f'|\t max_states={self.max_states}')
            #print(f'|\t min_states={self.min_states}')
            print('|\t State,Index',state,state_idx,'\n')
        return int(state_idx)

    def get_terminal_states(self):
        spf = [self.N_STATES ** (self.n_features - s - 1) for s in range(self.n_features)]
        gp100 = spf[0]*(self.N_STATES-1)
        terminal_states = np.arange(gp100,self.n_states)
        return terminal_states

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

    def transitional_probabilities(self,VERBOSE=False):

        """ Tranistional Prob/Counts have shape (State_i,Action_j,State_k)"""
        transition_count = np.ones((self.n_states, self.n_actions, self.n_states))
        trans_probs = np.zeros((self.n_states, self.n_actions, self.n_states))

        print(f'\nCalculating Transitional Probabilities...')
        print(f'|\t Mehtod: Observing from demonstrations')

        # Check if additional data for probabilities was defined
        if self.trans_probs_names == "":
            idx_demo = self.idx_demo
            print(f'|\t Using Data: Training Data')
        else:
            from Modules.Utils.DataHandler import Data_Handler
            DataHandler = Data_Handler()
            X_temp = np.empty((0, 3))
            y_temp = np.empty((0, 1))
            for name in self.trans_probs_names:
                X, y = DataHandler.import_custom(name,VERBOSE=VERBOSE)
                X_temp = np.append(X_temp, X, axis=0)
                y_temp = np.append(y_temp, y, axis=0)

            idx_demo = self.get_idx_demo(X_temp, y_temp)
            print(f'|\t Using Data: {self.trans_probs_names}')

        print(f'|\t Trans Probs Shape {np.shape(transition_count)}')
        print(f'|\t Trajectory {np.shape(idx_demo[0])}')

        # Calculate counts for next state_k given each action_j and state_i

        for trajectory in idx_demo:
            for i in range(np.shape(trajectory)[0] - 1):
                state_i = trajectory[i, 0]
                action_j = trajectory[i, 1]
                state_k = trajectory[i + 1, 0]
                if VERBOSE: print(f'state_i" {state_i} action_j" {action_j} state_k" {state_k}')
                transition_count[state_i, action_j, state_k] += 1

        # Calculate probabilities for next state_k given each action_j and state_i
        trans_probs[:,:,:] = transition_count[:,:,:]
        for i in range(np.shape(trans_probs)[0]):
            for j in range(np.shape(trans_probs)[1]):
                trans_probs[i, j, :] = trans_probs[i, j, :] / trans_probs[i, j, :].sum()
        #print(f'|\t Count Sample {int(transition_count[transition_count.shape[0]/2), int(transition_count.shape[1]/2), int(transition_count.shape[2]/2)]}')
        #print(f'|\t Probability Sample {trans_probs[int(trans_probs.shape[0]/2),int(trans_probs.shape[1]/2), int(trans_probs.shape[1]/2)]}')

        return trans_probs

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

    def reward_save(self,reward,path,SAVE=False):
        if SAVE:
            print(f'Saving reward map...')
            print(f'|\t Path: {path}')
            np.save(path, reward)
        else:
            print(f'Model was not saved...')

    def plot_test(self,policy,X_test,y_test,VERBOSE=False):
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


def printh(*args,ENABLE=True,LOG_OUTPUT=True):
    global printh_logger
    if not ('printh_logger' in globals()):
        print('\nprinth...\n|\t logger initialized')
        printh_logger = []

    header = ''
    disp = ''
    level = args[0]
    for arg in args[1:]:
        if type(arg)==tuple:
            for a in arg:
                disp = disp +str(a)
        else: disp = disp + str(arg)

    if ENABLE:
        if level==0: header='\n'
        for i in range(level): header = header + '|\t'
        disp =header + disp
        print(disp)
        if LOG_OUTPUT: printh_logger.append(disp)

    return printh_logger



def load_obj(model_name,model_dir = 'C:\\Users\\mason\Desktop\\Thesis\\ML_MasonSmithGit\Example_Scripts\\GeneratedModels\\IRL\\'):
    import pickle
    path = model_dir + model_name
    with open(path, 'rb') as IRL_obj:
        IRL = pickle.load(IRL_obj)

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
        IRL.unique_states = [IRL.N_STATES, IRL.N_STATES, IRL.N_STATES]
    return IRL

if __name__ == "__main__":
    from Modules.Learning.IRL import IRL_main
    IRL_main.main()