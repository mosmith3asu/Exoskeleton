from Modules.Learning.IRL.IRL_Tools import IRL_tools
from Modules.Learning.IRL import IRL_maxent as maxent
from Modules.Utils.DataHandler import Data_Handler
from datetime import datetime
import numpy as np

time_begin = datetime.now() # current date and time
DataHandler = Data_Handler() # Import Data handler

##########################################################################
# TRAINING PARAMETERS ####################################################
# Enables
VERBOSE = True
SAVE_MODEL = True

# Data
train_data_names= [
    'trimmed_train_V1P3S2',
    'trimmed_train_P3S3'
         ]
test_data_names= [
    'trimmed_test_V1P3S2',
    'trimmed_test_P3S3'
         ]
trans_prob_data= [
    'trimmed_train_V1P3S2',
    'trimmed_test_V1P3S2',
    'trimmed_train_P3S3',
    'trimmed_test_P3S3'
         ]

training_notes={
    "Training Data Names": train_data_names,
    "Time Begin": time_begin,
    "Time End": None, # added after finished with learning
    "Time Elapsed": None
}

##########################################################################
# IMPORT DATA ############################################################
for name in train_data_names:
    X_train, y_train = DataHandler.import_custom(name)
for name in test_data_names:
    X_train, y_train = DataHandler.import_custom(name)

X_train, y_train = DataHandler.import_custom(train_data_names)
X_test, y_test = DataHandler.import_custom(train_data_names)


def main(discount=0.01, epochs=100, learning_rate=0.01):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    n_states = 10
    n_actions = 20

    # Learn Reward ###########################
    IRL = IRL_tools(X_train,y_train,
                   trans_prob_names=trans_prob_data,
                   N_STATES=n_states,N_ACTIONS=n_actions)

    trajectories = IRL.idx_demo
    r = maxent.irl(IRL.feature_matrix, IRL.n_actions, discount,
                   IRL.transition_probability, trajectories, epochs, learning_rate)

    # Update training notes ###########################
    time_end = datetime.now()
    time_elapsed = time_end - time_begin
    training_notes["Time End"] = time_end
    training_notes["Time Elapsed"] = time_elapsed
    IRL.notes = training_notes

    # Recover Policy ###########################
    policy = maxent.find_policy(IRL.n_states,r,IRL.n_actions,discount,IRL.transition_probability)

    # Save Model ###########################
    if SAVE_MODEL:
        date = time_begin.strftime("%m_%d_%Y")  # get current date
        base = 'IRL'
        data_name = DataHandler.append_names(train_data_names)
        feature_prefix = DataHandler.feature_prefix()
        SA_prefix = f'{n_states}sx{n_actions}a'
        MODEL_NAME = f"{date}_{base}_{data_name}_{feature_prefix}_{SA_prefix}"  # Construnct final model name
        if VERBOSE: print("Saving Model as:\n|\t ", MODEL_NAME)

        IRL.save_obj(MODEL_NAME,reward=r,policy=policy,ENABLE=SAVE_MODEL)

    # Plot Results ###########################
    IRL.plot_test(policy,X_test,y_test)
    if np.shape(X_test)[1]==2:
        IRL.reward_output(reward=r)
        IRL.reward_plot(reward=r)
    if VERBOSE:
        print("\nFINISHED...")
        time_end = datetime.now()
        time_elapsed = time_end - time_begin
        print(f"|\t Elapse time (seconds): {time_elapsed.total_seconds()}")



if __name__ == '__main__':
    main(epochs=100)