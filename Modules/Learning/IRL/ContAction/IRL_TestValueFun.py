from Modules.Learning.IRL.IRL_Tools import load_obj
from Modules.Learning.IRL import IRL_value_iteration
from Modules.Utils.DataHandler import Data_Handler
import numpy as np
import matplotlib.pyplot as plt

model_name_irl= '03_02_2021_IRL_P3S3_PHS_7sx20a_obj'
IRL = load_obj(model_name_irl)

##########################################################################
# Set up Data Handler ####################################################
data_names= [
    #'trimmed_train_P3S2'
    'trimmed_train_P3S3'
         ]
test_data_names= [
    #'trimmed_test_P3S2'
    'trimmed_test_P3S3'
         ]
DataHandler = Data_Handler()
DataHandler.import_KneeAngle=False
#DataHandler.import_HipAngle=False
#DataHandler.import_ShankAngle=False
DataHandler.import_GRF=False

X_train,y_train = DataHandler.import_custom('trimmed_train_P3S3')
X_test,y_test = DataHandler.import_custom('trimmed_test_P3S3')




##########################################################################
# Value Function #########################################################

discount = 0.01
v= IRL_value_iteration.optimal_value(IRL.n_states,
                                  IRL.n_actions,
                                  IRL.transition_probability,
                                  IRL.reward,
                                  discount)

val=IRL_value_iteration.value(IRL.policy,
                            IRL.n_states,
                            IRL.transition_probability,
                            IRL.reward,
                            discount)
print(val)
##########################################################################
# Get Test Predictions  ##################################################
idx_test = IRL.get_idx_demo(X_test, y_test)

obs_states = []
pred_action= []
pred_action_d = []
for trajectory in idx_test:
    for i in range(np.shape(trajectory)[0]):
        state = trajectory[i, 0]
        obs_states.append(state)

for obs in obs_states:
    pred = val[obs]
    pred_action.append(pred)

    prob_of_action = IRL.policy[obs, :]
    pred_action_discrete = np.argmax(prob_of_action)
    pred_action_d.append(pred_action_discrete)



scale = 1.01 * (IRL.max_action - IRL.min_action) / IRL.n_actions
pred_torque = np.array(pred_action) * scale + IRL.min_action
pred_torque_d = np.array(pred_action_d) * scale + IRL.min_action

plt.subplot()
plt.plot(y_test,label='Observed')
plt.plot(pred_torque,label='Pred Val')
plt.plot(pred_torque_d,label='Pred Desc')
plt.legend()
plt.title("Testing Policy")
plt.show()


print(f'n_states {IRL.n_states}')
print(f'Size of Value Function {np.size(v)}')
print(f'Shape idx_demo {np.shape(IRL.idx_demo)}')