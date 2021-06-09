import gym
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from Modules.Learning.IRL.IRL_Tools import load_obj

def get_terminal_states(N_STATES,num_states,n_features):
    spf = [N_STATES ** (n_features - s - 1) for s in range(n_features)]
    gp100 = spf[0] * (N_STATES - 1)
    terminal_states = np.arange(gp100, num_states)
    return terminal_states.tolist()


model_name_irl= '03_02_2021_IRL_P3S3_PHS_7sx20a_obj'
IRL = load_obj(model_name_irl)
state0 = [11,  2.0049, 16.373 ]

scale = 1.01*(np.array(IRL.max_states) - np.array(IRL.min_states)) / (IRL.states_per_feature)
IRL.state_scales=scale
IRL.n_features = 3
TERMINAL_STATES= get_terminal_states(IRL.N_STATES,IRL.n_states,3)

# Hyperparameters
NUM_EPISODES = 10000
LEARNING_RATE = 0.000025
GAMMA = 0.99

# Create gym and seed numpy
nA = IRL.n_actions
#env = gym.make('CartPole-v0')
#nA = env.action_space.n
np.random.seed(1)

# Init weight
w = np.random.rand(3, 2)
#w = np.random.rand(IRL.policy.shape[0], IRL.policy.shape[1])
# Keep stats for final print of graph
episode_rewards = []


# Our policy that maps state to action parameterized by w
def policy(state, w):
    state = np.array([state])
    z = state.dot(w)
    exp = np.exp(z)
    return exp / np.sum(exp)


# Vectorized softmax Jacobian
def softmax_grad(softmax):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

def takeAction(state,action,transitional_probs,rewards):
    state = IRL.state2int(state)
    P_next_states = transitional_probs[state,action,:]
    next_state = random.choices(np.arange(len(P_next_states)), weights=P_next_states)
    if type(next_state)==list: next_state=next_state[0]

    state_reward=rewards[state]

    if state in TERMINAL_STATES:
        done=True
        print("################\nDone\n################\n\n")
    else: done=False

    return next_state, state_reward, done,None

# Main loop
# Make sure you update your weights AFTER each episode
for e in range(NUM_EPISODES):

    state = state0

    grads = []
    rewards = []

    # Keep track of game score to print
    score = 0

    while True:

        # Uncomment to see your model train in real time (slower)
        # env.render()

        # Sample from policy and take action in environment
        probs = policy(state, w)
        print(probs)
        action = np.random.choice(np.arange(nA), p=probs[0])
        #next_state, reward, done, _ = env.step(action)
        next_state, reward, done, _ = takeAction(state, action,
                                                 IRL.transition_probability,IRL.reward)
        #next_state = next_state[None, :]

        # Compute gradient and save with reward in memory for our weight updates
        dsoftmax = softmax_grad(probs)[action, :]
        dlog = dsoftmax / probs[0, action]
        grad = state.T.dot(dlog[None, :])

        grads.append(grad)
        rewards.append(reward)

        score += reward

        # Dont forget to update your old state to the new state
        state = next_state

        if done:
            break

    # Weight update
    for i in range(len(grads)):
        # Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward
        w += LEARNING_RATE * grads[i] * sum([r * (GAMMA ** r) for t, r in enumerate(rewards[i:])])

    # Append for logging and print
    episode_rewards.append(score)
    print("EP: " + str(e) + " Score: " + str(score) + "         ", end="\r", flush=False)

plt.plot(np.arange(num_episodes), episode_rewards)
plt.show()
env.close()