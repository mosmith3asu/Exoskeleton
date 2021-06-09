"""https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb"""
import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
from Modules.Learning.IRL.IRL_Tools import load_obj
import random
if "../" not in sys.path:
  sys.path.append("../")

from collections import defaultdict
from Modules.Learning.IRL.ContAction import QLearningSolution_plotting as plotting
#from lib import plotting

#matplotlib.style.use('ggplot')

def get_terminal_states(N_STATES,num_states,n_features):
    spf = [N_STATES ** (n_features - s - 1) for s in range(n_features)]
    gp100 = spf[0] * (N_STATES - 1)
    terminal_states = np.arange(gp100, num_states)
    return terminal_states.tolist()


model_name_irl= '03_02_2021_IRL_P3S3_PHS_7sx20a_obj'
IRL = load_obj(model_name_irl)
TERMINAL_STATES= get_terminal_states(IRL.N_STATES,IRL.n_states,3)

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn

def takeAction(state,action,transitional_probs,N_STATES,reward):
    P_next_states = transitional_probs[state,action,:]
    next_state = random.choices(np.arange(len(P_next_states)), weights=P_next_states)

    state_reward=reward[state]

    if state in TERMINAL_STATES:
        done=True
        print("################\nDone\n################\n\n")
    else: done=False

    return next_state, state_reward, done,None



def q_learning(policy,trans_probs,rewards,
               n_states,n_actions,
               num_episodes=100,state0=0,
               discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    print(n_states)
    Q = defaultdict(lambda: np.zeros(n_actions))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, n_actions)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        #state = env.reset()
        state=state0
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            #action_probs = policy(state)
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            #next_state, reward, done, _ = env.step(action)
            next_state, reward, done, _ = takeAction(state,action,trans_probs,n_states,rewards)
            next_state = next_state[0]
            print(f'Action {action} State {state} Next State {next_state} Reward {rewards[state]} Done {done}')
            #print(f' ')

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

    return Q, stats



Q, stats = q_learning(policy=IRL.policy,
                      trans_probs=IRL.transition_probability,
                      rewards=IRL.reward,
                      #n_states=IRL.N_STATES,
                      n_states=IRL.n_states,
                      n_actions=IRL.n_actions,
                      state0=IRL.idx_demo[0,0,0],
                      num_episodes=500)
state = [11,  2.0049, 16.373 ]
scale = 1.01*(np.array(IRL.max_states) - np.array(IRL.min_states)) / (IRL.states_per_feature)
IRL.state_scales=scale
IRL.n_features = 3
state = IRL.state2int(state)
print(f'Action Values {Q[state]}')
plotting.plot_episode_stats(stats)