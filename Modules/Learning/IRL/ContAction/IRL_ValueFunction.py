"""https://towardsdatascience.com/reinforcement-learning-generalisation-in-continuous-state-space-df943b04ebfa"""
import numpy as np
import random

class LinearValueFunction:
    def __init__(self, order,NUM_STATES, method="poly"):
        self.NUM_STATES=NUM_STATES
        if method == "poly":
            self.func = [lambda x, i=i: np.power(x, i) for i in range(0, order + 1)]  # s^i
        if method == "fourier":
            self.func = [lambda x, i=i: np.cos(np.pi * x * i) for i in range(0, order + 1)]  # cos(pi*s*i)
        self.weights = np.zeros(order + 1)

    def value(self, state):
        state = state / self.NUM_STATES
        features = np.array([f(state) for f in self.func])
        return np.dot(features, self.weights)

    def update(self, delta, state):
        state = state / self.NUM_STATES
        dev = np.array([f(state) for f in self.func])
        self.weights += delta * dev


class ContinousPrediction:
    def __init__(self, order, N_STATES,reward,reward_shape, method="poly"):
        self.reward_map = np.reshape(reward,reward_shape)

        self.N_STATES = N_STATES
        if method == "poly":
            self.func = [lambda x, i=i: np.power(x, i) for i in range(0, order + 1)]  # s^i
        if method == "fourier":
            self.func = [lambda x, i=i: np.cos(np.pi * x * i) for i in range(0, order + 1)]  # cos(pi*s*i)
        self.weights = np.zeros(order + 1)


    def giveReward(self,prev_state,next_state):
        transition_reward = self.reward_map[prev_state,next_state]
        return transition_reward
    def chooseAction(self):

    def takeAction(self, state,action,transitional_probs):
        P_next_states = transitional_probs[state,action,:]
        next_state = random.choices(np.arange(len(P_next_states)), weights=P_next_states)

        gait_phase = state/self.N_STATES
        if gait_phase==100: end=True
        else: end=False

        return next_state,end

    def main(self,state0,action0,round,VERBOSE=True):
        for rnd in range(rounds):
            #self.reset()
            t = 0
            T = np.inf
            action = action0

            state = state0
            actions = [action0]
            states = [state0]
            rewards = [0]
            while True:
                if t < T:
                    new_state,end = self.takeAction(action) # next state
                    reward = self.giveReward(state,new_state)      # next state-reward

                    state=new_state
                    states.append(state)
                    rewards.append(reward)

                    if end:
                        if VERBOSE:
                            if (rnd + 1) % 5000 == 0:
                                print("Round {}: End at state {} | number of states {}".format(rnd + 1, state, len(states)))
                        T = t + 1
                    else:
                        action = self.chooseAction()
                        actions.append(action)  # next action
                # state tau being updated
                tau = t - self.n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau + self.n + 1, T + 1)):
                        G += np.power(self.gamma, i - tau - 1) * rewards[i]
                    if tau + self.n < T:
                        state = states[tau + self.n]
                        G += np.power(self.gamma, self.n) * valueFunction.value(state)
                    # update value function
                    state = states[tau]
                    delta = self.lr * (G - valueFunction.value(state))
                    valueFunction.update(delta, state)

                if tau == T - 1:
                    break

                t += 1