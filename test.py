import gymnasium as gym
import matplotlib
import IPython
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
from utils import JupyterRender

env = gym.make("FrozenLake-v1", render_mode='rgb_array', is_slippery=False)#define the environment.
env = JupyterRender(env)


#env.P: {0: {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 4, 0.0, False)], 2: [(1.0, 1, 0.0, False)], 3: [(1.0, 0, 0.0, False)]}, 1: {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 5, 0.0, True)], 2: [(1.0, 2, 0.0, False)], 3: [(1.0, 1, 0.0, False)]}, 2: {0: [(1.0, 1, 0.0, False)], 1: [(1.0, 6, 0.0, False)], 2: [(1.0, 3, 0.0, False)], 3: [(1.0, 2, 0.0, False)]}, 3: {0: [(1.0, 2, 0.0, False)], 1: [(1.0, 7, 0.0, True)], 2: [(1.0, 3, 0.0, False)], 3: [(1.0, 3, 0.0, False)]}, 4: {0: [(1.0, 4, 0.0, False)], 1: [(1.0, 8, 0.0, False)], 2: [(1.0, 5, 0.0, True)], 3: [(1.0, 0, 0.0, False)]}, 5: {0: [(1.0, 5, 0, True)], 1: [(1.0, 5, 0, True)], 2: [(1.0, 5, 0, True)], 3: [(1.0, 5, 0, True)]}, 6: {0: [(1.0, 5, 0.0, True)], 1: [(1.0, 10, 0.0, False)], 2: [(1.0, 7, 0.0, True)], 3: [(1.0, 2, 0.0, False)]}, 7: {0: [(1.0, 7, 0, True)], 1: [(1.0, 7, 0, True)], 2: [(1.0, 7, 0, True)], 3: [(1.0, 7, 0, True)]}, 8: {0: [(1.0, 8, 0.0, False)], 1: [(1.0, 12, 0.0, True)], 2: [(1.0, 9, 0.0, False)], 3: [(1.0, 4, 0.0, False)]}, 9: {0: [(1.0, 8, 0.0, False)], 1: [(1.0, 13, 0.0, False)], 2: [(1.0, 10, 0.0, False)], 3: [(1.0, 5, 0.0, True)]}, 10: {0: [(1.0, 9, 0.0, False)], 1: [(1.0, 14, 0.0, False)], 2: [(1.0, 11, 0.0, True)], 3: [(1.0, 6, 0.0, False)]}, 11: {0: [(1.0, 11, 0, True)], 1: [(1.0, 11, 0, True)], 2: [(1.0, 11, 0, True)], 3: [(1.0, 11, 0, True)]}, 12: {0: [(1.0, 12, 0, True)], 1: [(1.0, 12, 0, True)], 2: [(1.0, 12, 0, True)], 3: [(1.0, 12, 0, True)]}, 13: {0: [(1.0, 12, 0.0, True)], 1: [(1.0, 13, 0.0, False)], 2: [(1.0, 14, 0.0, False)], 3: [(1.0, 9, 0.0, False)]}, 14: {0: [(1.0, 13, 0.0, False)], 1: [(1.0, 14, 0.0, False)], 2: [(1.0, 15, 1.0, True)], 3: [(1.0, 10, 0.0, False)]}, 15: {0: [(1.0, 15, 0, True)], 1: [(1.0, 15, 0, True)], 2: [(1.0, 15, 0, True)], 3: [(1.0, 15, 0, True)]}}

class Policy_Iter:
    def __init__(self, env, gamma=0.99, theta=1e-8):
        self.env = env

        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n

        self.gamma = gamma
        self.theta = theta

        # 1. Initialization
        self.pi = np.zeros([self.state_dim], dtype=int)
        self.v = np.zeros([self.state_dim])


    def run(self):
        while True:
            # 2. Policy Evaluation
            while True:
                delta = 0
                for i in range(self.state_dim):
                    v = self.v[i]

                    tmp = 0
                    print(i, self.pi[i])
                    for p, s_, r, done in self.env.P[i][self.pi[i]]:
                        tmp += p * (r + self.gamma * self.v[s_])

                    self.v[i] = tmp

                    delta = max(delta, abs(v - self.v[i]))

                if delta < self.theta:
                    break

            # 3. Policy Improvement
            policy_stable = True
            for i in range(self.state_dim):
                old_action = self.pi[i]

                values_by_action = []
                for j in range(self.action_dim):
                    tmp = 0
                    for p, s_, r, done in self.env.P[i][j]:
                        tmp += p * (r + self.gamma * self.v[s_])

                    values_by_action.append(tmp)

                self.pi[i] = np.argmax(values_by_action)

                if self.pi[i] != old_action:
                    policy_stable = False

            if policy_stable:
                print(self.pi.reshape((4, -1)))
                print(self.v.reshape((4, -1)))
                break



class Value_Iter:
    def __init__(self, env, gamma=0.99, theta=1e-8):
        self.env = env

        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n

        self.gamma = gamma
        self.theta = theta

        # 1. Initialization
        self.v = np.zeros([self.state_dim])
        self.pi = np.zeros([self.state_dim], dtype=int)

    def run(self):
        while True:
            delta = 0.
            for i in range(self.state_dim):
                current_v = self.v[i]

                values_by_action = []
                for j in range(self.action_dim):
                    tmp = 0
                    for p, s_, r, done in self.env.P[i][j]:
                        tmp += p * (r + self.gamma * self.v[s_])

                    values_by_action.append(tmp)
                self.v[i] = max(values_by_action)

                delta = max(delta, abs(self.v[i] - current_v))
            if delta < self.theta:
                break

        for i in range(self.state_dim):
            values_by_action = []
            for j in range(self.action_dim):
                tmp = 0
                for p, s_, r, done in self.env.P[i][j]:
                    tmp += p * (r + self.gamma * self.v[s_])

                values_by_action.append(tmp)

            self.pi[i] = np.argmax(values_by_action)

        print(self.pi.reshape((4, -1)))
        print(self.v.reshape((4, -1)))


a = Policy_Iter(env)
a.run()

b = Value_Iter(env)
b.run()