import numpy as np
import pandas as pd
import pymc3 as pm
from numpy.linalg import inv

# Base bandit class
class Agent:
    def __init__(self, feature_dim, action_dim):
        self.thetas = np.zeros((action_dim, feature_dim))
        # action space dim
        self.a_dim = action_dim
        # feature space dim
        self.x_dim = feature_dim

    def choose_action(self, x):
        pass

    def update_model(self, x, a, r):
        pass


# random bandit
class RandomBandit(Agent):
    def __init__(self, feature_dim, n_action):
        Agent.__init__(self, feature_dim, n_action)

    def choose_action(self, x):
        return np.random.randint(0, self.a_dim)


# LinUCB
class LinUCBBandit(Agent):
    def __init__(self, feature_dim, n_action):
        Agent.__init__(self, feature_dim, n_action)
        self.Ds = [np.zeros(self.x_dim).reshape(1, self.x_dim)] * self.a_dim  # exp
        self.Cs = [0] * self.a_dim  # rewards

    def choose_action(self, x):
        delta = 0.3
        max_E = np.NINF
        best_a = 0
        for a in range(self.a_dim):
            D = self.Ds[a]
            A = np.dot(D.T, D) + np.identity(self.x_dim)
            A_inverse = inv(A)
            alpha = 1. + np.sqrt(np.log(2. / delta) / 2.)
            bound = alpha * np.sqrt(np.dot(np.dot(x.T, A_inverse), x))
            E = np.dot(x.T, self.thetas[a]) + bound
            if E > max_E:
                max_E = E
                best_a = a

        return best_a

    def update_model(self, x, a, r):
        d = len(x)
        self.Ds[a] = np.append(self.Ds[a], [x], axis=0)
        self.Cs[a] = np.append(self.Cs[a], r)
        D = self.Ds[a]
        C = self.Cs[a]
        inverse = inv(np.dot(D.T, D) + np.identity(d))
        new_theta = np.dot(np.dot(inverse, D.T), C)
        self.thetas[a] = new_theta


# Thompson Sampling with Bayesian Regression, some parts are hard coded now, do not use
class ContextualTSAgent(Agent):
    def __init__(self, feature_dim, n_action):
        Agent.__init__(self, feature_dim, n_action)
        self.exp_dict = {}  # experience dict for each action

    def choose_action(self, x):
        r_list = []
        for a in range(self.a_dim):
            r_list.append(self.get_reward_sample(x, a))
        return np.argmax(r_list)

    def get_reward_sample(self, x, a):
        r = 0.
        if a not in self.exp_dict:
            return r

        with pm.Model() as model:
            pm.GLM.from_formula('r ~ x_0 + x_1 + x_2', data=self.exp_dict[a])
            traces = pm.sample(draws=1, chains=1, tune=100, compute_convergence_checks=False)
            params = traces[0]
            r = x[0] * params['x_0'] + x[1] * params['x_1'] + x[2] * params['x_2'] + params['Intercept']
        return r

    def update_model(self, x, a, r):
        exp = []
        exp.extend(x)
        exp.append(r)
        # new_df =  pd.DataFrame([exp])
        new_df = pd.DataFrame({**{'x_{}'.format(i): x[i] for i in range(len(x))}, "r": r}, index=[0])
        if a not in self.exp_dict:
            self.exp_dict[a] = new_df
        else:
            self.exp_dict[a] = pd.concat([new_df, self.exp_dict[a]])
