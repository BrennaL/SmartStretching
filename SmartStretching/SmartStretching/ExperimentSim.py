import numpy as np
from SmartStretching.Agents import ContextualTSAgent, LinUCBBandit

# simulated bandit experiment
class Experiment:
    def __init__(self, iter, agent, n_f, n_a, non_stationary):
        self.agent = agent
        self.iterations = iter
        self.non_stationary = non_stationary
        self.context_dim = n_f
        self.action_dim = n_a
        self.true_thetas = None

    def get_true_reward(self, x, a):
        return np.dot(x, self.true_thetas[a]) + np.random.rand() * 0.5

    def get_rand_context(self):
        return np.random.randint(0, 5, (self.context_dim))

    def gen_true_thetas(self):
        self.true_thetas = np.random.rand(self.action_dim, self.context_dim) * 3. - 1.5
        print('Ground truth thetas')
        for a in range(self.action_dim):
            print('action ' + str(a) + ': ' + str(self.true_thetas[a]))

    def get_optimal_action(self, x):
        return np.argmax([self.get_true_reward(x, a) for a in range(self.action_dim)])

    def run(self):
        print('\n\nExperiment Starts')
        np.random.seed(0)
        regret = 0.
        regret_list = []
        t_change = np.floor(self.iterations * 0.5)
        self.gen_true_thetas()
        for t in range(self.iterations):
            context = self.get_rand_context()
            a = self.agent.choose_action(context)
            r = self.get_true_reward(context, a)
            self.agent.update_model(context, a, r)
            regret += abs(r - self.get_true_reward(context, self.get_optimal_action(context)))
            regret_list.append(regret)

            if self.non_stationary and t == t_change:
                print('env changed!')
                self.gen_true_thetas()

        print('Estimated thetas')
        for a in range(self.agent.a_dim):
            print('action ' + str(a) + ': ' + str(self.agent.thetas[a]))

        return


if __name__ == '__main__':
    ITERATIONS = 120
    FEATURES = 3
    ACTIONS = 5
    exp1 = Experiment(ITERATIONS, LinUCBBandit(FEATURES, ACTIONS),FEATURES, ACTIONS, False)
    exp2 = Experiment(ITERATIONS, LinUCBBandit(FEATURES, ACTIONS),FEATURES, ACTIONS, True)
    exp1.run()
    exp2.run()
    #right now hard coded for 3 context features
    #exp3 = Experiment(ITERATIONS, agent = ContextualTSAgent(FEATURES, ACTIONS), 3, ACTIONS, False)
    #exp3.run()

