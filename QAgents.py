import numpy as np

class SimpleQAgent:
    def __init__(self, X, U, gamma, random=np.random.RandomState(0), **kwargs):
        self.X = X
        self.U = U
        self.gamma = gamma
        self.random = random

        # epsilon start
        self.epsilon_start = kwargs.get("epsilon_start", 0.9)
        self.epsilon_end = kwargs.get("epsilon_end", 0)
        self.epsilon_factor = kwargs.get("epsilon_factor", 0.999)

        self.step_start = kwargs.get("step_start", 1e-1)
        self.step_end = kwargs.get("step_end", 0)
        self.step_factor = kwargs.get("step_factor", 1e-1)

        self.steps_counter = 0

        # when maximizing for selecting the action, what is the tolerance of maximal actions.
        self.accuracy_maximal_Q = kwargs.get("accuracy_maximal_Q", 0)

        # optimistic Q-learning
        self.Q = np.ones(shape=(X,U))/(1-gamma)
        self.update_epsilon()
        self.update_learning_rate()

    def update_epsilon(self):
        self.epsilon = (self.epsilon_start-self.epsilon_end) * (self.epsilon_factor ** self.steps_counter) + self.epsilon_end

    def update_learning_rate(self):
        self.lr = self.step_start / ((self.steps_counter + 1) ** self.step_factor) + self.step_end

    def select_action(self, x):
        # Doing the epsilon greedy step
        if self.random.rand()<self.epsilon:
            return self.random.randint(0,self.U)
        # otherwise, take the maximum action
        action = self.get_maximal_action(x)
        return action

    def get_maximal_action(self, state):
        '''
        This method deals better with several maximal actions, up to tolerance
        This function doesn't deal with the epsilon mechanism
        '''
        maximal_value = np.amax(self.Q[state])
        where = np.where(self.Q[state]>=(maximal_value-self.accuracy_maximal_Q))[0]
        action = self.random.choice(where)
        return action

    def update(self, x, u, r, y):
        self.update_epsilon()
        self.update_learning_rate()
        next_action = np.argmax(self.Q[y])
        self.Q[x,u] = self.Q[x,u] + self.lr * (r + self.gamma * self.Q[y][next_action] - self.Q[x][u])
        self.steps_counter += 1

    def get_policy(self):
        policy = np.zeros(shape=(self.X, self.U))
        for x in range(self.X):
            maximal_action = self.get_maximal_action(x)
            policy[x][maximal_action] = 1.0
        return policy

    def get_Q(self):
        return self.Q




