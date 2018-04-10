import numpy as np

class SimpleQAgent:
    def __init__(self, X, U, gamma, epsil=0.1, alpha=1e-1, random=np.random.RandomState(0)):
        self.X = X
        self.U = U
        self.gamma = gamma
        self.epsilon = epsil
        # optimistic Q-learning
        self.random = random
        self.Q = np.ones(shape=(X,U))/(1-gamma)
        self.alpha = alpha

    def choose_action(self, x):
        # Doing the epsilon greedy step
        if self.random.rand()<self.epsilon:
            return self.random.randint(0,self.U)
        return np.argmax(self.Q[x])

    def get_maximal_action(self, state, epsil = 0):
        maximal_value = np.amax(self.Q[state])
        where = np.where(self.Q[state]>=(maximal_value-epsil))[0]
        action = self.random.choice(where)
        return action

    def update(self, x, u, r, y):
        next_action = np.argmax(self.Q[y])
        self.Q[x,u] = self.Q[x,u] + self.alpha * (r + self.gamma * self.Q[y][next_action] - self.Q[x][u])

    def get_policy(self):
        policy = np.zeros(shape=(self.X, self.U))
        for x in range(self.X):
            maximal_action = self.get_maximal_action(x)
            policy[x][maximal_action] = 1.0
        return policy




