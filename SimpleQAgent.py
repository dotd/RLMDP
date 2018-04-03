import numpy as np

class SimpleQAgent:
    def __init__(self, X, U, gamma, epsil=0.01, alpha=1e-3, random=np.random.RandomState(0)):
        self.X = X
        self.U = U
        self.gamma = gamma
        self.epsilon = epsil
        # optimistic Q-learning
        self.random = random
        self.Q = np.ones(shape=(X,U))/(1-gamma)


    def choose_action(self, x):
        # Doing the epsilon greedy step
        if self.random.rand()<self.epsilon:
            return self.random.randint(0,self.U)
        return np.argmax(self.Q[x])

    def update(self, x, u, r, y):
        next_action = np.argmax(self.Q[y])
        self.Q[x,u] = self.Q[x,u] + self.alpha * (r + self.gamma * self.Q[y][next_action] - self.Q[x][u])




