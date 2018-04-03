import numpy as np

class SimpleQAgent:
    def __init__(self, X, U, epsil=0.01, gamma): 
        self.X = X
        self.U = U
        self.Q = np.zeros(shape=(X,U))

