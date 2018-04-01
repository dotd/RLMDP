import numpy as np

def generate_uniform_policy(X,U):
    policy = np.ones(shape=(X,U))/U
    return policy

def generate_deterministic_policy(X, U, random = np.random.RandomState(0)):
    policy = np.zeros(shape=(X,U))
    for x in range(X):
        u = random.randint(0,U)
        policy[x,u] = 1.0
    return policy
