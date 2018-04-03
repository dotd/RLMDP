


def deep_simple_solver(mdp):
    pass

def get_J_as_deep(trajectory, gamma, X=None):
    X = np.array([vec[0] for vec in trajectory]).max()+1 if X is None else X
