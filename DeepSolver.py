


def deep_simple_solver(mdp):
    pass

def get_J_as_deep(trajectory, gamma, X=None):
    X = np.array([vec[0] for vec in trajectory]).max()+1 if X is None else X

class DeepSolver():
    def __init__(self, X, gamma, layer_sizes=(2, 2, 1), layer_activations=(torch.nn.ReLU, torch.nn.Linear), input_mode="one_hot"):
        self.X = X
        self.gamma = gamma

        x = torch.nn.Sequential(*chain(*[(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]), layer_activations[i]) for i in range(len(layer_sizes))]))

        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
        )

        if input_mode == "one_hot":


    def solve(self):