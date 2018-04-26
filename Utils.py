import numpy as np

def OneDVec2ThreeDVec(R_vec, U):
    X = R_vec.shape[0]
    R_mat = np.zeros((U,X,X))
    for x,reward in enumerate(R_vec):
        R_mat[:,x,:] = reward
    return R_mat

def ThreeDVec2OneDVec(R_mat, P, mu):
    X = R_mat.shape[1]
    U = R_mat.shape[0]

    R1D = np.zeros(shape=(X,))
    for x in range(X):
        for u in range(U):
            for y in range(X):
                if R_mat[u,x,y]!=0:
                    qqq=0
                R1D[x] += P[u,x,y] * mu[x,u] * R_mat[u,x,y]
    return R1D

def show_2dMat(mat, **kwargs):
    sep = kwargs.get("sep","\t")
    line_sep = kwargs.get("line_sep","\n")
    lines = []
    for y in range(mat.shape[0]):
        line = []
        for x in range(mat[0].shape[0]):
            line.append(str(mat[y,x]))
        lines.append(sep.join(line))
    return line_sep.join(lines)

def show_3dMat(mat, **kwargs):
    mat_sep = kwargs.get("mat_sep","\n")
    U = mat.shape[0]
    lines = []
    for u in range(U):
        lines.append("u=" + str(u))
        lines.append(show_2dMat(mat[u],**kwargs))
    return mat_sep.join(lines)

def get_random_sparse_vector(X, B, random_state):
    # initialize the size
    P_vec = np.zeros(shape=(X,))
    # put in the first places 0:B all the random variables.
    P_vec[0:B] = random_state.uniform(low=0, high=1.0, size=(B,))

    # initialize the size
    R_vec = np.zeros(shape=(X,))
    # put in the first places 0:B all the random variables.
    R_vec[0:B] = random_state.normal(size=(B,))

    # normalize
    P_vec[0:B] = P_vec[0:B] / np.sum(P_vec[0:B])
    # permute both P_vec and R_vec together.
    idx = random_state.permutation([x for x in range(X)])
    P_vec = P_vec[idx]
    R_vec = R_vec[idx]
    return P_vec, R_vec
