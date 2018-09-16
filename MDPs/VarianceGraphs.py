import numpy as np
import matplotlib.pyplot as plt

from MDP import get_R_as_V_minus
from MDPSolver import get_J, get_MRP
import SpecificMDPs
import Policies

"""
This script answers the following question:
Suppose we have two policies: mu1, mu2. 
Now, we want to iterpolate between them with some parameters a.
What is the nature of the interpolation?

Conclusion:
The nature is several fold:
1) It is polynomial
2) It is continuous

"""


X = 30
U = 10
B = 20
gamma = 0.5

random_state = np.random.RandomState(5)

mdp = SpecificMDPs.generate_random_MDP(X=X, U=U, B=B, std=0, random_state=random_state)
mu1 = Policies.generate_deterministic_policy(X, U, random_state)
mu2 = Policies.generate_deterministic_policy(X, U, random_state)

V_col=  []
x_axis = np.linspace(0, 1, 51)
for a in x_axis:
    mu = a*mu1 + (1-a)*mu2
    P, R, R_std = get_MRP(mdp, mu)
    J = get_J(P, R, gamma) # compute J exact
    R = get_R_as_V_minus(P, R, gamma, J)
    V = get_J(P, R, gamma**2)
    V_col.append(V)

plt.plot(x_axis, V_col)
plt.xlabel("a")
plt.ylabel("V")
plt.show()

