import matplotlib.pyplot as plt
import numpy as np
from matplotlib import interactive


def show_J_collector(J_collector):

    #interactive(True)
    plt.subplot(211)
    plt.xlabel("iteration")
    plt.ylabel("J")
    # The parameters can also be 2-dimensional. Then, the columns represent separate data sets.
    plt.plot(J_collector)
    plt.subplot(212)
    plt.plot(np.diff(J_collector, axis=0))
    plt.xlabel("iteration")
    plt.ylabel("diff J")
    plt.show(block=False)

