import random
import numpy as np
import math
from matplotlib import pyplot as plt

def testMidPoint():
    xs = [0] * 100
    xs[0] = 0.2372 # seed
    for i in range(1, 100):
        xs[i]=(int(xs[i-1]**2*1.0e6)%1e4)/1.0e4
    plt.scatter(range(100), xs)
    plt.show()

testMidPoint()

def testRNG():
    np.random.seed(0)
    data = np.random.normal(0, 1, 10000)
    bins = np.linspace(math.ceil(min(data)),
                       math.floor(max(data)),
                       50)  # fixed number of bins
    plt.xlim([min(data) - 0.5, max(data) + 0.5])
    plt.hist(data, bins=bins)
    plt.xlabel('variable X (50 evenly spaced bins)')
    plt.ylabel('count')
    plt.savefig('../figs/rngNormal.eps', format = 'eps')
    plt.show()

testRNG()


def testBrownian():
    np.random.seed(0)
    # generate 3 brownian motions for 1Y
    nBrownians = 3
    nTimeSteps = 366
    brownians = np.zeros((nBrownians, nTimeSteps))
    # each time step is 1 day, so standard deviation is sqrt(1/365.0)
    stdev = math.sqrt(1/365.0)
    for i in range(nBrownians):
        for j in range(1, nTimeSteps):
            brownians[i, j] = brownians[i, j-1] + np.random.normal(0, stdev)

    plt.title('3 brownian motions for 1Y')
    plt.plot(range(nTimeSteps), brownians[0])
    plt.plot(range(nTimeSteps), brownians[1])
    plt.plot(range(nTimeSteps), brownians[2])
    plt.savefig('../figs/3brownians.eps', format = 'eps')
    plt.show()

testBrownian()