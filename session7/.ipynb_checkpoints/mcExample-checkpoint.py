import math
import numpy as np

from treeEx import SpreadOption
from binomial import EuropeanOption, PayoffType, bsPrice
    
def mcEuropean(S0, T, r, q, vol, nPaths, trade):
    np.random.seed(0)
    sum,hsquare = 0,0
    stdev = math.sqrt(T)
    for i in range(nPaths):
        wT = np.random.normal(0, stdev)
        h = trade.payoff(S0 * math.exp((r - q - 0.5*vol*vol) * T + vol * wT))
        sum += h
        hsquare += h * h

    pv = math.exp(-r*T) * sum / nPaths
    stderr = math.sqrt((hsquare/nPaths - (sum/nPaths) * (sum/nPaths)) / nPaths)
    return pv, stderr


def mcLocalVol(S0, T, r, q, lv, nT, nPaths, trade):
    np.random.seed(0)
    sum, hsquare = 0, 0
    dt = T / nT
    sqrtdt = math.sqrt(dt)
    for i in range(nPaths):
        X = math.log(S0)
        for j in range(1, nT+1):
            vol = lv.LV((j-1)*dt, math.exp(X))
            a = (r - q - 0.5*vol * vol) * dt # drift
            b = np.rand.normal(0, sqrtdt) * vol
            X += a + b # update state variable
        h = trade.payoff(math.exp(X))
        sum += h
        hsquare += h * h
    pv = math.exp(-r * T) * sum / nPaths
    stderr = math.sqrt((hsquare / nPaths - (sum / nPaths) * (sum / nPaths)) / nPaths)
    return pv, stderr


def mcSpread(payoff, S1, S2, T, r, q1, q2, vol1, vol2, rho, nPaths, nT):
    np.random.seed(0)
    sum, hsquare, C = 0, 0, np.identity(2)
    C[0, 1] = C[1, 0] = rho
    L = np.linalg.cholesky(C)
    for i in range(nPaths):
        brownians = np.zeros((2, nT))
        dt = T / nT
        stdev = math.sqrt(dt)
        # generate brownian increments
        for j in range(2):
            brownians[j] = np.random.normal(0, stdev, nT)
        brownians = np.matmul(L, brownians)
        x1, x2 = math.log(S1), math.log(S2)
        for j in range(nT):
            # simulate asset 1
            a = (r-q1-0.5*vol1*vol1) * dt # drift for asset 1
            b = brownians[0, j] * vol1 # diffusion term for asset 1
            x1 += a + b  # update state variable
            # simulate asset 2
            a = (r-q2-0.5*vol2*vol1) * dt # drift for asset 1
            b = brownians[1, j] * vol2 # diffusion term for asset 1
            x2 += a + b  # update state variable
        h = payoff(math.exp(x1), math.exp(x2))
        sum += h
        hsquare += h*h
    pv = math.exp(-r * T) * sum / nPaths
    se = math.sqrt((hsquare/nPaths - (sum/nPaths)*(sum/nPaths))/nPaths)
    return pv, se

if __name__ == "__main__":
    payoff = lambda S1, S2: max(S1, S2)
    pv, se = mcSpread(payoff, 100, 100, 1, 0.05, 0.02, 0.03, 0.1, 0.15, 0.5, 1024, 100)
    print("Spread Option Price/Std: ", pv, se)

if __name__ == "__main__":
    S0, K, T, r, q, vol, nPaths = 100, 100, 1.0, 0.01, 0.0, 0.1, 100000
    trade = EuropeanOption(T,K, PayoffType.Call)
    pv, se = mcEuropean(S0, T, r, q, vol, nPaths, trade)
    bsPV = bsPrice(S0, r, vol, T, K, PayoffType.Call)
    print("European Option MC Price/Std: ", pv, se)
    print("European Option BS Price: ", bsPV)
    
    
