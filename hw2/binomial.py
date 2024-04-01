import time
import timeit
from enum import Enum
import math
import matplotlib.pyplot as plt
import numpy as np

# one step binomial
from enum import Enum
import math

#from intro import opTiming

class PayoffType(str, Enum):
    Call = 'Call'
    Put = 'Put'
    BinaryCall = 'BinaryCall'
    BinaryPut = 'BinaryPut'

def oneStepBinomial(S:float, r:float, u:float, d:float, optType:PayoffType, K:float, T:float) -> float:
    p = (math.exp(r * T) - d) / (u-d)
    if optType == PayoffType.Call:
        return math.exp(-r*T) * (p*max(S*u-K, 0) + (1-p) * max(S*d-K, 0))

def testoneStepBinomial():
    print("oneStepBinomial: ", oneStepBinomial(S=100, r=0.01, u=1.2, d=0.8, optType=PayoffType.Call, K=105, T=1.0))
if __name__ == "__main__":
    testoneStepBinomial()

# Black-Scholes analytic pricer
def cnorm(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bsPrice(S, r, vol, T, strike, payoffType):
    fwd = S * math.exp(r * T)
    stdev = vol * math.sqrt(T)
    d1 = math.log(fwd / strike) / stdev + stdev / 2
    d2 = d1 - stdev
    if payoffType == PayoffType.Call:
        return math.exp(-r * T) * (fwd * cnorm(d1) - cnorm(d2) * strike)
    elif payoffType == PayoffType.Put:
        return math.exp(-r * T) * (strike * cnorm(-d2) - cnorm(-d1) * fwd)
    elif payoffType == PayoffType.BinaryCall:
        return math.exp(-r * T) * cnorm(d1)
    elif payoffType == PayoffType.BinaryPut:
        return math.exp(-r * T) * (1-cnorm(-d1))
    else:
        raise Exception("not supported payoff type", payoffType)

def testoneStepBinomialBS():
# test ---
    S, r, vol, K, T, u, d = 100, 0.01, 0.2, 105, 1.0, 1.2, 0.8
    print("BSPrice: ", bsPrice(S, r, vol, T, K, PayoffType.Call))
    print("oneStepTree: ", oneStepBinomial(S, r, u, d, PayoffType.Call, K, T))

if __name__ == "__main__":
    testoneStepBinomialBS()

def oneStepBinomial2(S, r, vol, optType, K, T):
    b = math.exp(vol * vol * T+r*T) + math.exp(-r * T)
    u = (b + math.sqrt(b*b - 4)) / 2
    d = 1/u
    p = (math.exp(r * T) - d) / (u-d)
    if optType == PayoffType.Call:
        return math.exp(-r * T) * (p * max(S * u - K, 0) + (1-p) * max(S * d - K, 0))
# test ---
def testoneStepBinomialBS():
    S,r,vol,K,T,u,d = 100, 0.01, 0.2, 105, 1.0, 1.2, 0.8
    print("blackPrice: ", bsPrice(S, r, vol, T, K, PayoffType.Call))
    print("oneStepTree1: \t", oneStepBinomial(S, r, u, d, PayoffType.Call, K, T))
    print("oneStepTree2: \t", oneStepBinomial2(S, r, vol, PayoffType.Call, K, T))
if __name__ == "__main__":
    testoneStepBinomialBS()

def crrBinomial(S, r, vol, payoffType, K, T, n):
    t = T / n
    b = math.exp(vol * vol * t+r*t) + math.exp(-r * t)
    u = (b + math.sqrt(b*b - 4)) / 2
    p = (math.exp(r * t) - (1/u)) / (u - 1/u)
    # set up the last time slice, there are n+1 nodes at the last time slice
    payoffDict = {
        PayoffType.Call: lambda s: max(s-K, 0),
        PayoffType.Put: lambda s: max(K-s, 0),
    }
    vs = [payoffDict[payoffType]( S * u**(n-i-i)) for i in range(n+1)]
    # iterate backward
    for i in range(n-1, -1, -1):
        # calculate the value of each node at time slide i, there are i nodes
        for j in range(i+1):
            vs[j] = math.exp(-r * t) * (vs[j] * p + vs[j+1] * (1-p))
    return vs[0]

def testoneStepBinomialBSPlot():
    # test ---
    S, r, vol, K, T = 100, 0.01, 0.2, 105, 1.0
    print("blackPrice: \t", bsPrice(S, r, vol, T, K, PayoffType.Call))
    print("crrNStepTree: \t", crrBinomial(S, r, vol, PayoffType.Call, K, T, 300))

    start = time.time()
    crrBinomial(S, r, vol, PayoffType.Call, K, T, 200)
    print("crr Binomial Tree pricing time: ", time.time() - start)

    start = time.time()
    bsPrice(S, r, vol, T, K, PayoffType.Call)
    print("BS close form formula pricing time: ", (time.time() - start) * 1e9)

    import matplotlib.pyplot as plt
    n = 300
    S, r, vol, K, T = 100, 0.01, 0.2, 105, 1.0
    bsPrc = bsPrice(S, r, vol, T, K, PayoffType.Call)
    crrErrs = [(crrBinomial(S,r,vol,PayoffType.Call,K,T,i) - bsPrc) for i in range(1, n)]

    plt.plot(range(1, n), crrErrs, label = "CRR - BSAnalytic")
    plt.xlabel('number of tree steps')
    plt.legend()
    # plt.yscale('log')
    plt.show()
if __name__ == "__main__":
    testoneStepBinomialBSPlot()
# Different payoff class
class EuropeanOption():
  def __init__(self, expiry, strike, payoffType):
        self.expiry = expiry
        self.strike = strike
        self.payoffType = payoffType
  def payoff(self, S):
        if self.payoffType == PayoffType.Call:
            return max(S - self.strike, 0)
        elif self.payoffType == PayoffType.Put:
            return max(self.strike - S, 0)
        elif self.payoffType == PayoffType.BinaryCall:
            if S > self.strike:
                return 1.0
            else:
                return 0.0
        elif self.payoffType == PayoffType.BinaryPut:
            if S < self.strike:
                return 1.0
            else:
                return 0.0
        else:
            raise Exception("payoffType not supported: ", self.payoffType)
  def valueAtNode(self, t, S, continuation):
        if continuation == None:
            return self.payoff(S)
        else:
            return continuation

class AmericanOption():
    def __init__(self, expiry, strike, payoffType):
        self.expiry = expiry
        self.strike = strike
        self.payoffType = payoffType
    def payoff(self, S):
        if self.payoffType == PayoffType.Call:
            return max(S - self.strike, 0)
        elif self.payoffType == PayoffType.Put:
            return max(self.strike - S, 0)
        else:
            raise Exception("payoffType not supported: ", self.payoffType)
    def valueAtNode(self, t, S, continuation):
        return max(self.payoff(S), continuation)

class BarrierOption():
    def __init__(self, downBarrier, upBarrier, barrierStart, barrierEnd, underlyingOption):
        self.underlyingOption = underlyingOption
        self.barrierStart = barrierStart
        self.barrierEnd = barrierEnd
        self.downBarrier = downBarrier
        self.upBarrier = upBarrier
        self.expiry = underlyingOption.expiry
    def payoff(self, S):
        return self.underlyingOption.payoff(S)
    def valueAtNode(self, t, S, continuation):
        if t > self.barrierStart and t < self.barrierEnd:
            if self.upBarrier != None and S > self.upBarrier:
                return 0
            elif self.downBarrier != None and S < self.downBarrier:
                return 0
        return continuation

class AsianOption():
    def __init__(self, fixings, payoffFun, As, nT):
        self.fixings = fixings
        self.payoffFun = payoffFun
        self.expiry = fixings[-1]
        self.nFixings = len(fixings)
        self.As, self.nT = nT
        self.dt = self.expiry / nT
    def onFixingDate(self, t):
        # we say t is on a fixing date if there is a fixing date T_i \in (t-dt, t]
        return filter(lambda x: x > t - self.dt and x<=t, self.fixings)
    def valueAtNode(self, t, S, continuation):
        if continuation == None:
            return [self.payoffFun((a*(self.nFixings-1) + S)/self.nFixings) for a in self.As]
        else:
            if self.onFixingDate(t):
                i = len(filter(lambda x: x < t, self.fixings)) # number of previous fixings
                Ahats = [(a*(i-1) + S)/i for a in self.As]
                nodeValues = [np.interp(a, self.As, continuation) for a in Ahats]
            else:
                nodeValues = continuation
        return nodeValues

# Pricing American Products with Tree
def crrBinomialG(S, r, vol, trade, n):
    t = trade.expiry / n
    b = math.exp(vol * vol * t+r*t) + math.exp(-r * t)
    u = (b + math.sqrt(b*b - 4)) / 2
    p = (math.exp(r * t) - (1/u)) / (u - 1/u)
    # d = 1 / u
    # set up the last time slice, there are n+1 nodes at the last time slice
    vs = [trade.payoff( S * u**(n-i-i)) for i in range(n+1)]
    # iterate backward
    for i in range(n-1, -1, -1):
        # calculate the value of each node at time slide i, there are i nodes
        for j in range(i+1):
            nodeS = S * u**(i-j-j)
            continuation = math.exp(-r * t) * (vs[j] * p + vs[j+1] * (1-p))
            vs[j] = trade.valueAtNode(t*i, nodeS, continuation)
    return vs[0]

def crrBinomialGAmeEur():
# testing binomial tree pricer with American Option
    euroPrc, amerPrc = [],  []
    S, r, vol = 100, 0.05, 0.2
    ks = range(50, 150)
    for k in ks:
        euroPrc.append(crrBinomialG(S, r, vol, EuropeanOption(1, float(k), PayoffType.Call), 300))
        amerPrc.append(crrBinomialG(S, r, vol, AmericanOption(1, float(k), PayoffType.Call), 300))
    plt.plot(ks, euroPrc, 'r', label = 'euroCall')
    plt.plot(ks, amerPrc, 'g', label = 'amerCall')
    euroPrc, amerPrc = [],  []
    for k in ks:
        euroPrc.append(crrBinomialG(S, r, vol, EuropeanOption(1, float(k), PayoffType.Put), 300))
        amerPrc.append(crrBinomialG(S, r, vol, AmericanOption(1, float(k), PayoffType.Put), 300))
    plt.plot(ks, euroPrc, 'y', label = 'euroPut')
    plt.plot(ks, amerPrc, 'b', label = 'euroPut')
    plt.legend()
    plt.savefig('../figs/amerPrice.eps', format='eps')

if __name__ == "__main__":
    crrBinomialGAmeEur()
# questions: 1. why American Call is the same as Europen Call , for non-dividend paying stock? 
# 2. Test how the graph looks like when r=0.01, try to explain why ? 

# Generalize the Payoff Function 
class EuropeanPayoff():
    def __init__(self, expiry, payoffFun):
        self.expiry = expiry
        self.payoffFun = payoffFun
    def payoff(self, S):
        return self.payoffFun(S)
    def valueAtNode(self, t, S, continuation):
        return continuation

class AmericanPayoff():
    def __init__(self, expiry, payoffFun):
        self.expiry = expiry
        self.payoffFun = payoffFun
    def payoff(self, S):
        return self.payoffFun(S)
    def valueAtNode(self, t, S, continuation):
        return max(self.payoff(S), continuation)
    
def testAmerSpreadCRR():
    S, r, vol = 100, 0.05, 0.2
    callSpread = lambda S: min(max(S - 90, 0), 10)
    plt.plot(range(80, 120), [callSpread(i) for i in range(80, 120)])
    plt.show()
    print("Euro callspread: ", crrBinomialG(S, r, vol, EuropeanPayoff(1, callSpread), 300))
    print("Amer callspread: ", crrBinomialG(S, r, vol, AmericanPayoff(1, callSpread), 300))

if __name__ == "__main__":
    testAmerSpreadCRR()

def testUpBarrier():
#Testing Barrier Option - Upper Barrier
# varying up barrier
    S, r, vol , k = 100, 0.05, 0.2, 105
    eurOpt = EuropeanOption(1, k, PayoffType.Put)
    euroPrc = crrBinomialG(S, r, vol, eurOpt, 300)
    barrierPrc, ks = [], range(50, 150)
    for barrierLevel in ks:
        prc = crrBinomialG(S, r, vol, BarrierOption(barrierStart = 0, barrierEnd = 1.0, downBarrier = None,
            upBarrier = barrierLevel, underlyingOption = eurOpt), n = 300)
        barrierPrc.append(prc)
    plt.hlines(euroPrc, ks[0], ks[-1], label = "euroPrc")
    plt.plot(ks, barrierPrc, "g", label="barrierPrc")
    plt.xlabel("up barrier level"); plt.legend(); plt.savefig("../figs/upKO.eps", format="eps")

if __name__ == "__main__":
    testUpBarrier()

def testDownBarrier():
    #Testing Barrier Option - Down Barrier
    # varying down barrier
    S, r, vol , k = 100, 0.05, 0.2, 105
    eurOpt = EuropeanOption(1, k, PayoffType.Put)
    euroPrc = crrBinomialG(S, r, vol, eurOpt, 300)
    barrierPrc, ks = [], range(30, 130)
    for barrierLevel in ks:
        prc = crrBinomialG(S, r, vol, BarrierOption(barrierStart = 0, barrierEnd = 1.0, downBarrier = 
            barrierLevel, upBarrier = None, underlyingOption = eurOpt), n = 300)
        barrierPrc.append(prc)
    plt.hlines(euroPrc, ks[0], ks[-1], label = "euroPrc")
    plt.plot(ks, barrierPrc, "g", label="barrierPrc")
    plt.xlabel("down barrier level"); plt.legend(); plt.savefig("../figs/downKO.eps", format="eps")
if __name__ == "__main__":
    testDownBarrier()

def testWiondowBarrier():
#Testing Barrier Options - Wiondow Barrier
# varying barrier window, barrier end
    S, r, vol , k = 100, 0.05, 0.2, 105
    eurOpt = EuropeanOption(1, k, PayoffType.Put)
    euroPrc = crrBinomialG(S, r, vol, eurOpt, 300)
    barrierPrc, ks = [], range(0, 100)
    for t in ks:
        prc = crrBinomialG(S, r, vol, BarrierOption(barrierStart = 0, barrierEnd = t/100.0, downBarrier = 
            80, upBarrier = 150, underlyingOption = eurOpt), n = 300)
        barrierPrc.append(prc)
    plt.hlines(euroPrc, ks[0], ks[-1], label = "euroPrc")
    plt.plot(ks, barrierPrc, "g", label="barrierPrc")
    plt.xlabel("window end"); plt.legend(); plt.savefig("../figs/winBarrier.eps", format="eps")
if __name__ == "__main__":
    testWiondowBarrier()

def testWiondowBarrier2():
    #Testing Barrier Options - Wiondow Barrier
    # varying barrier window, barrier start
    S, r, vol , k = 100, 0.05, 0.2, 105
    eurOpt = EuropeanOption(1, k, PayoffType.Put)
    euroPrc = crrBinomialG(S, r, vol, eurOpt, 300)
    barrierPrc, ks = [], range(0, 100)
    for t in ks:
        prc = crrBinomialG(S, r, vol, BarrierOption(barrierStart = t/100, barrierEnd = 1.0, downBarrier = 
            80, upBarrier = 150, underlyingOption = eurOpt), n = 300)
        barrierPrc.append(prc)
    plt.hlines(euroPrc, ks[0], ks[-1], label = "euroPrc")
    plt.plot(ks, barrierPrc, "g", label="barrierPrc")
    plt.xlabel("window start"); plt.legend(); plt.savefig("../figs/winBarrierStart.eps", format="eps")

if __name__ == "__main__":
    testWiondowBarrier2()

############ binomial pricer and different binomial models
def simpleCRR(r, vol, t):
    u = math.exp(vol * math.sqrt(t))
    p = (math.exp(r * t) - (1 / u)) / (u - 1 / u)
    return (u, 1 / u, p)

def crrCalib(r, vol, t):
    b = math.exp(vol * vol * t + r * t) + math.exp(-r * t)
    u = (b + math.sqrt(b * b - 4)) / 2
    p = (math.exp(r * t) - (1 / u)) / (u - 1 / u)
    return (u, 1/u, p)

def jrrnCalib(r, vol, t):
    u = math.exp((r - vol * vol / 2) * t + vol * math.sqrt(t))
    d = math.exp((r - vol * vol / 2) * t - vol * math.sqrt(t))
    p = (math.exp(r * t) - d) / (u - d)
    return (u, d, p)

def jreqCalib(r, vol, t):
    u = math.exp((r - vol * vol / 2) * t + vol * math.sqrt(t))
    d = math.exp((r - vol * vol / 2) * t - vol * math.sqrt(t))
    return (u, d, 1/2)

def tianCalib(r, vol, t):
    v = math.exp(vol * vol * t)
    u = 0.5 * math.exp(r * t) * v * (v + 1 + math.sqrt(v*v + 2*v - 3))
    d = 0.5 * math.exp(r * t) * v * (v + 1 - math.sqrt(v*v + 2*v - 3))
    p = (math.exp(r * t) - d) / (u - d)
    return (u, d, p)

# Generical Binomial tree for various model defined in Calib 
def binomialPricer(S, r, vol, trade, n, calib):
    t = trade.expiry / n
    (u, d, p) = calib(r, vol, t)
    # set up the last time slice, there are n+1 nodes at the last time slice
    vs = [trade.payoff(S * u ** (n - i) * d ** i) for i in range(n + 1)]
    # iterate backward
    for i in range(n - 1, -1, -1):
        # calculate the value of each node at time slide i, there are i nodes
        for j in range(i + 1):
            nodeS = S * u ** (i - j) * d ** j
            continuation = math.exp(-r * t) * (vs[j] * p + vs[j + 1] * (1 - p))
            vs[j] = trade.valueAtNode(t * i, nodeS, continuation)
    return vs[0]

def test1DTiming():
    opt = EuropeanOption(1, 105, PayoffType.Call)
    S, r, vol = 100, 0.01, 0.2

    bsprc = bsPrice(S, r, vol, opt.expiry, opt.strike, opt.payoffType)
    print("bsPrice = \t ", bsprc)
    n = 20
    prc = [None] * n
    timing = [None] * n
    nSteps = [None] * n
    for i in range(1, n + 1):
        nSteps[i - 1] = 20 * i
        start = time.time()
        prc[i - 1] = binomialPricer(S, r, vol, opt, nSteps[i - 1], tianCalib) - bsprc
        timing[i - 1] = time.time() - start

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(nSteps, prc, 'g')
    ax2.plot(nSteps, timing, 'b')

    ax1.set_xlabel('nTreeSteps')
    ax1.set_ylabel('Pricing Error')
    ax2.set_ylabel('Timeing')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test1DTiming()

def testBTrees():
    opt = EuropeanOption(1, 105, PayoffType.Call)
    S, r, vol = 100, 0.01, 0.2

    bsprc = bsPrice(S, r, vol, opt.expiry, opt.strike, opt.payoffType)
    print("bsPrice = \t ", bsprc)
    n = 300

    crrErrs = [math.log(abs(binomialPricer(S, r, vol, opt, i, crrCalib) - bsprc)) for i in range(1, n)]
    jrrnErrs = [math.log(abs(binomialPricer(S, r, vol, opt, i, jrrnCalib) - bsprc)) for i in range(1, n)]
    jreqErrs = [math.log(abs(binomialPricer(S, r, vol, opt, i, jreqCalib) - bsprc)) for i in range(1, n)]
    tianErrs = [math.log(abs(binomialPricer(S, r, vol, opt, i, tianCalib) - bsprc)) for i in range(1, n)]

    plt.plot(range(1, n), crrErrs, label="crr")
    plt.plot(range(1, n), jrrnErrs, label="jrrn")
    plt.plot(range(1, n), jreqErrs, label="jreq")
    plt.plot(range(1, n), tianErrs, label="tian")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    testBTrees()


