import math
from enum import Enum
import numpy as np

import matplotlib.pyplot as plt

from binomial import *

function = lambda x: math.exp(x)

def functionderiv(x):
    return math.exp(x) 

function2 = lambda x: x* math.exp(x)

def function2deriv(x):
    return x* math.exp(x) +  math.exp(x)

def oneSide(func, x, h):
    return (func(x+h) - func(x))/ h

def centralDiff(func, x, h):
    return (func(x+h) - func(x-h))/(2*h)

def richardsonOneSide(func, x, h):
    return (4*oneSide(func,x,h/2) - oneSide(func,x,h))/3.0
def richardsonCentral(func, x, h):
    return (4*centralDiff(func,x,h/2) - centralDiff(func,x,h))/3.0

def testfinDiff(func):
    x0 = 1.0 #1
    deriv = functionderiv(x0)
    h = 0.2 #1
    n = 60
    hs = [h] * n
    errOneSide = [0] * n
    errCentralDiff = [0] * n
    errRichardsonOneSide = [0] * n
    errRichardsonCentralDiff = [0] * n
    for i in range(0, n):
        finDiffOneSide = oneSide(func, x0, h) 
        finDiffCentral = centralDiff(func, x0, h) 
        finDiffRichardsonOneSide = richardsonOneSide(func, x0, h) 
        finDiffRichardsonCentralDiff = richardsonCentral(func, x0, h) 
        hs[i] = h
        h = h/2
        errOneSide[i] = abs(finDiffOneSide - deriv)
        errCentralDiff[i] = abs(finDiffCentral - deriv)
        errRichardsonOneSide[i] = abs(finDiffRichardsonOneSide - deriv)
        errRichardsonCentralDiff[i] = abs(finDiffRichardsonCentralDiff - deriv)
        
    plt.plot(hs, errOneSide, label="finDiff error One Side")
    plt.plot(hs, errCentralDiff, label="finDiff error Central")
    plt.plot(hs, errRichardsonOneSide, label="Richardson error One Side")
    plt.plot(hs, errRichardsonCentralDiff, label="Richardson error Central Diff")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()


class BoundaryType(Enum):
    Dirichlet = 0
    Neumann = 1
    Linear = 2


def pdeExplicitPricer(S0, r, q, vol, NS, NT, bTy, trade):
    # set up pde grid
    mu = r - q
    T = trade.expiry
    srange = 5 * vol * math.sqrt(T)
    maxS = S0 * math.exp((mu - vol * vol * 0.5)*T + srange)
    minS = S0 * math.exp((mu - vol * vol * 0.5)*T - srange)
    dt = T / (NT-1)
    ds = (maxS - minS) / (NS-1)
    # set up spot grid
    sGrid = np.array([minS + i*ds for i in range(NS)])
    # initialize the payoff
    ps = np.array([trade.payoff(s) for s in sGrid])
    # set up the matrix, for BS the matrix does not change
    # for LV we need to update it for each iteration
    a, b = mu/2.0/ds, vol * vol / ds / ds
    M = np.zeros((NS, NS))
    D = np.zeros((NS, NS))
    for i in range (1, NS-1):
        M[i, i-1] = a*sGrid[i] - b*sGrid[i]*sGrid[i]/2.0
        M[i, i] = r + b * sGrid[i] * sGrid[i]
        M[i, i+1] = -a * sGrid[i] - b*sGrid[i]*sGrid[i]/2.0
        D[i,i] = 1.0
    # the first row and last row depends on the boundary condition
    if bTy == BoundaryType.Dirichlet:
        M[0,0], M[NS-1, NS-1] = 1.0, 1.0
    elif bTy == BoundaryType.Neumann:
        M[0,0], M[0,1] = -1.0, 1.0
        M[NS-1, NS-2], M[NS-1, NS-1] = 1.0
    elif bTy == BoundaryType.Linear:
        M[0,0], M[0,1], M[0,2] = 1, -2, 1
        M[NS-1, NS-3], M[NS-1, NS-2], M[NS-1, NS-1] = 1, -2, 1
    else:
        raise Exception("boundary condition not supported: ", bTy)

    M = D - dt * M
    # backward induction
    for j in range(1, NT):
        ps = M.dot(ps)  # Euler explicit
        if bTy == BoundaryType.Dirichlet:
            ps[0] = math.exp(-r*j*dt) * trade.payoff(sGrid[0]) # discounted payoff
            ps[NS-1] = math.exp(-r*j*dt) * trade.payoff(sGrid[NS-1])
        else:
            raise Exception("boundary type not implemented: ", bTy)
    # linear interpolate the price at S0
    return np.interp(S0, sGrid, ps)

def pdeImplicitPricer(S0, r, q, vol, NS, NT, bTy, trade):
    # set up pde grid
    mu = r - q
    T = trade.expiry
    srange = 5 * vol * math.sqrt(T)
    maxS = S0 * math.exp((mu - vol * vol * 0.5)*T + srange)
    minS = S0 * math.exp((mu - vol * vol * 0.5)*T - srange)
    dt = T / (NT-1)
    ds = (maxS - minS) / (NS-1)
    # set up spot grid
    sGrid = np.array([minS + i*ds for i in range(NS)])
    # initialize the payoff
    ps = np.array([trade.payoff(s) for s in sGrid])
    # set up the matrix, for BS the matrix does not change
    # for LV we need to update it for each iteration
    a, b = mu/2.0/ds, vol * vol / ds / ds
    M = np.zeros((NS, NS))
    D = np.zeros((NS, NS))
    for i in range (1, NS-1):
        M[i, i-1] = a*sGrid[i] - b*sGrid[i]*sGrid[i]/2.0
        M[i, i] = r + b * sGrid[i] * sGrid[i]
        M[i, i+1] = -a * sGrid[i] - b*sGrid[i]*sGrid[i]/2.0
        D[i,i] = 1.0
    # the first row and last row depends on the boundary condition
    if bTy == BoundaryType.Dirichlet:
        M[0,0], M[NS-1, NS-1] = 1.0, 1.0
    elif bTy == BoundaryType.Neumann:
        M[0,0], M[0,1] = -1.0, 1.0
        M[NS-1, NS-2], M[NS-1, NS-1] = 1.0
    elif bTy == BoundaryType.Linear:
        M[0,0], M[0,1], M[0,2] = 1, -2, 1
        M[NS-1, NS-3], M[NS-1, NS-2], M[NS-1, NS-1] = 1, -2, 1
    else:
        raise Exception("boundary condition not supported: ", bTy)

    M = D + dt * M
    M = np.linalg.inv(M)
    # backward induction
    for j in range(1, NT):
        if bTy == BoundaryType.Dirichlet:
            ps[0] = dt*math.exp(-r*j*dt) * trade.payoff(sGrid[0]) # discounted payoff
            ps[NS-1] = dt*math.exp(-r*j*dt) * trade.payoff(sGrid[NS-1])
        else:
            raise Exception("boundary type not implemented: ", bTy)
        ps = M.dot(ps)  # Euler implicit
    # linear interpolate the price at S0
    return np.interp(S0, sGrid, ps)


def pdeDouglasPricer(S0, r, q, vol, NS, NT, w, bTy, trade):
    # set up pde grid
    mu = r - q
    T = trade.expiry
    srange = 5 * vol * math.sqrt(T)
    maxS = S0 * math.exp((mu - vol * vol * 0.5)*T + srange)
    minS = S0 * math.exp((mu - vol * vol * 0.5)*T - srange)
    dt = T / (NT-1)
    ds = (maxS - minS) / (NS-1)
    # set up spot grid
    sGrid = np.array([minS + i*ds for i in range(NS)])
    # initialize the payoff
    ps = np.array([trade.payoff(s) for s in sGrid])
    # set up the matrix, for BS the matrix does not change
    # for LV we need to update it for each iteration
    a, b = mu/2.0/ds, vol * vol / ds / ds
    M = np.zeros((NS, NS))
    D = np.zeros((NS, NS))
    for i in range (1, NS-1):
        M[i, i-1] = a*sGrid[i] - b*sGrid[i]*sGrid[i]/2.0
        M[i, i] = r + b * sGrid[i] * sGrid[i]
        M[i, i+1] = -a * sGrid[i] - b*sGrid[i]*sGrid[i]/2.0
        D[i,i] = 1.0
    # the first row and last row depends on the boundary condition
    if bTy == BoundaryType.Dirichlet:
        M[0,0], M[NS-1, NS-1] = 1.0, 1.0
    elif bTy == BoundaryType.Neumann:
        M[0,0], M[0,1] = -1.0, 1.0
        M[NS-1, NS-2], M[NS-1, NS-1] = 1.0
    elif bTy == BoundaryType.Linear:
        M[0,0], M[0,1], M[0,2] = 1, -2, 1
        M[NS-1, NS-3], M[NS-1, NS-2], M[NS-1, NS-1] = 1, -2, 1
    else:
        raise Exception("boundary condition not supported: ", bTy)

    rhsM = (D - dt * M) * w + (1-w) * np.identity(NS)
    lhsM = w * np.identity(NS) + (D + dt * M) * (1-w)
    inv = np.linalg.inv(lhsM)
    # backward induction
    for j in range(1, NT):
        ps = rhsM.dot(ps)
        if bTy == BoundaryType.Dirichlet:
            ps[0] = dt*math.exp(-r*j*dt) * trade.payoff(sGrid[0]) # discounted payoff
            ps[NS-1] = dt*math.exp(-r*j*dt) * trade.payoff(sGrid[NS-1])
        else:
            raise Exception("boundary type not implemented: ", bTy)
        ps = inv.dot(ps)
    # linear interpolate the price at S0
    return np.interp(S0, sGrid, ps)

def pdePricer(S0, r, q, vol, NX, NT, w, trade):
    # set up pde grid
    mu = r - q
    T = trade.expiry
    X0 = math.log(S0)
    srange = 5 * vol * math.sqrt(T)
    maxX = X0 + (mu - vol * vol * 0.5)*T + srange
    minX = X0 - (mu - vol * vol * 0.5)*T - srange
    dt = T / (NT-1)
    dx = (maxX - minX) / (NX-1)
    # set up spot grid
    xGrid = np.array([minX + i*dx for i in range(NX)])
    # initialize the payoff
    ps = np.array([trade.payoff(math.exp(x)) for x in xGrid])
    # set up the matrix, for BS the matrix does not change
    # for LV we need to update it for each iteration
    a = (mu - vol*vol/2.0)/2.0/dx - vol*vol/2/dx/dx
    b = r + vol * vol / dx / dx
    c = -(mu - vol*vol/2.0)/2.0/dx - vol*vol/2/dx/dx
    M = np.zeros((NX, NX))
    D = np.zeros((NX, NX))
    for i in range (1, NX-1):
        M[i,i-1] = a
        M[i, i] = b
        M[i, i+1] = c
        D[i,i] = 1.0
    # the first row and last row depends on the boundary condition
    M[0,0], M[NX-1, NX-1] = 1.0, 1.0
    rhsM = (D - dt * M) * w + (1-w) * np.identity(NX)
    lhsM = w * np.identity(NX) + (D + dt * M) * (1-w)
    inv = np.linalg.inv(lhsM)
    # backward induction
    for j in range(1, NT):
        ps = rhsM.dot(ps)
        ps[0] = dt*math.exp(-r*j*dt) * trade.payoff(math.exp(xGrid[0])) # discounted payoff
        ps[NX-1] = dt*math.exp(-r*j*dt) * trade.payoff(math.exp(xGrid[NX-1]))
        ps = inv.dot(ps)
    # linear interpolate the price at S0
    return np.interp(X0, xGrid, ps)

def testPDE():
    opt = EuropeanOption(1, 101, PayoffType.Call)
    S, r, vol = 100, 0.01, 0.2

    bsprc = bsPrice(S, r, vol, opt.expiry, opt.strike, opt.payoffType)
    print("bsPrice = \t ", bsprc)
    n = 400

    pdeErrsE = [(abs(pdeExplicitPricer(S, r, 0.0, vol, int(math.sqrt(i/vol)), i, BoundaryType.Dirichlet, opt) - bsprc)) for i in range(10, n, 2)]
    pdeErrsI = [(abs(pdeImplicitPricer(S, r, 0.0, vol, i, i, BoundaryType.Dirichlet, opt) - bsprc)) for i in range(10, n, 2)]
    pdeCN = [(abs(pdeDouglasPricer(S, r, 0.0, vol, i, i, 0.5, BoundaryType.Dirichlet, opt) - bsprc)) for i in range(10, n, 2)]
    pdeDG = [(abs(pdeDouglasPricer(S, r, 0.0, vol, i, i, 0.5 - 1/12/i, BoundaryType.Dirichlet, opt) - bsprc)) for i in
             range(10, n, 2)]
    pdeLogSCN = [(abs(pdePricer(S, r, 0.0, vol, i, i, 0.5, opt) - bsprc)) for i in range(10, n, 2)]
    crrErrs = [(abs(binomialPricer(S, r, vol, opt, i, crrCalib) - bsprc)) for i in range(10, n, 2)]
    # jrrnErrs = [math.log(abs(binomialPricer(S, r, vol, opt, i, jrrnCalib) - bsprc)) for i in range(1, n)]
    # jreqErrs = [math.log(abs(binomialPricer(S, r, vol, opt, i, jreqCalib) - bsprc)) for i in range(1, n)]
    # tianErrs = [math.log(abs(binomialPricer(S, r, vol, opt, i, tianCalib) - bsprc)) for i in range(1, n)]

    plt.plot(range(10, n, 2), crrErrs, label="crr binomial")
    plt.plot(range(10, n, 2), pdeErrsE, label="pde explicit")
    plt.plot(range(10, n, 2), pdeErrsI, label="pde implicit")
    plt.plot(range(10, n, 2), pdeCN, label="pde crank-nicholson")
    plt.plot(range(10, n, 2), pdeLogSCN, label="pde logS")
    plt.plot(range(10, n, 2), pdeDG, label="pde douglas 1/2 - 1/12/i")
    # plt.plot(range(1, n), jrrnErrs, label="jrrn")
    # plt.plot(range(1, n), jreqErrs, label="jreq")
    # plt.plot(range(1, n), tianErrs, label="tian")
    plt.yscale('log')
    plt.xlabel('NT'), plt.ylabel('Error')
    plt.legend()
    plt.savefig('../figs/pdeCKError.eps', format='eps')
    plt.show()


testfinDiff(function)

testPDE()

# To Do: 
# calculate delta, gamma and vega from pde
# implement various boundary conditions for different kind of options
# implement logS pde, check error
# Price American/Bermudan opiton 