###### computational cost ####################
import timeit

def opTiming(op, opName, repeat):
    elapsed_time = timeit.timeit(op, setup='import math', number=repeat)
    print(opName, "\t", elapsed_time / repeat)

repeat = int(1e8)
opTiming("x = 5.0 + 7.0", "add", repeat)
opTiming("x = 5.0 * 7.0", "mul", repeat)
opTiming("x = 5.0 / 7.0", "div", repeat)
opTiming("x = math.log(7.0)", "log", repeat)
opTiming("x = math.exp(7.0)", "exp", repeat)
opTiming("x = math.sqrt(7.0)", "sqrt", repeat)

m1 = """
S = 100;K = 105;vol = 0.1;t=2;mu=0.01
d1 = (math.log(S * math.exp(mu*t) / K) + vol * vol * t / 2) / vol / math.sqrt(t)
"""
m2 = """
S = 100;K = 105;vol = 0.1;t=2;mu=0.01
stdev = vol * math.sqrt(t)
d1 = (math.log(S / K) + mu*t) / stdev + stdev / 2
"""
repeat = int(1e7)
opTiming(m1, 'm1', repeat)
opTiming(m2, 'm2', repeat)

###### fixed point representation ####################
def toFixedPoint(x : float, w : int, b : int) -> [int]:
    # set a[w-1] to 1 if x < 0, otherwise set a[w-1] to 0
    a = [0 for i in range(w)]
    if x < 0:
        a[0] = 1
        x += 2**(w-1-b)
    for i in range(1, w):
        y = x / (2**(w-1-i-b))
        a[i] = int(y)  # round y down to integer
        x -= a[i] * (2**(w-1-i-b))
    return a

print(toFixedPoint(-10, 8, 1))
print(toFixedPoint(-9.5, 8, 1))
print(toFixedPoint(9.25, 8, 2))

print(toFixedPoint(20, 8, 3))
print(toFixedPoint(20, 9, 3))

def toFixedPoint2(x : float, w : int, b : int) -> [int]:
    # set a[w-1] to 1 if x < 0, otherwise set a[w-1] to 0
    a = [0 for i in range(w)]
    if x < 0:
        a[0] = 1
        x += 2**(w-1-b)
    for i in range(1, w):
        y = x / (2**(w-1-i-b))
        if int(y) > 1:
            raise OverflowError('fixed<' + str(w) + "," + str(b) + "> is not sufficient to represent " + str(x))
        a[i] = int(y) # % 2  # round y down to integer
        x -= a[i] * (2**(w-1-i-b))
    return a

print(toFixedPoint2(20, 8, 3))


########### floating point representation #####################
import numpy as np
for f in (np.float32, np.float64, float):
    finfo = np.finfo(f)
    print(finfo.dtype, "\t exponent bits = ", finfo.nexp, "\t significand bits = ", finfo.nmant)

### rounding error and machine epsilon ###################
x = 10776321
nsteps = 1200
s = x / nsteps
y = 0
for i in range(nsteps):
    y += s
print(x - y)

x = 10.56
print(x == x + 5e-16)

x = 0.1234567891234567890
y = 0.1234567891
scale = 1e16
z1 = (x-y) * scale
print("z1 = ", z1)

z2 = (x*scale - y*scale)
print("z2 = ", z2)