import math
import matplotlib.pyplot as plt
import numpy as np

gr = (math.sqrt(5) + 1) / 2

def gss(f, a, b, tol=1e-5):
    """Golden section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]

    Example:
    >>> f = lambda x: (x-2)**2
    >>> x = gss(f, 1, 5)
    >>> print("%.15f" % x)
    2.000009644875678

    """
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    return (b + a) / 2


f = lambda x : x*x / 10 - 2 * math.sin(x)

xs = np.arange(0.0, 4.0, 0.1)
ys = [f(x) for x in xs]
plt.plot(xs, ys)
plt.savefig('../figs/gssExample.eps', format = 'eps')
plt.show()