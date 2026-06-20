from CayleyKernel import multiply
import cupy as cp

def conj(x):
     xstar = -x
     xstar[0] *= -1
     return xstar

def mult(x, y):
    n = len(x)
    if n == 1:
        return x*y
    m = n // 2
    a, b = x[:m], x[m:]
    c, d = y[:m], y[m:]
    z = cp.zeros(n)
    z[:m] = mult(a, c) - mult(conj(d), b)
    z[m:] = mult(d, a) + mult(b, conj(c))
    return z

parallel = multiply
recursive = mult

for level in range(6+1):
    dim = 2**level

    x = cp.random.rand(dim, dtype=cp.float32)
    y = cp.random.rand(dim, dtype=cp.float32)

    print('x:',x)
    print('y:',y)
    print('x*y:',parallel(x,y))
    print('x*y:',recursive(x,y))

