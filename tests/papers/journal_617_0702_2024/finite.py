"""Exact finite-field CD/Zorn arithmetic; no call to the CUDA multiply API.

The standard model uses CayleyKernel.basis and twist. Split models
use an independent parameterized doubling oracle, never an unsupported
signature argument to CayleyKernel. Tuple indices are mathematical e_i.
"""
from functools import lru_cache
from itertools import product
import numpy as np
import CayleyKernel as CK


def e(i,d):
    return tuple(int(j==i) for j in range(d))


def add(a,b,p):return tuple((x+y)%p for x,y in zip(a,b))
def scale(a,c,p):return tuple(x*c%p for x in a)
def conj(a,p):return (a[0],)+tuple(-x%p for x in a[1:])


@lru_cache(None)
def recursive_sign(i,j,gammas):
    if not gammas:return 1
    h=1<<(len(gammas)-1);a,b=i%h,j%h
    if i<h and j<h:return recursive_sign(a,b,gammas[:-1])
    if i<h<=j:return recursive_sign(b,a,gammas[:-1])
    if j<h<=i:return (-1 if b else 1)*recursive_sign(a,b,gammas[:-1])
    return gammas[-1]*(-1 if b else 1)*recursive_sign(b,a,gammas[:-1])


def parameters(d,split=False):
    n=d.bit_length()-1
    if d<2 or d!=1<<n:raise ValueError('dimension must be a power of two >=2')
    return (-1,)*(n-1)+((1,) if split else (-1,))


@lru_cache(None)
def basis_table(d,split=False,reference=False):
    gammas=parameters(d,split)
    rows=[]
    for i in range(d):
        row=[]
        for j in range(d):
            if split or reference:label=recursive_sign(i,j,gammas)*((i^j)+1)
            else:label=CK.basis(i+1,j+1)
            row.append(label)
        rows.append(tuple(row))
    return tuple(rows)


def mul(a,b,p,split=False,reference=False):
    if len(a)!=len(b):raise ValueError('dimension mismatch')
    d=len(a);table=basis_table(d,split,reference);out=[0]*d
    for i,x in enumerate(a):
        if x:
            for j,y in enumerate(b):
                if y:
                    label=table[i][j];out[abs(label)-1]+=(1 if label>0 else -1)*x*y
    return tuple(x%p for x in out)


def batch_product(a,b,p,split=False):
    """CPU integer bilinear expansion, explicitly separate from CK.multiply().

    Test inputs have p<=37,d<=64, so int64 accumulators are far from overflow.
    Reduction is after each complete product; this is valid over F_p.
    """
    a=np.asarray(a,dtype=np.int64);b=np.asarray(b,dtype=np.int64)
    a,b=np.broadcast_arrays(a,b);d=a.shape[-1]
    out=np.zeros_like(a);table=basis_table(d,split)
    for i in range(d):
        for j in range(d):
            label=table[i][j]
            out[...,abs(label)-1]+=(1 if label>0 else -1)*a[...,i]*b[...,j]
    return out%p


def norm_weights(d,split=False):
    table=basis_table(d,split)
    return (1,)+tuple(-1 if table[i][i]>0 else 1 for i in range(1,d))


def norm(a,p,split=False):
    return sum(w*x*x for w,x in zip(norm_weights(len(a),split),a))%p


def trace(a,p):return 2*a[0]%p


def chunks(p,d,size=8192):
    for start in range(0,p**d,size):
        ids=np.arange(start,min(start+size,p**d),dtype=np.int64)
        yield np.stack([(ids//(p**j))%p for j in range(d)],axis=-1)


@lru_cache(None)
def enumerate_counts(p,d,split=False):
    """Complete squares/conjugate products, not formula-generated counts.

    Nilpotents are counted by x^2=0. For the odd-characteristic quadratic
    CD models this is the whole nilpotent locus; the tests check the quadratic
    identity and explain the power argument. Unit/ZD counts below are reported
    only for composition dimensions <=8.
    """
    totals={'total':p**d,'idempotents':0,'square_zero_including_zero':0,'norm_zero_including_zero':0}
    weights=np.asarray(norm_weights(d,split))
    for x in chunks(p,d):
        square=batch_product(x,x,p,split)
        totals['idempotents']+=int(np.count_nonzero(np.all(square==x,axis=-1)))
        totals['square_zero_including_zero']+=int(np.count_nonzero(np.all(square==0,axis=-1)))
        N=np.sum(weights*x*x,axis=-1)%p
        xc=-x;xc[:,0]=x[:,0]
        expected=np.zeros_like(x);expected[:,0]=N
        if not np.array_equal(batch_product(x,xc,p,split),expected):raise AssertionError('norm identity')
        quadratic=(2*x[:,0,None]*x-expected)%p
        if not np.array_equal(square,quadratic):raise AssertionError('quadratic identity')
        totals['norm_zero_including_zero']+=int(np.count_nonzero(N==0))
    if d<=8:
        totals['nonzero_zero_divisors']=totals['norm_zero_including_zero']-1
        totals['units']=p**d-totals['norm_zero_including_zero']
    return totals


def quadratic_histogram(weights,p):
    """Independent count by convolution of one-variable square distributions."""
    hist=[1]+[0]*(p-1)
    for weight in weights:
        new=[0]*p
        for residue,count in enumerate(hist):
            for x in range(p):new[(residue+weight*x*x)%p]+=count
        hist=new
    return hist


def dot(v,w):return sum(x*y for x,y in zip(v,w))
def cross(v,w):
    return (v[1]*w[2]-v[2]*w[1],v[2]*w[0]-v[0]*w[2],v[0]*w[1]-v[1]*w[0])


def zorn_mul(x,y,p):
    # Ordering (a,v1,v2,v3,w1,w2,w3,b); signs as printed on page3.
    a,v,w,b=x[0],x[1:4],x[4:7],x[7]
    c,t,u,d=y[0],y[1:4],y[4:7],y[7]
    ww=cross(w,u);vv=cross(v,t)
    return tuple(q%p for q in (a*c+dot(v,u),
        *(a*t[i]+d*v[i]+ww[i] for i in range(3)),
        *(c*w[i]+b*u[i]-vv[i] for i in range(3)),b*d+dot(t,w)))


def zorn_norm(x,p):return (x[0]*x[7]-dot(x[1:4],x[4:7]))%p


def split_to_zorn(x,p):
    return tuple(a%p for a in (x[0]+x[4],*(x[i]-x[i+4] for i in (1,2,3)),
                              *(-x[i]-x[i+4] for i in (1,2,3)),x[0]-x[4]))


def zorn_to_split(z,p):
    inv2=pow(2,-1,p)
    return tuple(x*inv2%p for x in (z[0]+z[7],*(z[i]-z[i+3] for i in (1,2,3)),
                                    z[0]-z[7],*(-z[i]-z[i+3] for i in (1,2,3))))


@lru_cache(None)
def norm_minus_one_quaternion(p):
    return next(x for x in product(range(p),repeat=4) if norm(x,p)==p-1)


def main_to_split(x,p):
    """Explicit finite-field O -> split O: (a,b) -> (a,u*b), N(u)=-1."""
    return tuple(x[:4])+mul(norm_minus_one_quaternion(p),x[4:],p)


def split_to_main(x,p):
    u=norm_minus_one_quaternion(p)
    inverse=scale(conj(u,p),-1,p)
    return tuple(x[:4])+mul(inverse,x[4:],p)


def modular_rref(rows,p,width):
    rows=[[x%p for x in row] for row in rows];pivots=[];cursor=0
    for c in range(width):
        found=next((i for i in range(cursor,len(rows)) if rows[i][c]),None)
        if found is None:continue
        rows[cursor],rows[found]=rows[found],rows[cursor]
        rows[cursor]=[x*pow(rows[cursor][c],-1,p)%p for x in rows[cursor]]
        for i in range(len(rows)):
            if i!=cursor and rows[i][c]:
                factor=rows[i][c];rows[i]=[(x-factor*y)%p for x,y in zip(rows[i],rows[cursor])]
        pivots.append(c);cursor+=1
        if cursor==len(rows):break
    return rows,pivots


def annihilator(a,p,left=False,split=False):
    d=len(a)
    columns=[mul(e(j,d),a,p,split) if left else mul(a,e(j,d),p,split) for j in range(d)]
    rows=[list(row) for row in zip(*columns)]
    reduced,pivots=modular_rref(rows,p,d);basis=[]
    for free in range(d):
        if free in pivots:continue
        v=[0]*d;v[free]=1
        for row,pivot in zip(reduced,pivots):v[pivot]=-row[free]%p
        basis.append(tuple(v))
    return basis


def paper_id(p,d):return p**(d-2)+p**(d//2-1)+2
def paper_nil(p,d):return p**(d-2)
def paper_zd(p,d):return p**(d-1)+p**(d//2)+p**(d//2-1)
def paper_units(p,d):return p**d-paper_zd(p,d)
def correct_norm_zero(p,d):return p**(d-1)+p**(d//2)-p**(d//2-1)
def correct_intersection(p):return p*p*(p**3+p*p-p)
