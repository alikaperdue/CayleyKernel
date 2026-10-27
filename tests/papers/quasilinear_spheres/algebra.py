"""Exact arithmetic, tensor right multiplication and independent rank calculations.

Sparse tensor coordinate r*d+i is row r, algebra basis e_i. Algebra level n
means dimension 2**n. Kernels/ranks are computed from actual products.
The sparse arithmetic and RREF implementation follow the earlier paper suite;
this package is standalone and imports no other paper package.
"""
from fractions import Fraction as F
from functools import lru_cache
import random
import CayleyKernel as CK


def e(i):
    return {i:1}


def add(*vectors):
    out = {}
    for v in vectors:
        for i,x in v.items():
            out[i] = out.get(i,0)+x
    return {i:x for i,x in out.items() if x}


def scale(a,c):
    return {i:x*c for i,x in a.items() if x*c}


def conj(a):
    return {i:x if i==0 else -x for i,x in a.items()}


def dot(a,b):
    return sum(x*b.get(i,0) for i,x in a.items())


def norm2(a):
    return dot(a,a)


@lru_cache(None)
def oracle_sign(i,j,dimension):
    if dimension==1:
        return 1
    half=dimension//2
    a,b=i%half,j%half
    if i<half and j<half:
        return oracle_sign(a,b,half)
    if i<half<=j:
        return oracle_sign(b,a,half)
    if j<half<=i:
        return (-1 if b else 1)*oracle_sign(a,b,half)
    return -(-1 if b else 1)*oracle_sign(b,a,half)


def mul(a,b,n,reference=False):
    out={}
    for i,x in a.items():
        for j,y in b.items():
            if not 0<=i<(1<<n) or not 0<=j<(1<<n):
                raise ValueError('basis index outside algebra')
            if reference:
                k,s=i^j,oracle_sign(i,j,1<<n)
            else:
                label=CK.basis(i+1,j+1)
                k,s=abs(label)-1,1 if label>0 else -1
            out[k]=out.get(k,0)+x*y*s
    return {i:x for i,x in out.items() if x}


def rows_from_columns(columns,dimension):
    rows=[{} for _ in range(dimension)]
    for j,column in enumerate(columns):
        for i,x in column.items():
            if x:
                rows[i][j]=F(x)
    return rows


def rref(rows,width):
    """Sparse rational Gauss-Jordan; optional RHS is just another column."""
    rows=[{j:F(x) for j,x in row.items() if x} for row in rows if any(row.values())]
    pivot_rows=[]
    cursor=0
    for column in range(width):
        found=next((i for i in range(cursor,len(rows)) if rows[i].get(column)),None)
        if found is None:
            continue
        rows[cursor],rows[found]=rows[found],rows[cursor]
        pivot=rows[cursor][column]
        rows[cursor]={j:x/pivot for j,x in rows[cursor].items()}
        current=rows[cursor]
        for i in range(len(rows)):
            if i!=cursor and rows[i].get(column):
                factor=rows[i][column]
                for j,x in current.items():
                    value=rows[i].get(j,0)-factor*x
                    if value:
                        rows[i][j]=value
                    else:
                        rows[i].pop(j,None)
        pivot_rows.append(column)
        cursor+=1
        if cursor==len(rows):
            break
    return rows,pivot_rows


def nullspace(rows,width):
    reduced,pivots=rref(rows,width)
    result=[]
    for free in range(width):
        if free in pivots:
            continue
        v={free:F(1)}
        for row,pivot in zip(reduced,pivots):
            if row.get(free):
                v[pivot]=-row[free]
        result.append(v)
    return result


def span_rank(columns,dimension):
    return len(rref(rows_from_columns(columns,dimension),len(columns))[1])


def embed(v, old_d, new_d):
    return {(p//old_d)*new_d+p%old_d:x for p,x in v.items()}


def right(v,a,d):
    out={}
    for p,x in v.items():
        row,i=divmod(p,d)
        for j,y in a.items():
            label=CK.basis(i+1,j+1)
            q=row*d+abs(label)-1
            out[q]=out.get(q,0)+x*y*(1 if label>0 else -1)
    return {p:x for p,x in out.items() if x}


def orbit(v,d):
    return [right(v,e(j),d) for j in range(d)]


def constraints(frame,d):
    return [column for i,v in enumerate(frame) for column in orbit(embed(v,1<<i,d),d)]


def frame(n,length,seed):
    """Exact nonzero rays in W; normalize each ray by its norm for unit frames.

    Homogeneous orthogonality and all subspaces/ranks are unchanged by this
    positive normalization, so irrational square roots never enter RREF.
    """
    rng=random.Random(seed)
    result=[]
    for i in range(length):
        d=1<<i
        kernel=nullspace(constraints(result,d),n*d)
        if not kernel:
            raise ValueError('no nonzero next vector')
        chosen=rng.sample(kernel,min(3,len(kernel)))
        v=add(*(scale(x,rng.choice((-2,-1,1,2))) for x in chosen))
        result.append(v)
    return result


def transform(v,d,matrix):
    out={}
    for p,x in v.items():
        r,i=divmod(p,d)
        for s,row in enumerate(matrix):
            if row[r]:
                q=s*d+i
                out[q]=out.get(q,0)+row[r]*x
    return {p:x for p,x in out.items() if x}


def diagonal(signs):
    return [[int(i==j)*s for j in range(len(signs))] for i,s in enumerate(signs)]


def sphere_dimensions(n):
    return [(n-k)*(1<<k)-1 for k in range(5)]


def minimum_q(base,rank,c=1):
    """Smallest positive Whitney multiplicity satisfying Lemma 2.2."""
    if rank<=0:
        raise ValueError('rank must be positive')
    return max(1,(base-c+2+c*rank-1)//(c*rank))
