"""Exact sparse CayleyKernel arithmetic and rational linear algebra.

B(a,b) = sqrt(2)*{a,b}_paper is used throughout. This removes radicals from
all product/annihilator computations; factors are restored explicitly in tests.
Nullspaces are computed from products, never from the theorem being tested.
"""
from fractions import Fraction as F
from functools import lru_cache

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


def imaginary_unit(n):
    return e(1<<(n-1))


def pc(a,n):
    return {i:x for i,x in a.items() if i in (0,1<<(n-1))}


def perp(a,n):
    return add(a,scale(pc(a,n),-1))


def h_indices(n):
    q=1<<(n-2)
    return (0,q,2*q,3*q)


def hermitian(a,b,n):
    return pc(mul(a,conj(b),n),n)


def pair(a,b,n):
    return add(a,{i+(1<<n):x for i,x in b.items()})


def parts(a,n):
    half=1<<(n-1)
    return ({i:x for i,x in a.items() if i<half},
            {i-half:x for i,x in a.items() if i>=half})


def B(a,b,n):
    if pc(a,n) or pc(b,n):
        raise ValueError('B requires a,b in C_n perpendicular')
    return pair(add(a,b),mul(imaginary_unit(n),add(scale(a,-1),b),n),n)


def split_B(z,n):
    """Inverse of unnormalized B, z in A_n. Returns coordinates in A_(n-1)."""
    x,y=parts(z,n)
    iy=mul(imaginary_unit(n-1),y,n-1)
    return scale(add(x,iy),F(1,2)),scale(add(x,scale(iy,-1)),F(1,2))


def tilde(alpha,n):
    return {0:alpha[0]} if set(alpha)=={0} else {
        i:x for i,x in ((0,alpha.get(0,0)),(1<<n,alpha.get(1<<(n-1),0))) if x}


def J(n):
    """(0,i_n) in A_(n+1)."""
    return e((1<<n)+(1<<(n-1)))


def cspan_projection(a,b,n):
    if not a:
        return {}
    ia=mul(imaginary_unit(n),a,n)
    return add(scale(a,F(dot(a,b),norm2(a))),scale(ia,F(dot(ia,b),norm2(a))))


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


def same_span(a,b,dimension):
    ra,rb=span_rank(a,dimension),span_rank(b,dimension)
    return ra==rb==span_rank(a+b,dimension)


@lru_cache(None)
def _ann(items,n):
    a=dict(items)
    columns=[mul(a,e(j),n) for j in range(1<<n)]
    kernel=nullspace(rows_from_columns(columns,1<<n),1<<n)
    return tuple(tuple(sorted(v.items())) for v in kernel)


def ann(a,n):
    return [dict(v) for v in _ann(tuple(sorted(a.items())),n)]


def ann_dimension(a,n):
    return len(ann(a,n))


def coordinate_intersection(basis,forbidden):
    rows=[{j:v[i] for j,v in enumerate(basis) if v.get(i)} for i in forbidden]
    coefficients=nullspace(rows,len(basis))
    return [add(*(scale(basis[j],x) for j,x in c.items())) for c in coefficients]


def in_D(a,b,n):
    return not hermitian(a,b,n) and all(not dot(a,x) for x in ann(b,n)) and all(not dot(b,x) for x in ann(a,n))


def divide(a,b,n):
    """Definition 6.10: solve ax=b with x perpendicular to Ann(a)."""
    dim=1<<n
    rows=rows_from_columns([mul(a,e(j),n) for j in range(dim)],dim)
    for i,x in b.items():
        rows[i][dim]=F(x)
    rows += [dict(v) for v in ann(a,n)]
    reduced,pivots=rref(rows,dim)
    if any(not any(j<dim for j in row) and row.get(dim) for row in reduced):
        raise ValueError('b is not in the multiplication image')
    if len(pivots)!=dim:
        raise ValueError('complement did not give a unique solution')
    return {pivot:row[dim] for row,pivot in zip(reduced,pivots) if row.get(dim)}


def constraint_columns(a,b,n):
    """Prop 5.2 parameterization: four real equations, and B-coordinate map."""
    aa,bb=ann(a,n),ann(b,n)
    parameters=[]
    for alpha in ({0:1},imaginary_unit(n)):
        parameters.append((alpha,{}, {},{}))
    for beta in ({0:1},imaginary_unit(n)):
        parameters.append(({},beta,{},{}))
    parameters += [({}, {},x,{}) for x in aa]
    parameters += [({}, {},{},y) for y in bb]
    constraints=[];maps=[]
    for alpha,beta,x,y in parameters:
        eq3=add(scale(alpha,norm2(a)),scale(conj(beta),norm2(b)))
        eq4=add(mul(add(conj(beta),scale(alpha,-1)),pc(mul(a,b,n),n),n),
                pc(add(mul(a,y,n),scale(mul(x,b,n),-1)),n))
        constraints.append({i:value for i,value in enumerate((eq3.get(0,0),eq3.get(1<<(n-1),0),
                                                             eq4.get(0,0),eq4.get(1<<(n-1),0))) if value})
        maps.append(B(add(mul(alpha,a,n),x),add(mul(beta,b,n),y),n))
    return constraints,maps


def predicted_ann_h(a,b,n):
    constraints,maps=constraint_columns(a,b,n)
    coefficients=nullspace(rows_from_columns(constraints,4),len(constraints))
    return [add(*(scale(maps[j],x) for j,x in c.items())) for c in coefficients]


def d_decomposition(a,b,n):
    """Theorem 6.12 in rational B coordinates, including the relative factor 2."""
    if not a or not b or not in_D(a,b,n):
        raise ValueError('requires nonzero D-locus components')
    block1=[B(x,{},n) for x in ann(a,n)]+[B({},y,n) for y in ann(b,n)]
    g2=B(scale(a,norm2(b)),scale(b,-norm2(a)),n)
    g3=add(B(divide(a,b,n),scale(divide(b,a,n),-1),n),scale(J(n),2))
    i=imaginary_unit(n+1)
    return block1,[g2,mul(i,g2,n+1)],[g3,mul(i,g3,n+1)]


def max_ann(n):
    return (1<<n)-4*n+4


def maximal_pair(n):
    a,b=e(1),e(2)
    for k in range(3,n):
        a,b=B(a,{},k),B(b,{},k)
    return a,b


def modular_rank(columns,dimension,prime=1000003):
    """Independent modular row echelon rank; a lower bound on rational rank."""
    pivots={}
    for source in rows_from_columns(columns,dimension):
        row={j:(x.numerator*pow(x.denominator,-1,prime))%prime for j,x in source.items()}
        row={j:x for j,x in row.items() if x}
        while row:
            pivot=min(row)
            if pivot not in pivots:
                inverse=pow(row[pivot],-1,prime)
                pivots[pivot]={j:x*inverse%prime for j,x in row.items()}
                break
            factor=row[pivot]
            for j,x in pivots[pivot].items():
                value=(row.get(j,0)-factor*x)%prime
                if value:
                    row[j]=value
                else:
                    row.pop(j,None)
    return len(pivots)
