"""Independent product-based and bit-recipe Emanation Tables.

Indices are mathematical, zero based. Signed ET entries are -P for same-slope
DMZs (paper's marked cell), +P for opposite-slope DMZs, and 0 for blank.
No GPU API is called here. The table recipe never calls the product oracle.
"""
from functools import lru_cache
from itertools import combinations
import CayleyKernel as CK


def e(i):
    return {i:1}


def add(*vectors):
    out={}
    for v in vectors:
        for i,a in v.items():out[i]=out.get(i,0)+a
    return {i:a for i,a in out.items() if a}


def scale(v,c):
    return {i:a*c for i,a in v.items() if a*c}


@lru_cache(None)
def recursive_sign(i,j,d):
    """Four independent branches of (a,b)(c,d)=(ac-conj(d)b,da+b conj(c))."""
    if d==1:return 1
    h=d//2;a,b=i%h,j%h
    if i<h and j<h:return recursive_sign(a,b,h)
    if i<h<=j:return recursive_sign(b,a,h)
    if j<h<=i:return (-1 if b else 1)*recursive_sign(a,b,h)
    return -(-1 if b else 1)*recursive_sign(b,a,h)


def mul(a,b,level,reference=False):
    out={}
    for i,x in a.items():
        for j,y in b.items():
            if not 0<=i<(1<<level) or not 0<=j<(1<<level):raise ValueError('invalid index')
            if reference:k,s=i^j,recursive_sign(i,j,1<<level)
            else:
                label=CK.basis(i+1,j+1)
                k,s=abs(label)-1,1 if label>0 else -1
            out[k]=out.get(k,0)+s*x*y
    return {i:x for i,x in out.items() if x}


def assessor(r,S,N,slope=1):
    G=1<<(N-1)
    if not 0<r<G or r==S or not 0<S<G:raise ValueError('invalid assessor or strut constant')
    return {r:1,G+(r^S):slope}


def labels(S,N):
    """Increasing first representatives with XOR-S partners in reverse order."""
    G=1<<(N-1)
    if not 0<S<G:raise ValueError('S must be strictly between zero and G')
    first=[i for i in range(1,G) if i!=S and i<(i^S)]
    return first+[i^S for i in reversed(first)]


def cell(r,c,S,N):
    """Determine a signed cell from the four exact bilinear product coefficients.

    Fix the row slope +1 and solve the low-low/high-high cancellation for
    the column slope. Then require the independent cross terms to cancel.
    Off-diagonal support has only these two coordinates, so this is exact.
    """
    if r==c or r^c==S:return 0
    G=1<<(N-1);R=G+(r^S);C=G+(c^S)
    t=-CK.twist(r,c)*CK.twist(R,C)
    if t*CK.twist(r,C)+CK.twist(R,c):return 0
    return -(r^c) if t==1 else r^c


def direct_cell(r,c,S,N):
    if r==c or r^c==S:return 0
    a=assessor(r,S,N)
    for slope in (-1,1):
        b=assessor(c,S,N,slope)
        if not mul(a,b,N):return -(r^c) if slope==1 else r^c
    return 0


def high_bits(S,corrected=False):
    """Literal p25 preparation; optional proposed trailing-bit correction.

    The printed right-shift4 rule treats multiples of8. For multiples of16,
    dropping the actual least significant set bit instead is tested separately.
    This correction is ours, not text silently attributed to the paper.
    """
    cutoff=max(3,(S & -S).bit_length()) if corrected else (4 if S%8==0 else 3)
    return [i for i in reversed(range(cutoff,S.bit_length())) if S&(1<<i)]


def paint_cell(r,c,S,N,corrected=False):
    """Unsigned cookbook [0]--[7], explicit first-rule precedence."""
    if r==c or r^c==S:return False
    bits=high_bits(S,corrected)
    for position,gamma in enumerate(bits,1):
        q=1<<gamma;residue=S%q
        if any(x%q==0 or x%q==residue for x in (r,c,r^c)):
            return bool(position%2)
    return not len(bits)%2


def recipe_rules(S,N,corrected=False):
    G=1<<(N-1);rules=[]
    for i,gamma in enumerate(high_bits(S,corrected),1):
        q=1<<gamma;residue=S%q
        values=sorted({x for m in range(G//q+1) for x in (m*q,m*q+residue) if 0<x<G and x!=S})
        rules.append({'bit':q,'action':'fill' if i%2 else 'hide','values':values})
    return rules


def table(S,N):
    ll=labels(S,N)
    return [[cell(r,c,S,N) for c in ll] for r in ll]


def counts(S,N):
    G=1<<(N-1)
    filled=sum(bool(cell(r,c,S,N)) for r in range(1,G) if r!=S for c in range(1,G) if c!=S)
    return {'N':N,'S':S,'G':G,'filled_directed_cells':filled,'edges':filled//2,'edge_count_divided_by_12':filled//24,
            'divisible_by_24':filled%24==0}


def trip_count(n):
    d=1<<n
    return (d-1)*(d-2)//6


def whorf_count(N):
    t=1<<(N-4)
    return t*(t-1)+(2*t-1)*(2*t-2)//6


def box_kites(S,N):
    """Enumerate XOR-sail K_(2,2,2) octahedra via forced XOR-S struts.

    No count/24 identity is assumed. Each three-strut subset is tested for
    all twelve edges. Returns real vertex index sextuples, not graph drawings.
    """
    ll=labels(S,N);pairs=[(r,r^S) for r in ll if r<(r^S)]
    out=[]
    for triple in combinations(pairs,3):
        if (triple[0][0]^triple[1][0]^triple[2][0]) not in (0,S):continue
        if all(cell(a,b,S,N) for p,q in combinations(triple,2) for a in p for b in q):
            out.append(tuple(sorted(sum(triple,()))))
    return out


def trijoin(R,S,T):
    """Section5: (a,x,z) in R, (x,b,y) in S, (z,y,c) in T."""
    return {(a,b,c) for a,x,z in R for xx,b,y in S if xx==x for zz,yy,c in T if zz==z and yy==y}
