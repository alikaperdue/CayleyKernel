# SPDX-License-Identifier: Apache-2.0
"""Exact finite classification and signed-permutation centralizer calculations."""
from collections import Counter
from itertools import combinations, product, islice
import random
import numpy as np
import sympy as sp
import CayleyKernel as CK
from .algebra import basis_product, oracle


def xor_span(basis):
    values=[0]
    for b in basis:
        values += [x^b for x in values]
    return values


def rref_subspaces(n):
    """Each rank-three subspace of F2^n once, by unique reduced row echelon form.

    Columns are bits low to high. Pivots are the least set bits of rows.
    """
    for pivots in combinations(range(n),3):
        slots=[(r,c) for r,p in enumerate(pivots) for c in range(p+1,n) if c not in pivots]
        for bits in product((0,1),repeat=len(slots)):
            rows=[1<<p for p in pivots]
            for bit,(r,c) in zip(bits,slots):rows[r]|=bit<<c
            yield tuple(rows)


def canonical_basis(rows):
    """Unique fully reduced binary basis, with highest-bit pivots."""
    pivots={}
    for row in rows:
        for p in sorted(pivots,reverse=True):
            if row>>p&1:row^=pivots[p]
        if row:
            p=row.bit_length()-1
            for q,b in list(pivots.items()):
                if b>>p&1:pivots[q]^=row
            pivots[p]=row
    return tuple(pivots[p] for p in sorted(pivots,reverse=True))


def gaussian_count(n):
    return ((1<<n)-1)*((1<<n)-2)*((1<<n)-4)//168


PAIRS=np.array(list(combinations(range(1,8),2)))
II=PAIRS[:,0,None]; JJ=PAIRS[:,1,None]; KK=np.arange(8)[None,:]


def tables(bases):
    elements=[xor_span(b) for b in bases]
    pairs=((x,y) for elems in elements for x in elems for y in elems)
    return np.array(CK.twist_many(pairs),dtype=np.int8).reshape(-1,8,8)


def defects(signs):
    """Exact Hurwitz condition: all imaginary left matrices anticommute.

    Signed permutation columns make L_i^T=-L_i and L_i^T L_i=I;
    these remaining polarized equations are necessary and sufficient for N(xy)=Nx Ny.
    """
    s=np.asarray(signs,dtype=np.int8).reshape(-1,8,8)
    anti=s[:,JJ,KK]*s[:,II,JJ^KK]+s[:,II,KK]*s[:,JJ,II^KK]
    return np.count_nonzero(np.any(anti!=0,axis=-1),axis=-1)


def table_product(signs,a,b):
    out=[0]*8
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i^j]+=int(signs[i,j])*x*y
    return out


def failure_witness(signs):
    """Prefer an exact zero-divisor witness; otherwise return a norm defect."""
    first=None
    for i,j in combinations(range(1,8),2):
        for k in range(8):
            ell=i^j^k
            for sign in (1,-1):
                a=[0]*8;b=[0]*8;a[i]=a[j]=1;b[k]=1;b[ell]=sign
                out=table_product(signs,a,b)
                norm=sum(v*v for v in out)
                if norm!=4:
                    witness={'x':a,'y':b,'product':out,'product_norm_sq':norm,'expected_norm_sq':4}
                    if norm==0:return witness
                    if first is None:first=witness
    return first


def survey(n,bases,chunk_size=512,retain=False):
    iterator=iter(bases); hist=Counter();representatives={};passing=[];total=0;patterns=Counter()
    while True:
        chunk=list(islice(iterator,chunk_size))
        if not chunk:break
        ss=tables(chunk);ds=defects(ss)
        for b,s,d in zip(chunk,ss,ds):
            d=int(d);hist[d]+=1;total+=1;patterns[s.tobytes().hex()]+=1
            if d not in representatives:
                representatives[d]={'basis':list(b),'elements':xor_span(b),'signs':s.tolist(),
                                    'norm_counterexample':failure_witness(s) if d else None}
            if retain and not d:passing.append(list(b))
    return {'level':n,'total':total,'composition':hist[0],'noncomposition':total-hist[0],
            'defective_generator_pair_histogram':dict(sorted(hist.items())),
            'representatives':representatives,'passing_rref_bases':passing if retain else None,'sign_table_histogram':dict(sorted(patterns.items()))}


def sampled_bases(n,count,seed):
    if count>gaussian_count(n):raise ValueError('sample exceeds the number of distinct rank-three subspaces')
    rng=random.Random(seed+n);seen=set()
    while len(seen)<count:
        b=canonical_basis([rng.getrandbits(n) for _ in range(3)])
        if len(b)==3 and b not in seen:
            seen.add(b);yield b


def basis_profile(d,u):
    """Every column, no dense matrices. Compare L_u,R_u at a chosen basis unit."""
    js=list(range(d))
    requests=[]
    for j in js:
        requests += [(u,j),(u,u^j),(j,u),(j^u,u),(u,j^u),(u^j,u)]
    s=np.array(CK.twist_many(requests),dtype=np.int8).reshape(d,6)
    lsq=s[:,0]*s[:,1];rsq=s[:,2]*s[:,3]
    lr=s[:,2]*s[:,4];rl=s[:,0]*s[:,5]
    p=int(np.count_nonzero(lr==-1));q=d-p
    return {'unit':u,'left_square_minus_identity':bool(np.all(lsq==-1)),
            'right_square_minus_identity':bool(np.all(rsq==-1)),
            'commute':bool(np.array_equal(lr,rl)),'projector_ranks':[p,q],
            'joint_centralizer_real_dimension':(p*p+q*q)//2,
            'skew_joint_centralizer_real_dimension':(p*p+q*q)//4}


class SignedUnion:
    """Exact equations x_a = sign*x_b; a negative cycle forces a component to zero."""
    def __init__(self,n):
        self.parent=list(range(n));self.weight=[1]*n;self.dead=[False]*n
    def find(self,a):
        if self.parent[a]!=a:
            p,w=self.find(self.parent[a]);self.weight[a]*=w;self.parent[a]=p
        return self.parent[a],self.weight[a]
    def join(self,a,b,sign):
        ra,wa=self.find(a);rb,wb=self.find(b)
        if ra==rb:
            if wa!=sign*wb:self.dead[ra]=True
        else:
            self.parent[ra]=rb;self.weight[ra]=sign*wb*wa
            self.dead[rb]=self.dead[rb] or self.dead[ra]
    def dimension(self):
        return sum(self.parent[a]==a and not self.dead[a] for a in range(len(self.parent)))


def graph_centralizer(d,u,skew=False):
    """Independent full matrix-entry commutator solve for signed permutation L/R."""
    signs=CK.twist_many([(u,j) for j in range(d)]+[(j,u) for j in range(d)])
    left,right=signs[:d],signs[d:]
    uf=SignedUnion(d*d)
    for r in range(d):
        for c in range(d):
            a=r*d+c;b=(r^u)*d+(c^u)
            uf.join(a,b,left[r]*left[c]);uf.join(a,b,right[r]*right[c])
            if skew:uf.join(a,c*d+r,-1)
    return uf.dimension()


def sparse_unit_checks(n,u,columns):
    """High-bit independent oracle checks, without claiming a complete column scan."""
    for j in columns:
        for a,b in ((u,j),(u,u^j),(j,u),(j^u,u)):
            sign=oracle(a,b,n)
            assert CK.twist(a,b)==sign
            assert basis_product(a+1,b+1)==sign*((a^b)+1)
        l=CK.twist(u,j);r=CK.twist(j,u)
        assert l*CK.twist(u,u^j)==-1
        assert r*CK.twist(j^u,u)==-1
        assert r*CK.twist(u,j^u)==(-1 if j in (0,u) else 1)


def signed_isomorphism(source,target):
    """A signed XOR-linear basis isometry, or None; compare all 64 products."""
    for a in range(1,8):
        for b in range(1,8):
            if b==a:continue
            for c in range(1,8):
                p=xor_span((a,b,c))
                if len(set(p))!=8:continue
                for s1,s2,s4 in product((-1,1),repeat=3):
                    signs=[1,s1,s2,0,s4,0,0,0]
                    for i,j,k in ((1,2,3),(1,4,5),(2,4,6),(3,4,7)):
                        signs[k]=int(source[i,j])*signs[i]*signs[j]*int(target[p[i],p[j]])
                    if all(int(source[i,j])*signs[i^j]==signs[i]*signs[j]*int(target[p[i],p[j]])
                           for i in range(8) for j in range(8)):
                        return {'permutation':p,'signs':signs}
    return None


def gpu_bases(n,batch):
    """Guarantee both classes; repeat low-level spans when batch exceeds their count."""
    count=min(gaussian_count(n),max(64,batch*2))
    candidates=list(sampled_bases(n,count,2026140014))
    signs=tables(candidates);dd=defects(signs)
    good=[b for b,d in zip(candidates,dd) if not d]
    bad=[b for b,d in zip(candidates,dd) if d]
    # Both classes have known embedded witnesses at every requested n>=4.
    if not good:good=[(1,2,4)]
    if not bad:bad=[(1,2,12)]
    out=[good[0],bad[0]]
    out.extend(candidates[k%len(candidates)] for k in range(max(0,batch-2)))
    return out[:batch]
