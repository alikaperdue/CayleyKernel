# SPDX-License-Identifier: Apache-2.0
"""Explicit internal irreps from reference [7], arXiv:2505.07923v4, VII-VIII.

Octonion matrices come from CayleyKernel products through the original adapter.
The C2 quaternion representation is faithful for H; tensor products here are
operator tensor products, never a claim that C tensor H tensor O is a CD algebra.
"""
from functools import lru_cache
from itertools import combinations
import sympy as s
from .algebra import operators


def clean(x):
    return x.applyfunc(s.simplify)


def adj(x):
    return x.conjugate().T


def norm_basis(columns):
    return s.GramSchmidt(columns,True)


def restriction_matrix(g,columns):
    q=s.Matrix.hstack(*columns)
    return clean(adj(q)*g*q)


@lru_cache(None)
def model():
    l,r=operators(8);I=s.eye(8)
    a=[(-l[5]+s.I*l[4])/2,(-l[3]+s.I*l[1])/2,(-l[6]+s.I*l[2])/2]
    ad=list(map(adj,a))
    v=a[0]*a[1]*a[2]*ad[2]*ad[1]*ad[0]
    p=(I+s.I*l[7])/2;P=(I+s.I*r[7])/2;A=(I+l[2]*l[6]*l[7])/2
    ps=[p*P,p*(I-P),(I-p)*P*A,(I-p)*P*(I-A),(I-p)*(I-P)]
    ps=list(map(clean,ps))
    # Eq.26 as printed, retaining all normalizations and multiplication order.
    x=lambda i,j,k,m:ad[i-1]*ad[j-1]*v*a[k-1]*a[m-1]
    color=[-s.I*x(2,3,3,1)-s.I*x(1,3,3,2),
           -x(2,3,3,1)+x(1,3,3,2),
           -s.I*x(2,3,3,2)-s.I*x(1,3,3,1),
           -s.I*x(1,2,2,3)-s.I*x(3,2,2,1),
           x(1,2,2,3)-x(3,2,2,1),
           -s.I*x(3,1,1,2)-s.I*x(2,1,1,3),
           -x(3,1,1,2)+x(2,1,1,3),
           s.I/s.sqrt(3)*(x(1,3,3,1)+x(2,3,3,2)-2*x(1,2,2,1))]
    printed_color=list(map(clean,color))
    color=printed_color.copy()
    # Eq.26's printed third generator has trace -2i; retain it above and
    # test the minimal sign repair explicitly, rather than silently changing the source.
    color[2]=clean(-s.I*x(2,3,3,2)+s.I*x(1,3,3,1))
    weak=[s.I*(ad[0]*v*a[1]+ad[1]*v*a[0]),ad[0]*v*a[1]-ad[1]*v*a[0],
          s.I*(ad[0]*v*a[0]-ad[1]*v*a[1])]
    weak=list(map(clean,weak))
    y=(ps[0]+ps[1]/3+ps[3]-ps[4])/2
    # Canonical orthonormal states, grouped by the actual Peirce projectors.
    states=[norm_basis(p.columnspace()) for p in ps]
    eps1=s.Matrix([[0,1],[-1,0]]);eps2=s.Matrix([[0,s.I],[s.I,0]]);eps3=eps1*eps2
    beta=s.kronecker_product((-eps2+s.I*eps1)/2,s.I*l[7])
    vh=beta*adj(beta);vac=vh*s.kronecker_product(s.eye(2),v)
    triples=ad[0]*ad[1]*ad[2]
    annih=a[0]*a[1]*a[2]
    pairs=[(0,1),(1,2),(2,0)]
    double=[ad[i]*ad[j] for i,j in pairs]
    internal={
        '12':[v*a[i]*a[j] for i,j in pairs],
        '13':[v*a[i] for i in (0,1)],
        '14':[v*a[2]],'15':[v*annih],
        '23':[c*v*a[j] for c in double for j in (0,1)],
        '24':[c*v*a[2] for c in double],
        '25':[c*v*annih for c in double],
        '34':[ad[i]*v*a[2] for i in (0,1)],
        '35':[ad[i]*v*annih for i in (0,1)],
        '45':[ad[2]*v*annih],
    }
    # Full rank-one operator families with reference beta left/right excitations.
    full={}
    recipes={
        '12':[(s.eye(8),a[i]*a[j]) for i,j in pairs],
        '13':[(s.eye(8),a[i]) for i in (0,1)],
        '14':[(s.eye(8),a[2])],'15':[(s.eye(8),annih)],
        '23':[(c,a[j]) for c in double for j in (0,1)],
        '24':[(c,a[2]) for c in double], '25':[(c,annih) for c in double],
        '34':[(ad[i],a[2]) for i in (0,1)],
        '35':[(ad[i],annih) for i in (0,1)],'45':[(ad[2],annih)],
    }
    # v_O instead of v in H12 supplies the same four-dimensional spatial span.
    for label,terms in recipes.items():
        full[label]=[]
        for left,right in terms:
            base=s.kronecker_product(s.eye(2),left)*vac*s.kronecker_product(s.eye(2),right)
            for b,c in ((0,0),(1,0),(0,1),(1,1)):
                full[label].append(clean((adj(beta) if b else s.eye(16))*base*(beta if c else s.eye(16))))
    return {'left':l,'right':r,'a':a,'ad':ad,'v':v,'projectors':ps,'states':states,
            'printed_color':printed_color,'color':color,'weak':weak,'Y':y,'beta':beta,'vacuum16':vac,
            'epsilon':[eps1,eps2,eps3], 'internal_ladder_basis':internal,'full_ladder_basis':full}


def restricted_module(i,j):
    """Matrices acting on column-vectorized rectangular maps from block j to i."""
    m=model();bi=m['states'][i-1];bj=m['states'][j-1]
    ni,nj=len(bi),len(bj)
    def action(g):
        gi=restriction_matrix(g,bi);gj=restriction_matrix(g,bj)
        return clean(s.kronecker_product(s.eye(nj),gi)-s.kronecker_product(gj.T,s.eye(ni)))
    return [action(g/2) for g in m['color']],[action(g/2) for g in m['weak']],action(m['Y'])


def commutant_dimension(generators):
    d=generators[0].rows
    system=s.Matrix.vstack(*(s.kronecker_product(s.eye(d),g)-s.kronecker_product(g.T,s.eye(d)) for g in generators))
    return d*d-system.rank()


def sparse_matrix(matrix):
    return {'shape':[matrix.rows,matrix.cols],
            'entries':[[i,j,str(matrix[i,j])] for i in range(matrix.rows) for j in range(matrix.cols) if matrix[i,j]!=0]}


# Independent expected assignments transcribed from reference VIII.3, with orientation i<j.
EXPECTED={
    '12':('anti-bottom', '3bar',1,s.Rational(1,3)),
    '13':('anti-lepton doublet','1',2,s.Rational(1,2)),
    '14':('sterile anti-neutrino','1',1,s.Rational(0)),
    '15':('charged anti-lepton','1',1,s.Rational(1)),
    '23':('left quark doublet','3',2,s.Rational(1,6)),
    '24':('right down-type quark','3',1,s.Rational(-1,3)),
    '25':('right up-type quark','3',1,s.Rational(2,3)),
    '34':('Higgs times (anti)-neutrino','1',2,s.Rational(-1,2)),
    '35':('anti-tau doublet','1',2,s.Rational(1,2)),
    '45':('anti-tau singlet','1',1,s.Rational(1)),
}


EXPECTED_CHARGES={
    '12':{s.Rational(1,3):3},'13':{s.Integer(0):1,s.Integer(1):1},
    '14':{s.Integer(0):1},'15':{s.Integer(1):1},
    '23':{s.Rational(2,3):3,s.Rational(-1,3):3},
    '24':{s.Rational(-1,3):3},'25':{s.Rational(2,3):3},
    '34':{s.Integer(0):1,s.Integer(-1):1},'35':{s.Integer(0):1,s.Integer(1):1},
    '45':{s.Integer(1):1},
}
