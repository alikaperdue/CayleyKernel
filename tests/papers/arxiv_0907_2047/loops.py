"""Exact signed basis loops, exhaustive subspaces and isomorphism certificates.

Loop elements are signed ONE-based CayleyKernel labels. A support is a bit mask
of mathematical ZERO-based indices. No printed '-0' is parsed as integer zero.
"""
from collections import Counter
from functools import lru_cache
from itertools import product

import numpy as np
import CayleyKernel as CK

REPS = {
    'S': tuple(range(16)),
    'S_alpha': (0,1,2,3,8,9,10,11,20,21,22,23,28,29,30,31),
    'S_beta': (0,1,2,3,12,13,14,15,20,21,22,23,24,25,26,27),
    'S_gamma': (0,1,2,3,4,5,6,7,24,25,26,27,28,29,30,31),
    'O': tuple(range(8)),
    'O_quasi': (0,1,2,3,12,13,14,15),
    'Q8': (0,1,2,3), 'C4': (0,1), 'C2': (0,),
}


def support_mask(indices):
    return sum(1 << i for i in indices)


def indices(mask):
    return tuple(i for i in range(32) if mask >> i & 1)


def labels(mask):
    return tuple(s*(i+1) for s in (1,-1) for i in indices(mask))


@lru_cache(None)
def independent_sign(i, j, dimension=32):
    """Four doubling branches, independent of both module sign routines."""
    if dimension == 1:
        return 1
    half = dimension//2
    a,b = i % half,j % half
    if i < half and j < half:
        return independent_sign(a,b,half)
    if i < half <= j:
        return independent_sign(b,a,half)
    if j < half <= i:
        return (-1 if b else 1)*independent_sign(a,b,half)
    return -(-1 if b else 1)*independent_sign(b,a,half)


SIGNS = np.array([[CK.twist(i,j) for j in range(32)] for i in range(32)],dtype=np.int8)
ELEMENTS = tuple(range(1,33))+tuple(range(-1,-33,-1))
POSITION = {x:i for i,x in enumerate(ELEMENTS)}
TABLE = np.array([[POSITION[CK.basis(x,y)] for y in ELEMENTS] for x in ELEMENTS],dtype=np.int16)
INVERSES = np.array([next(y for y in range(64) if TABLE[x,y]==TABLE[y,x]==0) for x in range(64)])


def signed_product(a,b):
    return int(ELEMENTS[TABLE[POSITION[a],POSITION[b]]])


def sparse_mul(a,b,reference=False):
    out = {}
    for i,x in a.items():
        for j,y in b.items():
            if reference:
                k,s = i ^ j, independent_sign(i,j)
            else:
                label = CK.basis(i+1,j+1)
                k,s = abs(label)-1,1 if label>0 else -1
            out[k] = out.get(k,0)+s*x*y
    return {i:x for i,x in out.items() if x}


def sparse_add(a,b):
    out = dict(a)
    for i,x in b.items():
        out[i] = out.get(i,0)+x
    return {i:x for i,x in out.items() if x}


def sparse_scale(a,c):
    return {i:x*c for i,x in a.items() if x*c}


@lru_cache(None)
def all_spaces():
    """Adjoin every possible vector; deduplicate spans by 32-bit support."""
    found = {1}
    queue = [1]
    while queue:
        mask = queue.pop()
        u = indices(mask)
        for v in range(1,32):
            if not mask >> v & 1:
                new = mask | support_mask(tuple(x ^ v for x in u))
                if new not in found:
                    found.add(new)
                    queue.append(new)
    return tuple(sorted(found,key=lambda m:(m.bit_count(),indices(m))))


@lru_cache(None)
def coordinates(mask):
    """Deterministic basis and map from binary coordinates to ambient index."""
    values = [0]
    basis = []
    for v in indices(mask):
        if v not in values:
            basis.append(v)
            values += [v ^ x for x in values]
    return tuple(basis),tuple(values)


@lru_cache(None)
def invariant(mask):
    """Per-element associator-failure profile and total Moufang failures.

    Central input signs cancel from each identity; positive inputs suffice for
    these signed-loop isomorphism invariants. Direct signed tests are separate.
    """
    u = indices(mask)
    failures = []
    moufang_failures = 0
    for a in u:
        count = 0
        for b,c in product(u,repeat=2):
            count += int(SIGNS[a,b]*SIGNS[a ^ b,c] != SIGNS[b,c]*SIGNS[a,b ^ c])
            lhs = SIGNS[a,b]*SIGNS[c,a]*SIGNS[a ^ b,c ^ a]
            rhs = SIGNS[b,c]*SIGNS[b ^ c,a]*SIGNS[a,(b ^ c) ^ a]
            moufang_failures += int(lhs != rhs)
        failures.append(count)
    return tuple(sorted(Counter(failures).items())),int(moufang_failures)


@lru_cache(None)
def gl_maps(rank):
    """Every invertible F2-linear map, as its vector-image permutation."""
    size = 1 << rank
    maps = []
    def extend(values,depth):
        if depth == rank:
            maps.append(values)
            return
        for v in range(1,size):
            if v not in values:
                extend(values+[x ^ v for x in values],depth+1)
    extend([0],0)
    return np.array(maps,dtype=np.int16)


def word_phases(ambient):
    """Signs of left-associated ordered generator words, vectorized."""
    ambient = np.asarray(ambient)
    phases = np.ones(ambient.shape,dtype=np.int8)
    for v in range(1,ambient.shape[-1]):
        bit = 1 << (v.bit_length()-1)
        previous = v ^ bit
        phases[...,v] = phases[...,previous]*SIGNS[ambient[...,previous],ambient[...,bit]]
    return phases


def find_isomorphism(source,target):
    """Return an explicit signed map; search induced GL maps, not invariants.

    Any isomorphism fixes the unique nonidentity involution -1 and descends to
    F2 vector quotients. Generator signs can be made positive by a character;
    ordered generator words then force all other signs. Full products are
    checked afterward by verify_isomorphism().
    """
    source_basis,source_values = coordinates(source)
    target_basis,target_values = coordinates(target)
    if len(source_basis) != len(target_basis):
        return None
    ambient = np.array(target_values,dtype=np.int16)[gl_maps(len(source_basis))]
    phases = word_phases(np.array(source_values))*word_phases(ambient)
    active = np.arange(len(ambient))
    for v,w in product(range(len(source_values)),repeat=2):
        lhs = SIGNS[source_values[v],source_values[w]]*phases[active,v ^ w]
        rhs = phases[active,v]*phases[active,w]*SIGNS[ambient[active,v],ambient[active,w]]
        active = active[lhs == rhs]
        if not len(active):
            return None
    chosen = int(active[0])
    mapping = {}
    for v,x in enumerate(source_values):
        y = int(phases[chosen,v])*(int(ambient[chosen,v])+1)
        mapping[x+1] = y
        mapping[-(x+1)] = -y
    return mapping


def verify_isomorphism(source,target,mapping):
    """Certificate validation directly against public basis()."""
    if mapping is None or set(mapping) != set(labels(source)) or set(mapping.values()) != set(labels(target)):
        return False
    return all(mapping[CK.basis(x,y)] == CK.basis(mapping[x],mapping[y])
               for x,y in product(labels(source),repeat=2))


@lru_cache(None)
def classification():
    """Discover classes by invariants; certify every membership by an isomorphism."""
    rep_signatures = {invariant(support_mask(u)):name for name,u in REPS.items()}
    result = {}
    for mask in all_spaces():
        if mask.bit_count() == 32:
            continue
        signature = invariant(mask)
        name = rep_signatures.get(signature)
        if name is None:
            raise AssertionError(('unexpected class',mask,signature))
        rep = support_mask(REPS[name])
        mapping = find_isomorphism(rep,mask)
        if not verify_isomorphism(rep,mask,mapping):
            raise AssertionError(('invalid isomorphism',name,mask))
        result[mask] = (name,mapping,signature)
    return result


def norm_composition_defect(mask):
    """Exact quartic polynomial N(ab)-N(a)N(b), symbolic real coordinates.

    A monomial key (i,p,j,q) means a_i*a_p*b_j*b_q, with each pair sorted.
    Empty output certifies the identity for ALL real vectors in this span.
    """
    u = indices(mask)
    out = Counter()
    for k in u:
        terms = [(i,i ^ k,int(SIGNS[i,i ^ k])) for i in u]
        for i,j,s in terms:
            for p,q,t in terms:
                key = tuple(sorted((i,p)))+tuple(sorted((j,q)))
                out[key] += s*t
    for i,j in product(u,repeat=2):
        out[i,i,j,j] -= 1
    return {key:c for key,c in out.items() if c}


def zero_divisor(mask):
    """Search two-term partners; do not infer nonexistence from failed search."""
    u = indices(mask)
    for i,j in product(u[1:],repeat=2):
        if i >= j:
            continue
        for k in u[1:]:
            l = i ^ j ^ k
            if l <= k or l not in u:
                continue
            for sign in (-1,1):
                a,b = {i:1,j:1},{k:1,l:sign}
                if not sparse_mul(a,b):
                    return a,b
    return None


@lru_cache(None)
def lattice_data():
    """All 375 subloops, including trivial singleton and ambient loop.

    At node zero support_mask=0 denotes {+1} specially. All other masks denote
    both signs of every basis index in that mask.
    """
    spaces = (0,)+all_spaces()
    lookup = {m:i for i,m in enumerate(spaces)}
    meet = np.empty((len(spaces),len(spaces)),dtype=np.int16)
    join = np.empty_like(meet)
    for i,a in enumerate(spaces):
        for j,b in enumerate(spaces):
            meet[i,j] = lookup[a & b]
            if not a:
                joined = b
            elif not b:
                joined = a
            else:
                joined = a
                for v in indices(b):
                    if not joined >> v & 1:
                        joined |= support_mask(tuple(x ^ v for x in indices(joined)))
            join[i,j] = lookup[joined]
    covers = []
    for i,a in enumerate(spaces):
        for j,b in enumerate(spaces):
            if a & b == a and ((not a and b==1) or (a and b.bit_count()==2*a.bit_count())):
                covers.append((i,j))
    return spaces,meet,join,covers
