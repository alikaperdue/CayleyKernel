"""Tensor multiplication, HyperDense/HyperConv and CD functional maps."""
from itertools import product
import unittest
from .common import e,mul,add,scale,gpu_replay
from .test_2205_07986v2 import independent_sign
try:
    import numpy as np
except ImportError:np=None
try:
    import pytest
    pytestmark=pytest.mark.cpu
except ImportError:pass
MEASUREMENTS={}


def tensor(dim):
    A=np.zeros((dim,dim,dim),dtype=np.int64)
    for i,j in product(range(dim),repeat=2):A[i,j,i^j]=independent_sign(i,j)
    return A


def vec(array):return {i:float(c) for i,c in enumerate(array) if c}


def dense(X,K,A):
    dim,n,u=K.shape
    W=np.tensordot(A,K,axes=(1,0)).transpose(0,2,1,3).reshape(dim*n,dim*u)
    return (X.reshape(len(X),-1)@W).reshape(len(X),dim,u)


def convolution(X,K,A):
    """Algorithm 2, valid cross-correlation with unit strides, no activation."""
    k=X.ndim-3;dim=K.shape[0];F=K.shape[-1]
    W=np.tensordot(A,K,axes=(1,0)).transpose((1,)+tuple(range(2,k+2))+(0,k+2,k+3))
    spatial=tuple(X.shape[1+j]-K.shape[1+j]+1 for j in range(k))
    Y=np.zeros((len(X),)+spatial+(dim,F))
    for b,pos,o in product(range(len(X)),product(*(range(n) for n in spatial)),range(dim)):
        patch=X[(b,)+tuple(slice(p,p+K.shape[1+j]) for j,p in enumerate(pos))]
        Y[(b,)+pos+(o,)]=patch.reshape(-1)@W[o].reshape(-1,F)
    return Y


@unittest.skipIf(np is None,'NumPy unavailable')
class Sections(unittest.TestCase):
    def test_02_structure_tensor_complex_example_and_cd_products(self):
        rng=np.random.default_rng(240700449)
        for dim in (1,2,4,8,16,32,64):
            A=tensor(dim)
            for _ in range(8):
                x,y=rng.integers(-2,3,size=(2,dim))
                actual=np.einsum('ijk,i,j->k',A,x,y)
                expected=mul(vec(x),vec(y))
                np.testing.assert_array_equal(actual,[expected.get(i,0) for i in range(dim)])
        A=tensor(2)
        np.testing.assert_array_equal(A[:,:,0],[[1,0],[0,-1]])
        np.testing.assert_array_equal(A[:,:,1],[[0,1],[1,0]])

    def test_03_1_dense_contraction_permute_reshape_bias_activation(self):
        rng=np.random.default_rng(240700450)
        for dim in (2,4,8,16,32):
            X=rng.integers(-2,3,size=(2,dim,3));K=rng.integers(-2,3,size=(dim,3,4))
            actual=dense(X,K,tensor(dim));expected=np.zeros_like(actual)
            for b,u in product(range(2),range(4)):
                p=add(*(mul(vec(X[b,:,i]),vec(K[:,i,u])) for i in range(3)))
                expected[b,:,u]=[p.get(j,0) for j in range(dim)]
            np.testing.assert_array_equal(actual,expected)
            bias=rng.normal(size=(dim,4))
            np.testing.assert_allclose(np.tanh(actual+bias),np.tanh(expected+bias),atol=1e-14)

    def test_03_2_convolution_1d_2d_3d(self):
        rng=np.random.default_rng(240700451)
        for k,dim in product((1,2,3),(4,16)):
            X=rng.integers(-1,2,size=(2,)+(3,)*k+(dim,2))
            K=rng.integers(-1,2,size=(dim,)+(2,)*k+(2,2))
            actual=convolution(X,K,tensor(dim));expected=np.zeros_like(actual)
            for b,pos,f in product(range(2),product(range(2),repeat=k),range(2)):
                terms=[]
                for offset,c in product(product(range(2),repeat=k),range(2)):
                    location=tuple(p+q for p,q in zip(pos,offset))
                    terms.append(mul(vec(X[(b,)+location+(slice(None),c)]),vec(K[(slice(None),)+offset+(c,f)])))
                value=add(*terms)
                expected[(b,)+pos+(slice(None),f)]=[value.get(i,0) for i in range(dim)]
            np.testing.assert_array_equal(actual,expected)

    def test_04_1_cd_component_functional_reconstruction(self):
        rng=np.random.default_rng(240700452)
        for dim in (2,4,8,16,32,64):
            A=tensor(dim);r=rng.normal(size=dim);x=rng.normal(size=dim)
            for k in range(dim):
                B=A[:,:,k]
                np.testing.assert_array_equal(B@B.T,np.eye(dim))
                v=B@r
                self.assertAlmostEqual(float(v@B@x),float(r@x),delta=1e-12)

    def test_04_2_nondegeneracy_definition_does_not_imply_lemma_for_all_algebras(self):
        # Dual numbers R[epsilon]/epsilon^2 are unital, hence have no
        # nonzero element annihilating ALL partners; scalar projection is singular.
        A=np.zeros((2,2,2));A[0,0,0]=1;A[0,1,1]=A[1,0,1]=1
        np.testing.assert_array_equal(A[0],np.eye(2))
        self.assertEqual(np.linalg.matrix_rank(A[:,:,0]),1)
        v=np.array([0.,1.]);np.testing.assert_array_equal(v@A[:,:,0],[0,0])
        MEASUREMENTS['dual_number_component_ranks']=[int(np.linalg.matrix_rank(A[:,:,k])) for k in range(2)]

    def test_05_appendix_B_tensor_free_high_index_product(self):
        for n in (20,128,1024,4096):
            x={1:2,(1<<(n-1))+3:1};y={2:-1,(1<<(n-2))+5:3}
            expected={}
            for i,a in x.items():
                for j,b in y.items():expected[i^j]=expected.get(i^j,0)+independent_sign(i,j)*a*b
            expected={k:c for k,c in expected.items() if c}
            self.assertEqual(mul(x,y),expected)
        MEASUREMENTS['explicit_tensor_storage_scaling']=[dict(dimension=d,entries=d**3,nonzero=d**2) for d in (8,16,64,1024)]

    def test_06_gpu_tensor_replay(self):
        gpu_replay(self,[({0:1,1:2,15:-1},{0:3,2:1,7:2})],16)

    def test_07_gpu_typed_dense_layer(self):
        import CayleyKernel as CK
        from tests.papers.type_replay import require_gpu, host
        require_gpu(self)
        rng=np.random.default_rng(240700453)
        for dim in (2,4,8,16,32):
            for dtype in (np.float16,np.complex64,np.complex128):
                with self.subTest(dim=dim,dtype=np.dtype(dtype).name):
                    X=rng.integers(-1,2,(2,3,dim)).astype(dtype)
                    K=rng.integers(-1,2,(3,4,dim)).astype(dtype)
                    if np.dtype(dtype).kind=='c':
                        X+=1j*rng.integers(-1,2,X.shape)
                        K+=1j*rng.integers(-1,2,K.shape)
                    # Match Algorithm 1's contraction with broadcast CD
                    # products, followed by the input-channel sum.
                    products=host(CK.multiply(X[:,:,None,:],K[None,:,:,:]))
                    self.assertEqual(products.dtype,np.dtype(dtype))
                    actual=products.sum(axis=1)
                    expected=dense(X.transpose(0,2,1),K.transpose(2,0,1),tensor(dim)).transpose(0,2,1)
                    np.testing.assert_array_equal(actual,expected)
