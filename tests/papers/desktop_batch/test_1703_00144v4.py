"""LDR networks: finite reconstruction, embedding and gradient checks."""
import unittest
from .common import e, mul, gpu_replay
try:
    import numpy as np
except ImportError:
    np=None
try:
    import pytest
    pytestmark=pytest.mark.cpu
except ImportError:
    pass


def left_matrix(unit,dim):
    out=np.zeros((dim,dim))
    for j in range(dim):
        for i,c in mul(e(unit),e(j)).items(): out[i,j]=c
    return out


def reconstruct(A,B,G,H):
    # Cayley basis action: q=2 and a=-1; matches Lemma 3.1.
    T=np.linalg.inv(np.eye(len(A))+B@B)
    return (G@H.T + A@G@H.T@B)@T


@unittest.skipIf(np is None,'NumPy unavailable for matrix experiments')
class Sections(unittest.TestCase):
    def test_03_2_lemma_3_1_reconstruction_with_kernel_operators(self):
        rng=np.random.default_rng(170300144)
        for dim in (2,4,8,16,32,64):
            A=left_matrix(dim-1,dim); I=np.eye(dim)
            np.testing.assert_array_equal(A@A,-I)
            B=np.diag(np.arange(1,dim+1)/(2*dim))
            M=rng.integers(-3,4,size=(dim,dim)).astype(float)
            D=M-A@M@B
            got=(D+A@D@B)@np.linalg.inv(I+B@B)
            np.testing.assert_allclose(got,M,atol=2e-14)

    def test_03_1_table_1_cauchy_stein_counterexample(self):
        s=np.array([2.,3.,5.,7.]); t=np.array([11.,13.,17.,19.])
        M=1/(s[:,None]-t[None,:]); A=np.diag(s); B=np.diag(t)
        # Printed table labels the rank-one bound as Stein. Standard
        # Cauchy has rank-one SYLVESTER displacement with these operators.
        self.assertEqual(np.linalg.matrix_rank(A@M-M@B,tol=1e-10),1)
        self.assertGreater(np.linalg.matrix_rank(M-A@M@B,tol=1e-10),1)

    def test_04_theorem_4_1_and_4_2_single_ridge_embedding(self):
        rng=np.random.default_rng(170300145)
        for dim in (2,4,8,16):
            A=left_matrix(dim-1,dim); B=np.diag(np.arange(1,dim+1)/(2*dim))
            H=np.ones((dim,1)); j=0
            columns=[]
            for k in range(dim):
                G=np.eye(dim)[:,k:k+1]
                columns.append(reconstruct(A,B,G,H)[:,j])
            K=np.column_stack(columns)
            v=rng.normal(size=dim)
            G=np.linalg.solve(K,v)[:,None]
            M=reconstruct(A,B,G,H)
            np.testing.assert_allclose(M[:,j],v,atol=1e-13)
            self.assertEqual(np.linalg.matrix_rank(M-A@M@B,tol=1e-10),1)
            x=rng.uniform(size=(20,dim)); bias=.3
            y=np.tanh(x@M+bias)[:,j]
            np.testing.assert_allclose(y,np.tanh(x@v+bias),atol=1e-13)

    def test_05_lemma_5_1_preserves_finite_network_output(self):
        rng=np.random.default_rng(170300146); dim=8
        A=left_matrix(7,dim); B=np.diag(np.arange(1,dim+1)/16)
        H=np.ones((dim,1))
        K=np.column_stack([reconstruct(A,B,np.eye(dim)[:,i:i+1],H)[:,0] for i in range(dim)])
        V=rng.normal(size=(dim,5)); alpha=rng.normal(size=5); bias=rng.normal(size=5)
        blocks=[reconstruct(A,B,np.linalg.solve(K,V[:,i])[:,None],H) for i in range(5)]
        x=rng.uniform(size=(32,dim))
        expected=np.tanh(x@V+bias)@alpha
        actual=sum(alpha[i]*np.tanh(x@blocks[i][:,0]+bias[i]) for i in range(5))
        np.testing.assert_allclose(actual,expected,atol=1e-13)

    def test_06_gradient_finite_differences_and_printed_transpose(self):
        rng=np.random.default_rng(170300147); dim=4
        A=left_matrix(3,dim); B=np.diag([.1,.2,.3,.4])
        G=rng.normal(size=(dim,2)); H=rng.normal(size=(dim,2)); D=rng.normal(size=(dim,dim))
        T=np.linalg.inv(np.eye(dim)+B@B)
        gradG=D@T.T@H + A.T@D@T.T@B.T@H
        gradH=T@D.T@G + B@T@D.T@A@G
        loss=lambda g,h:np.sum(D*reconstruct(A,B,g,h))
        for target,grad in [(G,gradG),(H,gradH)]:
            for index in np.ndindex(target.shape):
                old=target[index]; eps=1e-6
                target[index]=old+eps; plus=loss(G,H)
                target[index]=old-eps; minus=loss(G,H)
                target[index]=old
                self.assertAlmostEqual((plus-minus)/(2*eps),grad[index],delta=2e-8)
        printedG=D.T@T.T@H + A.T@D.T@T.T@B.T@H
        self.assertGreater(np.max(np.abs(printedG-gradG)),.01)

    def test_07_extension_basis_operators_through_dimension_256(self):
        for dim in (32,64,128,256):
            A=left_matrix(dim-1,dim)
            np.testing.assert_array_equal(A.T@A,np.eye(dim))
            np.testing.assert_array_equal(A@A,-np.eye(dim))

    def test_08_gpu_operator_replay(self):
        gpu_replay(self,[(e(15),e(i)) for i in range(16)],16)
