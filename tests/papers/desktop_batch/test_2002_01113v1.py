"""Cayley matrix transform: Stiefel identities and finite-iteration limits."""
import unittest
from .common import e, mul, scale, add, close, gpu_replay
from .test_1703_00144v4 import left_matrix, np
try:
    import pytest
    pytestmark=pytest.mark.cpu
except ImportError:
    pass
MEASUREMENTS={}


def project(X,Z):
    hat=Z@X.T-.5*X@(X.T@Z@X.T)
    W=hat-hat.T
    return W,W@X


def cayley(W,X,alpha):
    I=np.eye(len(W))
    return np.linalg.solve(I-alpha/2*W,(I+alpha/2*W)@X)


def iterate(W,X,alpha,s):
    Y=X+alpha*W@X
    for _ in range(s):Y=X+alpha/2*W@(X+Y)
    return Y


@unittest.skipIf(np is None,'NumPy unavailable')
class Sections(unittest.TestCase):
    def test_03_1_projection_tangent_and_transport_linearity(self):
        rng=np.random.default_rng(200201113)
        for n,p in [(4,2),(8,8),(16,5)]:
            X=np.linalg.qr(rng.normal(size=(n,p)))[0]
            Z=rng.normal(size=(n,p));V=rng.normal(size=(n,p))
            W,P=project(X,Z)
            np.testing.assert_allclose(W.T,-W,atol=1e-14)
            np.testing.assert_allclose(X.T@P+P.T@X,0,atol=1e-13)
            direct=Z-X@((X.T@Z+Z.T@X)/2)
            np.testing.assert_allclose(P,direct,atol=1e-13)
            np.testing.assert_allclose(project(X,2*Z-3*V)[1],2*P-3*project(X,V)[1],atol=1e-13)

    def test_03_2_closed_cayley_and_module_complex_plane(self):
        for n in (4,16,64,128):
            J=left_matrix(n-1,n);X=np.eye(n)[:,:min(3,n)]
            for a in (.01,.2,.8):
                Y=cayley(J,X,a); t=a/2
                np.testing.assert_allclose(Y.T@Y,np.eye(X.shape[1]),atol=2e-14)
                np.testing.assert_allclose(Y,((1-t*t)*X+2*t*J@X)/(1+t*t),atol=1e-14)
                rotation={0:(1-t*t)/(1+t*t),n-1:2*t/(1+t*t)}
                for j in range(X.shape[1]):
                    close(self,mul(rotation,e(j)),{i:float(c) for i,c in enumerate(Y[:,j]) if c})

    def test_04_algorithm_1_two_iterations_are_not_exactly_orthonormal(self):
        J=left_matrix(3,4);X=np.eye(4)[:,:2];alpha=.8
        Y=iterate(J,X,alpha,2)
        defect=float(np.linalg.norm(Y.T@Y-np.eye(2)))
        self.assertGreater(defect,.01)
        MEASUREMENTS['two_iteration_orthogonality_defect']=defect

    def test_05_1_theorem_1_contraction(self):
        J=left_matrix(7,8); X=np.eye(8)[:,:3];alpha=.4
        target=cayley(J,X,alpha)
        errors=[float(np.linalg.norm(iterate(J,X,alpha,s)-target)) for s in range(8)]
        for x,y in zip(errors,errors[1:]):self.assertLessEqual(y,alpha/2*x+1e-14)
        MEASUREMENTS['fixed_point_errors']=errors

    def test_05_2_appendix_B_big_O_not_little_o_counterexample(self):
        # Analytic error for W=J, X=e0, after s steps:
        # ||error||/alpha^(s+2) = 1/(2^(s+1)*sqrt(1+alpha^2/4)).
        rows=[]
        for s in (0,1,2):
            ratios=[1/(2**(s+1)*(1+a*a/4)**.5) for a in (.1,.01,.001)]
            self.assertGreater(ratios[-1],.1)
            J=left_matrix(1,2);X=np.eye(2)[:,:1];a=.1
            got=np.linalg.norm(iterate(J,X,a,s)-cayley(J,X,a))/a**(s+2)
            self.assertAlmostEqual(got,ratios[0],delta=1e-10)
            rows.append(dict(iterations=s,normalized_errors=ratios))
        MEASUREMENTS['little_o_counterexample']=rows

    def test_06_gpu_rotation_replay(self):
        gpu_replay(self,[({0:3,15:4},e(i)) for i in range(16)],16)
