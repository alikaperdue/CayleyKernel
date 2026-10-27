"""Section-ordered verification of the quasilinear multiplication schedule."""
from itertools import product
import math,time,unittest
import CayleyKernel as CK
from tests.papers.type_replay import replay_product
from .common import e,mul,gpu_replay
try:
    import numpy as np
except ImportError:np=None
try:
    import pytest
    pytestmark=pytest.mark.cpu
except ImportError:pass
if np is not None:
    from scripts.quasilinear_cd import multiply,complex_cross,real_cross,direct,arithmetic_counts
MEASUREMENTS={}


@unittest.skipIf(np is None,'NumPy unavailable')
class Sections(unittest.TestCase):
    def test_02_scalar_imaginary_decomposition(self):
        rng=np.random.default_rng(260911588)
        for n in range(1,8):
            N=1<<n
            X=rng.integers(-2,3,size=(3,N)).astype(float);Y=rng.integers(-2,3,size=(3,N)).astype(float)
            X[:,0]=Y[:,0]=0
            cross=(direct(X,Y)-direct(Y,X))/2
            np.testing.assert_array_equal(real_cross(X,Y),cross)
            np.testing.assert_array_equal(cross[:,0],0)

    def test_03_1_complex_two_call_recursion_is_bilinear_not_hermitian(self):
        rng=np.random.default_rng(260911589)
        for n in range(1,7):
            N=1<<n
            X=rng.integers(-2,3,size=(4,N))+1j*rng.integers(-2,3,size=(4,N))
            Y=rng.integers(-2,3,size=(4,N))+1j*rng.integers(-2,3,size=(4,N))
            X[:,0]=Y[:,0]=0
            np.testing.assert_array_equal(complex_cross(X,Y),(direct(X,Y)-direct(Y,X))/2)
            np.testing.assert_array_equal(complex_cross(1j*X,Y),1j*complex_cross(X,Y))

    def test_03_2_all_basis_and_dense_products_against_module_and_doubling(self):
        rng=np.random.default_rng(260911590)
        for n in range(7):
            N=1<<n;I=np.eye(N)
            a=np.repeat(I,N,axis=0);b=np.tile(I,(N,1))
            result=multiply(a,b)
            np.testing.assert_array_equal(result,direct(a,b))
            for i,j in product(range(N),repeat=2):
                expected=np.zeros(N);label=CK.basis(i+1,j+1)
                expected[abs(label)-1]=1 if label>0 else -1
                np.testing.assert_array_equal(result[i*N+j],expected)
            a=rng.integers(-2,3,size=(16,N));b=rng.integers(-2,3,size=(16,N))
            np.testing.assert_array_equal(multiply(a,b),direct(a.astype(float),b.astype(float)))

    def test_04_operation_counts_and_table_1(self):
        expected=[(4,2,6),(16,12,28),(58,68,126),(178,284,474),(490,900,1442),
                  (1258,2500,3922),(3082,6436,9970),(7306,15780,24242),
                  (16906,37412,57138),(38410,86564,131634)]
        rows=[]
        for n in range(1,21):
            N=1<<n;counts=arithmetic_counts(n)
            if n<=10:self.assertEqual(tuple(counts[k] for k in ('M','A','T')),expected[n-1])
            if n<=12:
                stats={};multiply(np.ones(N),np.ones(N),stats)
                self.assertEqual(stats['real_bilinear_products_per_pair'],counts['M'])
            direct_total=2*N*N-N
            cc_M=N*(N-1)//2+2;cc_A=3*n*N+(N*(N-3)+4)//2;cc_S=3 if N==2 else 2*N
            if n>=4:self.assertLess(counts['T'],direct_total)
            if n>=5:self.assertLess(counts['T'],cc_M+cc_A+cc_S)
            rows.append(dict(level=n,dimension=N,quasilinear=counts,direct_total=direct_total,cc_total=cc_M+cc_A+cc_S))
        MEASUREMENTS['operation_counts']=rows

    def test_06_dense_extension_through_level_20(self):
        rng=np.random.default_rng(260911591);rows=[]
        for n in (10,12,14,16,18,20):
            N=1<<n;a=rng.choice([-1.,1.],N);b=rng.choice([-1.,1.],N)
            start=time.perf_counter();out=multiply(a,b);elapsed=time.perf_counter()-start
            errors=[]
            for k in (0,1,N//3,N-1):
                expected=sum(CK.twist(i,i^k)*a[i]*b[i^k] for i in range(N))
                self.assertEqual(out[k],expected);errors.append(float(abs(out[k]-expected)))
            # Every coordinate checked for a dense-by-four-term product.
            sparse=np.zeros(N);sparse[[0,1,N//2+1,N-1]]=[1,2,-1,3]
            got=multiply(a,sparse);expected=np.zeros(N)
            for i in range(N):
                for j in (0,1,N//2+1,N-1):expected[i^j]+=CK.twist(i,j)*a[i]*sparse[j]
            np.testing.assert_array_equal(got,expected)
            rows.append(dict(level=n,dimension=N,python_cpu_seconds=elapsed,checked_dense_coordinates=[0,1,N//3,N-1],max_sampled_error=max(errors),dense_sparse_all_coordinates=True))
        MEASUREMENTS['dense_extension']=rows

    def test_05_broadcasting_and_float_error_protocol(self):
        rng=np.random.default_rng(260911592)
        for n in (3,5,8):
            N=1<<n;a=rng.normal(size=(2,1,N));b=rng.normal(size=(1,3,N))
            a/=np.linalg.norm(a,axis=-1,keepdims=True);b/=np.linalg.norm(b,axis=-1,keepdims=True)
            a,b=np.broadcast_arrays(a,b);out=multiply(a,b);ref=direct(a,b)
            tolerance=128*N*np.finfo(float).eps
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertLessEqual(float(np.max(abs(out-ref))),tolerance)

    def test_07_gpu_direct_comparison(self):
        # Once prerequisites are satisfied, compare QL directly to multiply().
        if CK.cp is None:self.skipTest('CuPy unavailable; direct GPU comparison not executed')
        if not CK.loaded():self.skipTest('compiled kernel unavailable')
        try:count=CK.cp.cuda.runtime.getDeviceCount()
        except CK.cp.cuda.runtime.CUDARuntimeError as exc:self.skipTest(str(exc))
        if not count:self.skipTest('no CUDA device')
        rng=np.random.default_rng(260911593)
        for n in (3,4,5,8,10):
            N=1<<n;a=rng.integers(-2,3,size=(8,N)).astype(float);b=rng.integers(-2,3,size=(8,N)).astype(float)
            np.testing.assert_array_equal(CK.cp.asnumpy(replay_product(a,b)),multiply(a,b))

    def test_08_gpu_complex_quasilinear_extension(self):
        from tests.papers.type_replay import require_gpu, host
        require_gpu(self)
        rng=np.random.default_rng(260911594)
        for n in (3,4,5):
            N=1<<n
            a=rng.integers(-2,3,(4,N))+1j*rng.integers(-2,3,(4,N))
            b=rng.integers(-2,3,(4,N))+1j*rng.integers(-2,3,(4,N))
            # Complexify the real QL algorithm by bilinearity; its real
            # operation-count table is not a complex-input cost estimate.
            expected=(multiply(a.real,b.real)-multiply(a.imag,b.imag)
                      +1j*(multiply(a.real,b.imag)+multiply(a.imag,b.real)))
            np.testing.assert_array_equal(expected,direct(a,b))
            for dtype in (np.complex64,np.complex128):
                actual=host(CK.multiply(a.astype(dtype),b.astype(dtype)))
                self.assertEqual(actual.dtype,np.dtype(dtype))
                np.testing.assert_array_equal(actual,expected)
