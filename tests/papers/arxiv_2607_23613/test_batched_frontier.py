# SPDX-License-Identifier: Apache-2.0
import json
import numpy as np
import pytest
from scripts.batched_eigen_frontier import (
    BatchProducts, residual, derivative, solve_batch, retry_group, starting_points,
)
from scripts.paper_quadratic_frontier import reference_product, Products, encoded, newton

pytestmark = pytest.mark.cpu


def independent_rows(co, x, lam):
    a, b, c, d = co
    return a+reference_product(b, x)-lam, c+reference_product(d, x)-reference_product(x, lam)


def test_products_residual_and_derivative_against_individual_and_reference():
    rng = np.random.default_rng(456)
    for dim in (2, 8, 16):
        co = tuple(rng.normal(size=(3, dim)) for _ in range(4))
        x, h = rng.normal(size=(2, 3, dim))
        P = BatchProducts(dim, 'cpu')
        f, bx = residual(P, co, x)
        jh = derivative(P, co, x, bx, h)
        for i in range(3):
            a, b, c, d = (v[i] for v in co)
            individual = Products(dim, 'cpu')
            B, D, RA = individual.fixed(b), individual.fixed(d), individual.fixed(a, False)
            expected = reference_product(x[i], B(x[i]))+RA(x[i])-D(x[i])-c
            np.testing.assert_allclose(f[i], expected, atol=1e-11)
            expected_j = reference_product(h[i], B(x[i]))+reference_product(x[i], B(h[i]))+RA(h[i])-D(h[i])
            np.testing.assert_allclose(jh[i], expected_j, atol=1e-11)
        eps = 1e-5
        plus, _ = residual(P, co, x+eps*h)
        minus, _ = residual(P, co, x-eps*h)
        np.testing.assert_allclose(jh, (plus-minus)/(2*eps), atol=1e-8)


def mixed_problem():
    # scalar x²-c with rows already solved, Newton converging, and stalled at 0.
    co = [np.zeros((5, 4)) for _ in range(4)]
    co[1][:, 0] = 1
    co[2][:, 0] = [1, 4, 1, 9, 16]
    x = np.zeros((5, 4)); x[:, 0] = [1, 3, 0, 5, 7]
    return co, x


def test_mixed_convergence_compaction_partial_chunk_and_both_rows():
    co, x = mixed_problem()
    results, stats = solve_batch(co, x, batch_size=3, maxiter=20)
    assert [r['status'] for r in results] == ['eigenpair_verified', 'eigenpair_verified', 'unresolved',
                                            'eigenpair_verified', 'eigenpair_verified']
    assert results[0]['solver']['iterations'] == 0
    assert results[2]['solver']['reason'] == 'line search stalled'
    assert set(stats['batch_sizes']) == {1, 2, 3}
    single, _ = solve_batch(co, x, batch_size=1, maxiter=20)
    for i, (r, s) in enumerate(zip(results, single)):
        np.testing.assert_allclose(r['_root'], s['_root'], atol=1e-10)
        rows = independent_rows([v[i] for v in co], r['_root'], r['_lambda'])
        np.testing.assert_allclose(r['row_residuals'], [np.linalg.norm(v) for v in rows], atol=1e-12)
        if r['status'] == 'eigenpair_verified':
            assert max(np.linalg.norm(v) for v in rows) < 1e-8


def test_dense_small_newton_matches_existing_solver():
    rng = np.random.default_rng(123)
    co = tuple(rng.normal(size=(3, 8))*.1 for _ in range(4))
    co[0][:, 0] += 2
    x = rng.normal(size=(3, 8))*.1
    batch, _ = solve_batch(co, x, batch_size=2)
    for i, r in enumerate(batch):
        a, b, c, d = (v[i] for v in co)
        mul = reference_product
        f = lambda z: mul(z, mul(b, z))+mul(z, a)-mul(d, z)-c
        jac = lambda z,h: mul(h, mul(b,z))+mul(z,mul(b,h))+mul(h,a)-mul(d,h)
        ref, info = newton(f, jac, x[i])
        assert info['converged'] and r['status'] == 'eigenpair_verified'
        np.testing.assert_allclose(r['_root'], ref, atol=1e-8)


def test_retry_rounds_stop_success_and_keep_seeds():
    co, x = mixed_problem()
    records = [{'dimension': 4, 'level': 2, 'seed': 100+i, 'matrix_id': f'matrix-{i}',
                'coefficients': {k: encoded(v[i]) for k,v in zip('abcd', co)}} for i in range(5)]
    results, _ = retry_group(records, x, 'cpu', 4, 12, batch_size=3, maxiter=20)
    assert results[0]['attempts'] == []
    assert len(results[1]['attempts']) == 1
    assert [r['matrix_id'] for r in results] == [r['matrix_id'] for r in records]
    assert [r['restart_seed'] for r in results] == [112+i for i in range(5)]
    for a, b in zip(starting_points(x[1], 8, 113), starting_points(x[1], 8, 113)):
        np.testing.assert_array_equal(a[0], b[0])


def test_budget_nonfinite_and_shape_validation():
    co, x = mixed_problem()
    result, _ = solve_batch(co, x, maxiter=0)
    assert result[1]['status'] == 'unresolved'
    assert result[1]['solver']['reason'] == 'iteration budget'
    x[1] = np.nan
    with np.errstate(invalid='ignore'):
        result, _ = solve_batch(co, x, maxiter=1)
    assert result[1]['status'] == 'unresolved'
    assert result[1]['solver']['reason'] == 'nonfinite residual'
    json.dumps([{k:v for k,v in r.items() if not k.startswith('_')} for r in result], allow_nan=False)
    with pytest.raises(ValueError): solve_batch(co, x, batch_size=0)
    with pytest.raises(ValueError): BatchProducts(4, 'cpu').full(x[0], x[0])


def test_gpu_uses_real_2d_inputs_and_device_outputs(monkeypatch):
    import CayleyKernel as CK
    try:
        BatchProducts(4, 'gpu')
    except Exception as exc:
        pytest.skip(f'CUDA unavailable: {exc}')
    original = CK.multiply
    shapes = []
    def traced(a, b):
        assert isinstance(a, CK.cp.ndarray) and isinstance(b, CK.cp.ndarray)
        assert a.ndim == b.ndim == 2 and a.shape == b.shape
        shapes.append(a.shape)
        result = original(a, b)
        assert isinstance(result, CK.cp.ndarray)
        return result
    monkeypatch.setattr(CK, 'multiply', traced)
    co, x = mixed_problem()
    gpu, _ = solve_batch(co, x, backend='gpu', batch_size=3)
    cpu, _ = solve_batch(co, x, batch_size=3)
    assert {shape[0] for shape in shapes} == {1, 2, 3}
    for g, c in zip(gpu, cpu):
        assert g['status'] == c['status']
        np.testing.assert_allclose(g['_root'], c['_root'], atol=1e-8)
    # Noncommuting, nonassociative coefficients also exercise actual CUDA
    # products against the recursive oracle (scalar-only tests cannot do this).
    rng = np.random.default_rng(711)
    co = tuple(rng.normal(size=(3, 16))*.1 for _ in range(4))
    co[0][:, 0] += 3
    x = rng.normal(size=(3, 16))*.1
    P = BatchProducts(16, 'gpu')
    device_co = tuple(CK.cp.asarray(v) for v in co)
    device_x = CK.cp.asarray(x)
    f, bx = residual(P, device_co, device_x)
    h = rng.normal(size=x.shape)
    jh = derivative(P, device_co, device_x, bx, CK.cp.asarray(h))
    oracle = BatchProducts(16, 'cpu')
    expected, expected_bx = residual(oracle, co, x)
    np.testing.assert_allclose(P.host(f), expected, atol=1e-11)
    np.testing.assert_allclose(P.host(jh), derivative(oracle, co, x, expected_bx, h), atol=1e-11)
    results, _ = solve_batch(co, x, backend='gpu', batch_size=2)
    for i, r in enumerate(results):
        assert r['status'] == 'eigenpair_verified'
        rows = independent_rows([v[i] for v in co], r['_root'], r['_lambda'])
        assert max(np.linalg.norm(v) for v in rows) <= 1e-8



def test_batched_cli_and_retry_archive(tmp_path, monkeypatch):
    from scripts.paper_quadratic_frontier import main
    from scripts.retry_eigen_frontier import main as retry_main
    source = tmp_path/'first'
    monkeypatch.setattr('sys.argv', ['frontier', '--experiment', 'eigen', '--solver', 'batched',
                                    '--levels', '2,3', '--samples', '3', '--maxiter', '1', '--starts', '1',
                                    '--batch-size', '2', '--output', str(source)])
    main()
    before = (source/'results.json').read_bytes()
    report = json.loads(before)
    assert len(report['runs']) == 6
    assert all(r['dimension'] == 1 << r['level'] for r in report['runs'])
    output = tmp_path/'retry'
    monkeypatch.setattr('sys.argv', ['retry', '--input', str(source), '--output', str(output),
                                    '--backend', 'cpu', '--solver', 'batched', '--starts', '3',
                                    '--batch-size', '2'])
    retry_main()
    assert (source/'results.json').read_bytes() == before
    assert (output/'original/results.json').read_bytes() == before
    assert json.loads((output/'results.json').read_text())['backup_verified']
    with pytest.raises(SystemExit): retry_main()


def test_restarted_krylov_against_dense_independent_jacobian():
    from scripts.batched_eigen_frontier import gmres_batch
    rng = np.random.default_rng(99)
    co = tuple(rng.normal(size=(3, 4))*.1 for _ in range(4))
    co[0][:, 0] += 3
    x, rhs = rng.normal(size=(2, 3, 4))
    rhs[0] = 0  # exits Krylov immediately, other rows need multiple restarts
    P = BatchProducts(4, 'cpu')
    _, bx = residual(P, co, x)
    step, counts = gmres_batch(P, co, x, bx, rhs, 1e-12, restart=2, cycles=20, rtol=1e-10)
    assert counts[0] == 0 and max(counts) > 2
    for i in range(3):
        a, b, _, d = (v[i] for v in co)
        mul = reference_product
        columns = [mul(h,mul(b,x[i]))+mul(x[i],mul(b,h))+mul(h,a)-mul(d,h) for h in np.eye(4)]
        expected = np.linalg.solve(np.stack(columns, axis=1), rhs[i])
        np.testing.assert_allclose(step[i], expected, atol=1e-9)


def test_benchmark_archives_identical_work_and_iteration_counts(tmp_path, monkeypatch):
    from scripts.benchmark_eigen_frontier import main
    output = tmp_path/'benchmark'
    monkeypatch.setattr('sys.argv', ['benchmark', '--level', '2', '--matrices', '2', '--starts', '2',
                                    '--batch-size', '3', '--maxiter', '3', '--repeats', '1', '--output', str(output)])
    main()
    report = json.loads((output/'results.json').read_text())
    assert len(report['measurements']) == 4
    assert len(report['searches']) == 4
    for m in report['measurements']:
        assert m['verified_searches']+m['unresolved_searches'] == 4
        assert len(m['newton_iterations']) == len(m['krylov_iterations']) == 4
        assert sum(m['batch_sizes'].values()) == m['multiply_calls']
        assert m['end_to_end_seconds'] > 0
    with pytest.raises(SystemExit): main()


def test_independent_line_search_decisions():
    co = [np.zeros((2, 2)) for _ in range(4)]
    co[1][:, 0] = 1
    co[2][:, 0] = [1, 4]
    x = np.array([[.1, 0], [3., 0]])
    results, _ = solve_batch(co, x, batch_size=2, maxiter=20)
    assert all(r['status'] == 'eigenpair_verified' for r in results)
    a, b = (r['solver'] for r in results)
    assert a['line_search_evaluations'] > a['newton_steps']
    assert b['line_search_evaluations'] == b['newton_steps']
    single, _ = solve_batch(co, x, batch_size=1, maxiter=20)
    for r, s in zip(results, single):
        np.testing.assert_allclose(r['_root'], s['_root'], atol=1e-12)
        assert r['solver'] == s['solver']


def test_benchmark_screen_overrides_labels_and_interrupt(tmp_path, monkeypatch, capsys):
    import scripts.benchmark_eigen_frontier as benchmark
    from argparse import Namespace
    defaults = dict(level=None, matrices=None, starts=None, batch_size=None,
                    maxiter=None, gmres_restart=None, gmres_cycles=None, repeats=None)
    profile = benchmark.apply_profile(Namespace(screen=True, **defaults))
    assert (profile.level, profile.matrices, profile.starts, profile.batch_size) == (6, 32, 2, 64)
    assert (profile.maxiter, profile.gmres_restart, profile.gmres_cycles, profile.repeats) == (15, 20, 3, 1)
    output = tmp_path/'screen'
    monkeypatch.setattr('sys.argv', ['benchmark', '--screen', '--level', '2', '--matrices', '1',
                                    '--starts', '1', '--maxiter', '1', '--progress-seconds', '0',
                                    '--output', str(output)])
    original = benchmark.individual
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(*args, **kwargs)
    monkeypatch.setattr(benchmark, 'individual', interrupted)
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 130
    report = json.loads((output/'results.json').read_text())
    assert report['status'] == 'interrupted'
    assert len(report['measurements']) == 2
    assert report['configuration']['level'] == 2
    assert report['configuration']['batch_size'] == 64
    assert report['active_measurement']['solver'] == 'individual'
    text = capsys.readouterr().out
    assert '[cpu level=2 dim=4 batch=64]' in text
    assert 'individual first_use START' in text
    assert 'individual steady_state 1/1 START' in text
    assert 'INTERRUPTED' in text


def test_benchmark_heartbeat_stops(capsys):
    import time
    from scripts.benchmark_eigen_frontier import heartbeat
    with heartbeat('test-measurement', .01):
        time.sleep(.06)
    output = capsys.readouterr().out
    assert 'test-measurement RUNNING:' in output
    time.sleep(.03)
    assert capsys.readouterr().out == ''
