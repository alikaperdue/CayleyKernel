"""Search workflow regressions; no kernel product validation."""
import hashlib
import json
import numpy as np
import pytest
from scripts.audit_eigen_benchmarks import audit

pytestmark = pytest.mark.cpu


def benchmark_archive(path):
    path.mkdir()
    # Synthetic archived diagnostics: two starts per matrix, not solver evidence.
    a = np.zeros((4, 4)); a[2:, 0] = 2
    np.savez(path/'inputs.npz', a=a, b=a, c=a, d=a, initial=a)
    searches = [{'matrix_id': f'm-{i//2}', 'seed': i//2, 'start': i%2} for i in range(4)]
    measurements = []
    for name in ('individual', 'batched'):
        rows = [[0, 1e-10], [0, .1], [0, .2], [0, .3]]
        if name == 'batched': rows[2][1] = .05
        filename = name+'.npz'
        np.savez(path/filename, root=a+1, **{'lambda': a})
        measurements.append(dict(solver=name, phase='steady_state', repeat=0, row_residuals=rows,
            verified_searches=1, unresolved_searches=3, verified_matrices=1, unresolved_matrices=1,
            end_to_end_seconds=1, newton_iterations=[1,15,15,15], krylov_iterations=[2,900,900,900],
            vector_file=filename, vector_sha256=hashlib.sha256((path/filename).read_bytes()).hexdigest()))
    report = dict(configuration={'tol':1e-9, 'level':2, 'maxiter':15}, searches=searches,
        measurements=measurements, inputs_sha256=hashlib.sha256((path/'inputs.npz').read_bytes()).hexdigest())
    (path/'results.json').write_text(json.dumps(report))
    return report


def test_audit_deduplicates_siblings_and_selects_best(tmp_path):
    source = tmp_path/'source'
    benchmark_archive(source)
    before = (source/'results.json').read_bytes()
    result = audit([source], tmp_path/'audit')
    assert result['unique_matrices'] == 2 and result['verified_matrices'] == 1
    assert len(result['unresolved_matrices']) == 1
    unresolved = result['unresolved_matrices'][0]
    assert unresolved['matrix_id'] == 'm-1'
    assert unresolved['eigenpair_residual'] == .05
    assert unresolved['selected_candidate']['solver'] == 'batched'
    assert result['benchmarks'][0]['outcome_disagreements'] == []
    assert (source/'results.json').read_bytes() == before
    assert (tmp_path/'audit/original/benchmark-0/results.json').read_bytes() == before
    with pytest.raises(ValueError): audit([source], tmp_path/'audit')


def test_audit_hash_and_count_guards(tmp_path):
    source = tmp_path/'source'
    report = benchmark_archive(source)
    report['measurements'][0]['verified_searches'] = 2
    (source/'results.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='counts'): audit([source], tmp_path/'counts')
    assert not (tmp_path/'counts').exists()
    report['measurements'][0]['verified_searches'] = 1
    (source/'results.json').write_text(json.dumps(report))
    (source/'individual.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='hash'): audit([source], tmp_path/'hash')


def test_audit_reports_different_successful_starts(tmp_path):
    source = tmp_path/'source'
    report = benchmark_archive(source)
    report['measurements'][1]['row_residuals'][:2] = [[0,.1],[0,1e-10]]
    (source/'results.json').write_text(json.dumps(report))
    result = audit([source], tmp_path/'audit')
    assert len(result['benchmarks'][0]['outcome_disagreements'][0]['searches']) == 2
    assert result['verified_matrices'] == 1


def test_default_eigen_pipeline_is_batched(tmp_path, monkeypatch):
    from scripts.paper_quadratic_frontier import main
    output = tmp_path/'eigen'
    monkeypatch.setattr('sys.argv', ['frontier','--experiment','eigen','--levels','2',
                                    '--samples','2','--output',str(output)])
    main()
    report = json.loads((output/'results.json').read_text())
    assert report['configuration']['solver'] == 'batched'
    assert report['configuration']['batch_size'] == 64
    assert len(report['runs']) == 2
    assert all(r['solver_mode'] == 'batched_direct' for r in report['runs'])


def test_resume_saved_without_fresh_starts(tmp_path, monkeypatch):
    from scripts.retry_eigen_frontier import main
    source = tmp_path/'source'; source.mkdir()
    # Continue a known nearly converged search, keeping its branch.
    np.savez(source/'candidate.npz', root=np.array([0., .8, 0., 0.]))
    record = dict(level=2, dimension=4, seed=1, matrix_id='continued', status='unresolved',
        eigenpair_residual=.36, coefficients={'a':{}, 'b':{'0':1}, 'c':{'0':-1}, 'd':{}},
        vector_file='candidate.npz', vector_sha256=hashlib.sha256((source/'candidate.npz').read_bytes()).hexdigest())
    (source/'results.json').write_text(json.dumps({'runs':[record]}))
    output = tmp_path/'retry'
    monkeypatch.setattr('sys.argv', ['retry','--input',str(source),'--output',str(output),
        '--backend','cpu','--resume-saved','--starts','0','--progress-seconds','0'])
    main()
    report = json.loads((output/'results.json').read_text())
    r = report['runs'][0]
    assert report['configuration']['solver'] == 'batched'
    assert r['status'] == 'eigenpair_verified'
    assert r['matrix_id'] == 'continued' and r['previous_eigenpair_residual'] == .36
    assert len(r['attempts']) == 1 and r['attempts'][0]['kind'] == 'continued_saved_candidate'
    with np.load(output/r['vector_file']) as data:
        assert data['root'][1] > 0


def test_retry_checkpoints_before_next_chunk_and_graceful_stop(tmp_path, monkeypatch):
    import signal
    import scripts.retry_eigen_frontier as retry
    source = tmp_path/'source'; source.mkdir()
    np.savez(source/'candidate.npz', root=np.array([0., .8, 0., 0.]))
    digest = hashlib.sha256((source/'candidate.npz').read_bytes()).hexdigest()
    rows = [dict(level=2, dimension=4, seed=i, matrix_id=f'retry-{i}', status='unresolved',
        coefficients={'a':{},'b':{'0':1},'c':{'0':-1},'d':{}},
        vector_file='candidate.npz',vector_sha256=digest) for i in range(3)]
    (source/'results.json').write_text(json.dumps({'runs':rows}))
    output = tmp_path/'retry'
    monkeypatch.setattr('sys.argv',['retry','--input',str(source),'--output',str(output),'--backend','cpu',
        '--resume-saved','--starts','0','--batch-size','1','--minutes','1','--progress-seconds','0'])
    original = retry.retry_group
    calls = [0]
    def stop_on_second(*a, **kw):
        calls[0] += 1
        if calls[0] == 2:
            assert (output/'chunk-0.json').exists()
            assert len(json.loads((output/'results.json').read_text())['runs']) == 1
            signal.raise_signal(signal.SIGINT)
        return original(*a, **kw)
    monkeypatch.setattr(retry,'retry_group',stop_on_second)
    retry.main()
    report = json.loads((output/'results.json').read_text())
    assert report['status']=='interrupted'
    assert len(report['runs'])==2
    assert report['unprocessed_matrix_ids']==['retry-2']
    assert (output/'chunk-1.json').exists()
