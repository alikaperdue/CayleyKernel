"""Incremental survey and stopping tests; no kernel validation."""
import hashlib
import json
import signal
from argparse import Namespace

import numpy as np
import pytest
from scripts import survey_eigen_frontier as survey
from scripts.batched_eigen_frontier import solve_batch

pytestmark = pytest.mark.cpu


def config(output):
    return Namespace(output=output, backend='cpu', levels=[2,3], matrices_per_level=3,
        starts=2, support=3, seed=20261030, batch_size=4, maxiter=2, gmres_restart=3,
        gmres_cycles=1, line_search=3, tol=1e-9, minutes=1., progress_seconds=0)


def test_generation_is_chunk_independent():
    co, x, cases = survey.make_chunk(3, 0, 3, 42, 2, 3)
    single_co, single_x, single_cases = survey.make_chunk(3, 2, 1, 42, 2, 3)
    for a,b in zip(co,single_co): np.testing.assert_array_equal(a[-2:],b)
    np.testing.assert_array_equal(x[-2:],single_x)
    assert cases[-1] == single_cases[0]


def test_survey_partial_chunks_checkpoint_before_next_solve(tmp_path, monkeypatch):
    args = config(tmp_path/'survey')
    original = survey.solve_batch
    observed = []
    def checked(*a, **kw):
        manifest = json.loads((args.output/'status.json').read_text())
        observed.append(manifest['completed_chunks'])
        for chunk in range(manifest['completed_chunks']):
            assert (args.output/f'chunk-{chunk}.json').exists()
        return original(*a, **kw)
    monkeypatch.setattr(survey, 'solve_batch', checked)
    report = survey.run(args)
    assert report['status'] == 'complete'
    assert observed == [0,1,2,3]
    assert [c['level'] for c in report['chunks']] == [2,3,2,3]
    assert len(report['runs']) == 6
    assert len({r['matrix_id'] for r in report['runs']}) == 6
    assert report['summary'][2]['matrices'] == report['summary'][3]['matrices'] == 3
    for r in report['runs']:
        assert hashlib.sha256((args.output/r['vector_file']).read_bytes()).hexdigest() == r['vector_sha256']
        assert len(r['attempts']) == 2
    with pytest.raises(ValueError): survey.run(args)


def test_ctrl_c_saves_current_batch(tmp_path, monkeypatch):
    args = config(tmp_path/'interrupt')
    original = survey.solve_batch
    old_handler = signal.getsignal(signal.SIGINT)
    def interrupted(*a, **kw):
        signal.raise_signal(signal.SIGINT)
        return original(*a, **kw)
    monkeypatch.setattr(survey, 'solve_batch', interrupted)
    report = survey.run(args)
    assert report['status'] == 'interrupted'
    assert len(report['chunks']) == 1 and len(report['runs']) == 2
    assert report['summary'][2]['stopped_searches'] == 4
    assert signal.getsignal(signal.SIGINT) == old_handler
    assert json.loads((args.output/'results.json').read_text())['status'] == 'interrupted'


def test_deadline_saves_current_batch(tmp_path, monkeypatch):
    args = config(tmp_path/'deadline')
    clock = [0.]
    monkeypatch.setattr(survey.time, 'monotonic', lambda: clock[0])
    original = survey.solve_batch
    def timed(*a, **kw):
        clock[0] = 61.
        return original(*a, **kw)
    monkeypatch.setattr(survey, 'solve_batch', timed)
    report = survey.run(args)
    assert report['status'] == 'time_limit'
    assert len(report['chunks']) == 1
    assert len(report['runs']) == 2


def test_stop_inside_krylov_keeps_counts_and_last_candidate():
    rng = np.random.default_rng(11)
    co = tuple(rng.normal(size=(2,8)) for _ in range(4))
    x = rng.normal(size=(2,8))
    checks = [0]
    def stop():
        checks[0] += 1
        return checks[0] >= 3
    results, _ = solve_batch(co,x,should_stop=stop)
    assert all(r['solver']['reason']=='run stopped' for r in results)
    assert all(r['solver']['krylov_iterations']==1 for r in results)
    for i,r in enumerate(results): np.testing.assert_array_equal(r['_root'],x[i])


def test_error_preserves_previous_chunks(tmp_path, monkeypatch):
    args = config(tmp_path/'error')
    original = survey.solve_batch
    calls = [0]
    def fail(*a, **kw):
        calls[0] += 1
        if calls[0] == 2: raise RuntimeError('simulated failure')
        return original(*a, **kw)
    monkeypatch.setattr(survey, 'solve_batch', fail)
    with pytest.raises(RuntimeError, match='simulated'): survey.run(args)
    report = json.loads((args.output/'results.json').read_text())
    assert report['status'] == 'error' and len(report['chunks']) == 1
    assert len(report['runs']) == 2
    assert (args.output/'chunk-0.json').exists()


def test_aggregate_only_written_at_start_and_finish_and_recovery(tmp_path, monkeypatch):
    args = config(tmp_path/'bounded-writes')
    original = survey.atomic_json
    writes = []
    def record(path, value):
        writes.append(path.name)
        original(path, value)
    monkeypatch.setattr(survey, 'atomic_json', record)
    result = survey.run(args)
    assert writes.count('results.json') == 2
    assert len([p for p in writes if p.startswith('chunk-')]) == 4
    recovered = survey.recover(args.output, args.output/'recovered.json')
    assert recovered['runs'] == result['runs']
    assert recovered['summary'] == result['summary']
    assert json.loads((args.output/'status.json').read_text())['saved_matrices'] == 6
