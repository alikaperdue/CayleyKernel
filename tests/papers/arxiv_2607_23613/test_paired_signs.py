"""Paired experiment design and reporting; no kernel validation."""
import json
import signal
from argparse import Namespace

import numpy as np
import pytest
from scripts import survey_eigen_frontier as survey

pytestmark = pytest.mark.cpu


def test_sign_pairs_share_coefficients_and_are_chunk_independent():
    co, initial, cases = survey.make_chunk(3, 0, 3, 73, 2, 3, paired_signs=True)
    single_co, single_initial, single_cases = survey.make_chunk(3, 2, 1, 73, 2, 3, paired_signs=True)
    np.testing.assert_array_equal(initial[::2], -initial[1::2])
    assert np.all(np.linalg.norm(initial[::2], axis=1) > 0)
    for values, single in zip(co, single_co):
        np.testing.assert_array_equal(values[::2], values[1::2])
        np.testing.assert_array_equal(values[-2:], single)
    np.testing.assert_array_equal(initial[-2:], single_initial)
    assert cases[-1] == single_cases[0]
    with pytest.raises(ValueError, match='exactly two'):
        survey.make_chunk(3, 0, 1, 73, 3, 3, paired_signs=True)


def test_all_four_outcomes_and_stopped_pairs_remain_distinct():
    rows = []
    for positive, negative, stopped in [(True,True,False), (True,False,False),
                                       (False,True,False), (False,False,False),
                                       (True,False,True)]:
        attempts = [dict(kind=kind, status='eigenpair_verified' if verified else 'unresolved',
                         solver={'reason':'run stopped' if stopped and i==1 else 'iteration budget'})
                    for i, (kind, verified) in enumerate(zip(
                        ('positive_random_normal','negative_random_normal'), (positive,negative)))]
        rows.append(dict(level=3, status='eigenpair_verified' if positive or negative else 'unresolved',
                         attempts=attempts))
    summary = survey.summarize(rows)
    assert summary[3]['paired_sign_outcomes'] == dict(both_verified=1, positive_only=1,
        negative_only=1, neither_verified=1, incomplete=1)
    survey.merge_summary(summary, survey.summarize(rows))
    assert set(summary[3]['paired_sign_outcomes'].values()) == {2}


def test_both_signs_are_saved_even_when_positive_verifies(tmp_path, monkeypatch):
    args = Namespace(output=tmp_path/'pairs', backend='cpu', levels=[2], matrices_per_level=3,
        starts=2, paired_signs=True, support=3, seed=73, batch_size=4, maxiter=2,
        gmres_restart=3, gmres_cycles=1, line_search=3, tol=1e-9, minutes=1., progress_seconds=0)
    calls = []
    def fake_solve(coeff, initial, **kwargs):
        calls.append(len(initial))
        np.testing.assert_array_equal(initial[::2], -initial[1::2])
        return [dict(status='eigenpair_verified' if i%2==0 else 'unresolved',
                     eigenpair_residual=0. if i%2==0 else 1.,
                     row_residuals=[0.,0. if i%2==0 else 1.],
                     solver={'reason':'residual tolerance' if i%2==0 else 'iteration budget'},
                     _root=x.copy(), _lambda=np.zeros_like(x)) for i,x in enumerate(initial)], {
                         'multiply_calls':0}
    monkeypatch.setattr(survey, 'solve_batch', fake_solve)
    report = survey.run(args)
    assert calls == [4,2]  # Positive successes never suppress negative searches.
    assert report['summary'][2]['paired_sign_outcomes']['positive_only'] == 3
    with np.load(args.output/'chunk-0-inputs.npz') as inputs:
        np.testing.assert_array_equal(inputs['initial'][::2], -inputs['initial'][1::2])
    for row in report['runs']:
        assert [a['kind'] for a in row['attempts']] == ['positive_random_normal','negative_random_normal']
    saved = json.loads((args.output/'results.json').read_text())
    assert saved['configuration']['paired_signs'] is True
    recovered = survey.recover(args.output, args.output/'recovered.json')
    assert recovered['summary'] == report['summary']


def test_interrupted_pair_is_not_reported_as_neither(tmp_path, monkeypatch):
    args = Namespace(output=tmp_path/'stopped', backend='cpu', levels=[2], matrices_per_level=2,
        starts=2, paired_signs=True, support=3, seed=73, batch_size=4, maxiter=2,
        gmres_restart=3, gmres_cycles=1, line_search=3, tol=1e-9, minutes=1., progress_seconds=0)
    original = survey.solve_batch
    def interrupted(*a, **kw):
        signal.raise_signal(signal.SIGINT)
        return original(*a, **kw)
    monkeypatch.setattr(survey, 'solve_batch', interrupted)
    report = survey.run(args)
    assert report['status']=='interrupted'
    counts = report['summary'][2]['paired_sign_outcomes']
    assert counts['incomplete']==2 and counts['neither_verified']==0
