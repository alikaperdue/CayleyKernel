"""Retry selection and archival regression test."""
import hashlib
import json
import sys

import pytest
np = pytest.importorskip('numpy')
pytest.importorskip('scipy')
from scripts.retry_eigen_frontier import main

pytestmark = pytest.mark.cpu


def test_retry_archives_and_selects_unresolved(tmp_path, monkeypatch):
    source, output = tmp_path/'source', tmp_path/'retry'
    source.mkdir()
    # [[0,1],[-1,0]], original equation x²+1=0 in H.
    np.savez(source/'candidate.npz', root=np.array([0., .8, 0., 0.]))
    digest = hashlib.sha256((source/'candidate.npz').read_bytes()).hexdigest()
    record = dict(level=2, dimension=4, seed=1, status='unresolved',
                  coefficients={'a':{}, 'b':{'0':1}, 'c':{'0':-1}, 'd':{}},
                  vector_file='candidate.npz', vector_sha256=digest)
    (source/'results.json').write_text(json.dumps({'runs':[record,dict(record,status='eigenpair_verified')]}))
    original = (source/'results.json').read_bytes()
    monkeypatch.setattr(sys, 'argv', ['retry','--input',str(source),'--output',str(output),
                                    '--backend','cpu','--starts','2'])
    main()
    report = json.loads((output/'results.json').read_text())
    assert report['selected_cases'] == 1
    assert report['backup_verified']
    assert (output/'original/results.json').read_bytes() == original
    assert (source/'results.json').read_bytes() == original
    result = report['runs'][0]
    assert result['attempts'][0]['kind'] == 'negative_saved_candidate'
    assert result['status'] == 'eigenpair_verified'
    with np.load(output/result['vector_file']) as data:
        assert abs(np.linalg.norm(data['root'])-1) < 1e-8
        assert data['root'][1] < 0  # Starting branch really changed.
    with pytest.raises(SystemExit):
        main()  # Never overwrite the previous retry.
