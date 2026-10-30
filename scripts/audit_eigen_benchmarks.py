# SPDX-License-Identifier: Apache-2.0
"""Audit archived search diagnostics and prepare unresolved matrices for retry.

No algebra products or solver runs: this audits outcomes, provenance and budgets.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np


def checked_file(source, name, digest):
    path = (source/name).resolve()
    if source not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Invalid path or hash: {path}')
    return path


def audit(sources, output):
    output = Path(output).resolve()
    sources = [Path(p).resolve() for p in sources]
    if output.exists() or any(output == p or p in output.parents for p in sources):
        raise ValueError('Choose a new output directory outside all inputs')
    summaries, matrices = [], {}
    search_outcomes = {}
    for source_index, source in enumerate(sources):
        report = json.loads((source/'results.json').read_text())
        cfg, searches = report['configuration'], report['searches']
        threshold = 10*cfg['tol']
        with np.load(checked_file(source, 'inputs.npz', report['inputs_sha256']), allow_pickle=False) as data:
            coeff = [data[k].copy() for k in 'abcd']
            initial = data['initial'].copy()
        if any(v.shape != initial.shape for v in coeff) or len(initial) != len(searches):
            raise ValueError('Input/search shape mismatch')
        measurements, masks, vectors = [], [], []
        for measurement_index, m in enumerate(report['measurements']):
            rows = np.asarray([[np.nan if v is None else v for v in row] for row in m['row_residuals']])
            if rows.shape != (len(searches), 2): raise ValueError('Row residual shape mismatch')
            ok = np.isfinite(rows).all(axis=1) & (rows >= 0).all(axis=1) & (rows <= threshold).all(axis=1)
            solved = {s['matrix_id'] for s, valid in zip(searches, ok) if valid}
            if (int(ok.sum()) != m['verified_searches'] or len(solved) != m['verified_matrices'] or
                    len(searches)-int(ok.sum()) != m['unresolved_searches']):
                raise ValueError('Stored counts disagree with per-search residuals')
            with np.load(checked_file(source, m['vector_file'], m['vector_sha256']), allow_pickle=False) as data:
                roots, lambdas = data['root'].copy(), data['lambda'].copy()
            if roots.shape != initial.shape or lambdas.shape != initial.shape:
                raise ValueError('Output vector shape mismatch')
            if not (np.isfinite(roots[ok]).all() and np.isfinite(lambdas[ok]).all()):
                raise ValueError('Nonfinite verified vector')
            masks.append(ok); vectors.append(roots)
            unresolved = ~ok
            budget = np.asarray(m['newton_iterations']) >= cfg['maxiter']
            measurements.append({'solver': m['solver'], 'phase': m['phase'], 'repeat': m['repeat'],
                                 'seconds': m['end_to_end_seconds'], 'verified_starts': int(ok.sum()),
                                 'verified_matrices': len(solved), 'unresolved_starts_at_newton_budget': int((unresolved & budget).sum()),
                                 'unresolved_starts_below_newton_budget': int((unresolved & ~budget).sum()),
                                 'max_verified_row_residual': float(rows[ok].max()) if ok.any() else None})
            for i, search in enumerate(searches):
                key = search['matrix_id']
                search_key = (key, search['start'], hashlib.sha256(initial[i].tobytes()).hexdigest(), threshold)
                search_outcomes.setdefault(search_key, []).append({'source': source_index,
                    'measurement': measurement_index, 'verified': bool(ok[i])})
                values = [v[i] for v in coeff]
                if key not in matrices:
                    matrices[key] = {'coeff': values, 'seed': search['seed'], 'level': cfg['level'],
                                     'verified': False, 'best': None, 'observations': 0}
                matrix = matrices[key]
                if matrix['seed'] != search['seed'] or any(not np.array_equal(a,b) for a,b in zip(matrix['coeff'], values)):
                    raise ValueError(f'Matrix identifier collision: {key}')
                matrix['verified'] |= bool(ok[i]); matrix['observations'] += 1
                score = float(rows[i].max()) if np.isfinite(rows[i]).all() else float('inf')
                if not (np.isfinite(roots[i]).all() and np.isfinite(lambdas[i]).all()): continue
                if matrix['best'] is None or score < matrix['best']['score']:
                    matrix['best'] = {'score': score, 'root': roots[i], 'lambda': lambdas[i],
                        'rows': m['row_residuals'][i], 'source': source_index, 'measurement': measurement_index,
                        'solver': m['solver'], 'start': search['start'], 'search_index': i,
                        'newton_steps': m['newton_iterations'][i], 'krylov_iterations': m['krylov_iterations'][i]}
        disagreements = []
        for i, mask in enumerate(masks):
            changed = np.flatnonzero(mask != masks[0]).tolist()
            if changed:
                disagreements.append({'measurement': i, 'searches': [searches[j] for j in changed]})
        # Vector distance is informational: different roots can both be valid.
        distances = []
        for repeat in sorted({m['repeat'] for m in report['measurements']}):
            pair = {m['solver']: i for i,m in enumerate(report['measurements']) if m['repeat'] == repeat}
            if not {'individual', 'batched'} <= pair.keys(): continue
            i, j = pair['individual'], pair['batched']
            both = masks[i] & masks[j]
            distances.append({'repeat': repeat, 'jointly_verified_starts': int(both.sum()),
                'max_relative_root_difference': float(np.max(np.linalg.norm(vectors[i][both]-vectors[j][both], axis=1)/
                    np.maximum(1, np.linalg.norm(vectors[i][both], axis=1)))) if both.any() else None})
        medians = {solver: float(np.median([m['seconds'] for m in measurements if m['solver']==solver and m['phase']=='steady_state']))
                   for solver in ('individual', 'batched') if any(m['solver']==solver and m['phase']=='steady_state' for m in measurements)}
        summaries.append({'source': str(source), 'level': cfg['level'], 'threshold': threshold,
                          'measurements': measurements, 'outcome_disagreements': disagreements,
                          'paired_vectors': distances, 'steady_state_medians': medians})
    # Validate every source before creating the output. Keep original archives intact.
    output.mkdir(parents=True)
    for index, source in enumerate(sources):
        shutil.copytree(source, output/'original'/f'benchmark-{index}')
    runs, unresolved_summary = [], []
    for index, (key, matrix) in enumerate(sorted(matrices.items())):
        best = matrix['best']
        if best is None: raise ValueError(f'No finite candidate saved for {key}')
        filename = f'matrix-{index}.npz'
        np.savez_compressed(output/filename, root=best['root'], **{'lambda': best['lambda']})
        provenance = {k:v for k,v in best.items() if k not in ('root','lambda','score','rows')}
        row = {'matrix_id': key, 'level': matrix['level'], 'dimension': len(best['root']), 'seed': matrix['seed'],
               'status': 'eigenpair_verified' if matrix['verified'] else 'unresolved',
               'coefficients': {k:{str(j):float(v[j]) for j in np.flatnonzero(v)} for k,v in zip('abcd',matrix['coeff'])},
               'eigenpair_residual': best['score'] if np.isfinite(best['score']) else None,
               'row_residuals': best['rows'], 'selected_candidate': provenance,
               'vector_file': filename, 'vector_sha256': hashlib.sha256((output/filename).read_bytes()).hexdigest()}
        runs.append(row)
        if not matrix['verified']:
            unresolved_summary.append({k:row[k] for k in ('matrix_id','eigenpair_residual','selected_candidate')})
    cross_disagreements = [{'matrix_id': key[0], 'start': key[1], 'observations': values}
        for key, values in search_outcomes.items() if len({v['verified'] for v in values}) > 1]
    result = {'cross_archive_outcome_disagreements': cross_disagreements,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'audit_scope': 'Archived diagnostics and vectors; no multiplication or equation recomputation.',
              'meaning': 'Unresolved searches are not evidence of nonexistence.',
              'benchmarks': summaries, 'unique_matrices': len(matrices),
              'verified_matrices': sum(m['verified'] for m in matrices.values()),
              'unresolved_matrices': unresolved_summary, 'runs': runs}
    (output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input, args.output)
    print(f"Audited {result['unique_matrices']} matrices: {result['verified_matrices']} verified, "
          f"{len(result['unresolved_matrices'])} unresolved. Retry input: {args.output}")
    for r in result['unresolved_matrices']:
        print(r['matrix_id'], 'best residual:', r['eigenpair_residual'])


if __name__ == '__main__':
    main()
