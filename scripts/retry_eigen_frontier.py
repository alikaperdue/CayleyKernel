# SPDX-License-Identifier: Apache-2.0
"""Retry unresolved eigen matrices with new starts; archive originals first."""
import argparse
import hashlib
import json
import shutil
import signal
import time
from pathlib import Path

import numpy as np
from scripts.paper_quadratic_frontier import Products, newton
from scripts.batched_eigen_frontier import starting_points, retry_group, add_controls, controls


def retry_case(record, saved_x, backend, starts, seed, maxiter, tol, restart=40, cycles=15, line_search=12):
    dim = record['dimension']
    P = Products(dim, backend)
    coefficients = {}
    for name, values in record['coefficients'].items():
        v = np.zeros(dim)
        for index, value in values.items():
            v[int(index)] = value
        coefficients[name] = v
    a, b, c, d = (coefficients[k] for k in 'abcd')
    B, D, RA = P.fixed(b), P.fixed(d), P.fixed(a, False)
    equation = lambda x: P.full(x, B(x))+RA(x)-D(x)-c
    derivative = lambda x,h: P.full(h, B(x))+P.full(x, B(h))+RA(h)-D(h)
    attempts = []
    best_x, best_lambda = saved_x.copy(), a+B(saved_x)
    best_residual = float(np.linalg.norm(c+D(saved_x)-P.full(saved_x, best_lambda)))
    for i, (initial, label) in enumerate(starting_points(saved_x, starts, seed)):
        x, info = newton(equation, derivative, initial, tol=tol, maxiter=maxiter,
                         restart=restart, cycles=cycles, line_search=line_search)
        lam = a+B(x)
        residual = float(np.linalg.norm(c+D(x)-P.full(x, lam)))
        attempts.append({'start': i, 'kind': label, 'initial_norm': float(np.linalg.norm(initial)),
                         'eigenpair_residual': residual, 'solver': info})
        if residual < best_residual:
            best_x, best_lambda, best_residual = x, lam, residual
        if best_residual <= 10*tol:
            break
    return {'level': record['level'], 'seed': record['seed'], 'restart_seed': seed,
            'coefficients': record['coefficients'], 'attempts': attempts,
            'status': 'eigenpair_verified' if best_residual <= 10*tol else 'unresolved',
            'eigenpair_residual': best_residual}, best_x, best_lambda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['cpu','gpu'], default='gpu')
    parser.add_argument('--starts', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--maxiter', type=int, default=60)
    parser.add_argument('--tol', type=float, default=1e-9)
    add_controls(parser, default_solver='batched')
    parser.add_argument('--resume-saved', action='store_true',
                        help='Continue saved candidates before fresh starts (batched only)')
    parser.add_argument('--progress-seconds', type=float, default=30,
                        help='Elapsed heartbeat interval, 0 disables')
    parser.add_argument('--minutes', type=float, default=30, help='Cooperative search time ceiling')
    args = parser.parse_args()
    if not np.isfinite(args.minutes) or args.minutes <= 0:
        parser.error('minutes must be finite and positive')
    if min(args.batch_size, args.gmres_restart, args.gmres_cycles, args.line_search) < 1:
        parser.error("batch and correction controls must be positive")
    if args.resume_saved and args.solver != 'batched':
        parser.error('--resume-saved requires the batched solver')
    if not np.isfinite(args.progress_seconds) or args.progress_seconds < 0:
        parser.error('progress-seconds must be finite and nonnegative')
    source, output = args.input.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        parser.error('output must be outside the input directory')
    if output.exists():
        parser.error('output already exists; choose a new directory')
    if args.starts < (0 if args.resume_saved else 1) or args.maxiter < 1 or not np.isfinite(args.tol) or args.tol <= 0:
        parser.error('maxiter/tolerance must be positive; starts may be zero only with --resume-saved')
    original = json.loads((source/'results.json').read_text())
    records = [r for r in original['runs'] if r['status'] == 'unresolved']
    for r in original['runs']:
        path = (source/r['vector_file']).resolve()
        if source not in path.parents:
            parser.error('vector path is outside input directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != r['vector_sha256']:
            parser.error('original vector hash mismatch: '+str(path))
    if args.backend == 'cpu' and any(r['dimension'] > 256 for r in records):
        parser.error('CPU retries support dimension <=256; use GPU')
    Products(2, args.backend)  # device availability before writing output
    output.mkdir(parents=True)
    shutil.copytree(source, output/'original')
    # Byte-for-byte verification before any retries begin.
    for path in source.rglob('*'):
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(
                (output/'original'/path.relative_to(source)).read_bytes()).digest()
    report = {'configuration': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'original_backup': 'original', 'backup_verified': True,
              'selected_cases': len(records), 'runs': [],
              'meaning': 'Unresolved is not evidence of nonexistence.',
              'batched_source_sha256': hashlib.sha256(Path(__file__).with_name('batched_eigen_frontier.py').read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print(f'Retrying {len(records)} unresolved matrices with {args.solver}; output={output}', flush=True)
    from scripts.survey_eigen_frontier import atomic_json
    from scripts.benchmark_eigen_frontier import heartbeat
    destination = output/'results.json'
    report.update(status='running', chunks=[], batch_statistics={})
    atomic_json(destination, report)
    started = time.monotonic()
    deadline = started+args.minutes*60
    interrupted = False
    def interrupt(signum, frame):
        nonlocal interrupted
        if interrupted: raise KeyboardInterrupt
        interrupted = True
        print('STOP requested: saving current retry chunk before exit.', flush=True)
    previous_handler = signal.signal(signal.SIGINT, interrupt)
    def should_stop():
        return interrupted or time.monotonic() >= deadline
    done = set()
    try:
        groups = []
        for dimension in sorted({r['dimension'] for r in records}):
            indices = [i for i,r in enumerate(records) if r['dimension']==dimension]
            groups.extend([indices[j:j+args.batch_size] for j in range(0,len(indices),args.batch_size)])
        for chunk_index, indices in enumerate(groups):
            if should_stop(): break
            group = [records[i] for i in indices]
            saved = []
            for r in group:
                with np.load(output/'original'/r['vector_file'], allow_pickle=False) as vectors:
                    saved.append(vectors['root'].copy())
            label = f'[{args.backend} retry chunk={chunk_index} dim={group[0]["dimension"]} rows={len(group)}]'
            print(f'{label} START; saved={len(done)}/{len(records)}', flush=True)
            def progress(message): print(f'{label} {message}', flush=True)
            chunk_started = time.monotonic()
            if args.solver == 'batched':
                with heartbeat(label, args.progress_seconds):
                    results, stats = retry_group(group, saved, args.backend, args.starts, args.seed,
                        resume_saved=args.resume_saved, progress=progress, should_stop=should_stop, **controls(args))
            else:
                results, stats = [], {}
                for r,x in zip(group,saved):
                    if should_stop(): break
                    result, root, lam = retry_case(r,x,args.backend,args.starts,args.seed+r['seed'],
                        args.maxiter,args.tol,args.gmres_restart,args.gmres_cycles,args.line_search)
                    results.append({**result, '_root':root, '_lambda':lam})
            chunk_runs = []
            for index, r, result in zip(indices,group,results):
                x, lam = result.pop('_root'), result.pop('_lambda')
                result['previous_eigenpair_residual'] = r.get('eigenpair_residual')
                result['selected_candidate'] = r.get('selected_candidate')
                result['matrix_id'] = r.get('matrix_id', f"level-{r['level']}-seed-{r['seed']}")
                if 'generation_seed' in r: result['generation_seed'] = r['generation_seed']
                filename = f'case-{index}-level-{r["level"]}-seed-{r["seed"]}.npz'
                np.savez_compressed(output/filename, root=x, **{'lambda':lam})
                result.update(dimension=r['dimension'], vector_file=filename,
                    vector_sha256=hashlib.sha256((output/filename).read_bytes()).hexdigest())
                chunk_runs.append(result)
                done.add(index)
                print(f'{result["matrix_id"]}: {result["status"]}; residual {result["eigenpair_residual"]}', flush=True)
            chunk = {'chunk':chunk_index, 'dimension':group[0]['dimension'],
                     'elapsed_seconds':time.monotonic()-chunk_started, **stats}
            atomic_json(output/f'chunk-{chunk_index}.json', {'chunk':chunk,'runs':chunk_runs})
            report['runs'].extend(chunk_runs)
            report['chunks'].append(chunk)
            report['batch_statistics'][str(chunk_index)] = stats
            report['elapsed_seconds'] = time.monotonic()-started
            atomic_json(destination, report)
            print(f'{label} SAVED: {len(done)}/{len(records)} processed, '
                  f'{sum(r["status"]=="eigenpair_verified" for r in report["runs"])} newly verified; '
                  f'elapsed={report["elapsed_seconds"]:.1f}s', flush=True)
        report['status'] = 'interrupted' if interrupted else 'complete' if len(done)==len(records) else 'time_limit'
    except BaseException as exc:
        report['status'] = 'forced_interrupt' if isinstance(exc,KeyboardInterrupt) else 'error'
        report['error'] = str(exc)
        raise
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        report['elapsed_seconds'] = time.monotonic()-started
        report['unprocessed_matrix_ids'] = [r.get('matrix_id',f"level-{r['level']}-seed-{r['seed']}")
                                          for i,r in enumerate(records) if i not in done]
        atomic_json(destination, report)
        print(f'RETRY {report["status"]}: {len(done)}/{len(records)} saved; '
              f'{sum(r["status"]=="eigenpair_verified" for r in report["runs"])} newly verified; results={destination}', flush=True)


if __name__ == '__main__':
    main()
