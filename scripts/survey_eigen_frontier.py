# SPDX-License-Identifier: Apache-2.0
"""Finite fresh-matrix survey with incremental archives and cooperative stopping."""
import argparse
import hashlib
import json
import os
import signal
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scripts.paper_quadratic_frontier import Products, coefficient, encoded
from scripts.batched_eigen_frontier import solve_batch
from scripts.benchmark_eigen_frontier import heartbeat


def atomic_json(path, value):
    """Replace a manifest only after its new contents have been flushed."""
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def make_chunk(level, offset, count, seed, starts, support, paired_signs=False):
    if paired_signs and starts != 2:
        raise ValueError('paired signs requires exactly two starts')
    dim = 1 << level
    coefficients, initial, cases = [], [], []
    for sample in range(offset, offset+count):
        # Separate RNG streams by level and matrix; chunk size cannot affect data.
        rng = np.random.default_rng(np.random.SeedSequence([seed, level, sample]))
        values = [coefficient(rng, dim, support) for _ in range(4)]
        cases.append({'matrix_id': f'survey-{seed}-level-{level}-sample-{sample}',
                      'seed': seed+sample, 'generation_seed': [seed, level, sample],
                      'level': level, 'dimension': dim,
                      'coefficients': {k:encoded(v) for k,v in zip('abcd',values)}})
        point = rng.normal(size=dim)/np.sqrt(dim) if paired_signs else None
        for start in range(starts):
            coefficients.append(values)
            initial.append((point if start == 0 else -point) if paired_signs else
                           np.zeros(dim) if start == 0 else rng.normal(size=dim)/np.sqrt(dim))
    return tuple(np.stack(v) for v in zip(*coefficients)), np.stack(initial), cases


def run(args):
    if args.output.exists(): raise ValueError('output exists; choose a new directory')
    Products(2, args.backend)
    args.output.mkdir(parents=True)
    report = {'configuration': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'status': 'running', 'runs': [], 'chunks': [], 'summary': {},
              'meaning': 'Finite random sparse-matrix survey. Unresolved does not imply nonexistence.',
              'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                  for name in ('survey_eigen_frontier.py','batched_eigen_frontier.py','paper_quadratic_frontier.py')}}
    destination = args.output/'results.json'
    atomic_json(destination, report)
    def save_status():
        atomic_json(args.output/'status.json', {k:report[k] for k in
            ('status','summary','elapsed_seconds','active_chunk') if k in report} |
            {'completed_chunks':len(report['chunks']), 'saved_matrices':len(report['runs'])})
    save_status()
    started = time.monotonic()
    deadline = started+args.minutes*60
    interrupted = False
    def interrupt(signum, frame):
        nonlocal interrupted
        if interrupted:
            raise KeyboardInterrupt
        interrupted = True
        print('STOP requested: finishing current operation, verifying and saving current candidates. '
              'A second Ctrl-C forces exit; prior checkpoints remain.', flush=True)
    previous = signal.signal(signal.SIGINT, interrupt)
    def should_stop():
        return interrupted or time.monotonic() >= deadline
    offsets = {level: 0 for level in args.levels}
    matrices_per_chunk = max(1, args.batch_size//args.starts)
    print(f'SURVEY: levels={args.levels}, target={args.matrices_per_level} matrices/level, '
          f'{args.starts} starts/matrix; limit={args.minutes:g} minutes; output={args.output}', flush=True)
    print('Alternating levels each chunk; stop at target or time limit. Ctrl-C once saves current work.', flush=True)
    try:
        while not should_stop() and any(n < args.matrices_per_level for n in offsets.values()):
            for level in args.levels:
                if should_stop(): break
                offset = offsets[level]
                if offset >= args.matrices_per_level: continue
                count = min(matrices_per_chunk, args.matrices_per_level-offset)
                chunk_id = len(report['chunks'])
                label = f'[survey chunk={chunk_id} level={level} matrices={offset}:{offset+count}]'
                print(f'{label} START; elapsed={time.monotonic()-started:.1f}s', flush=True)
                coeff, initial, cases = make_chunk(level, offset, count, args.seed, args.starts, args.support,
                                                   paired_signs=getattr(args, 'paired_signs', False))
                input_name = f'chunk-{chunk_id}-inputs.npz'
                np.savez_compressed(args.output/input_name, **dict(zip('abcd',coeff)), initial=initial)
                report['active_chunk'] = {'chunk':chunk_id, 'level':level, 'offset':offset,
                    'matrices':count, 'input_file':input_name,
                    'input_sha256':hashlib.sha256((args.output/input_name).read_bytes()).hexdigest()}
                save_status()
                chunk_started = time.monotonic()
                with heartbeat(label, args.progress_seconds):
                    results, stats = solve_batch(coeff, initial, backend=args.backend, batch_size=args.batch_size,
                        maxiter=args.maxiter, restart=args.gmres_restart, cycles=args.gmres_cycles,
                        line_search=args.line_search, tol=args.tol, should_stop=should_stop)
                elapsed = time.monotonic()-chunk_started
                # All starts are archived; pick a verified candidate first, otherwise lowest residual.
                output_name = f'chunk-{chunk_id}-searches.npz'
                np.savez_compressed(args.output/output_name, root=np.stack([r['_root'] for r in results]),
                                    **{'lambda':np.stack([r['_lambda'] for r in results])})
                chunk_runs = []
                for i, case in enumerate(cases):
                    group = results[i*args.starts:(i+1)*args.starts]
                    best_index = min(range(len(group)), key=lambda j: (
                        group[j]['status'] != 'eigenpair_verified',
                        group[j]['eigenpair_residual'] if group[j]['eigenpair_residual'] is not None else float('inf')))
                    best = group[best_index]
                    filename = f'level-{level}-sample-{offset+i}.npz'
                    np.savez_compressed(args.output/filename, root=best['_root'], **{'lambda':best['_lambda']})
                    row = {**case, **{k:v for k,v in best.items() if not k.startswith('_')},
                        'selected_start': best_index, 'attempts': [dict(start=j,
                            kind=(('positive_random_normal' if j==0 else 'negative_random_normal')
                                  if getattr(args, 'paired_signs', False) else
                                  'zero' if j==0 else 'random_normal'),
                            **{k:v for k,v in r.items() if not k.startswith('_')}) for j,r in enumerate(group)],
                        'vector_file': filename, 'vector_sha256':hashlib.sha256((args.output/filename).read_bytes()).hexdigest(),
                        'chunk':chunk_id}
                    chunk_runs.append(row)
                report['runs'].extend(chunk_runs)
                info = report.pop('active_chunk')
                info.update(elapsed_seconds=elapsed, **stats, search_vector_file=output_name,
                    search_vector_sha256=hashlib.sha256((args.output/output_name).read_bytes()).hexdigest())
                report['chunks'].append(info)
                offsets[level] += count
                merge_summary(report['summary'], summarize(chunk_runs))
                report['elapsed_seconds'] = time.monotonic()-started
                # The separate chunk manifest survives a damaged or interrupted aggregate update.
                atomic_json(args.output/f'chunk-{chunk_id}.json', {'chunk':info, 'runs':chunk_runs})
                save_status()
                verified = sum(r['status']=='eigenpair_verified' for r in chunk_runs)
                reasons = Counter(a['solver']['reason'] for r in chunk_runs for a in r['attempts'])
                print(f'{label} SAVED: {verified}/{count} matrices verified, {count-verified} unresolved; '
                      f'{elapsed:.2f}s; calls={stats["multiply_calls"]}; '
                      f'stalled_starts={reasons["line search stalled"]}, budget_starts={reasons["iteration budget"]}; '
                      f'total={len(report["runs"])} matrices, '
                      f'verified={sum(v["verified"] for v in report["summary"].values())}; '
                      f'elapsed={report["elapsed_seconds"]:.1f}s', flush=True)
        report['status'] = ('interrupted' if interrupted else 'complete' if
            all(n >= args.matrices_per_level for n in offsets.values()) else 'time_limit')
    except BaseException as exc:
        report['status'] = 'forced_interrupt' if isinstance(exc, KeyboardInterrupt) else 'error'
        report['error'] = str(exc)
        raise
    finally:
        signal.signal(signal.SIGINT, previous)
        report['elapsed_seconds'] = time.monotonic()-started
        atomic_json(destination, report)
        save_status()
        print(f'SURVEY {report["status"]}: {len(report["runs"])} matrices saved; results={destination}', flush=True)
        for level, summary in report['summary'].items():
            print(f'LEVEL {level}: {summary["verified"]}/{summary["matrices"]} matrices verified, '
                  f'{summary["unresolved"]} unresolved; later-start rescues={summary["rescued_by_later_start"]}; '
                  f'stopped starts={summary["stopped_searches"]}', flush=True)
            if 'paired_sign_outcomes' in summary:
                print(f'LEVEL {level} paired signs: {summary["paired_sign_outcomes"]}', flush=True)
    return report



def merge_summary(total, addition):
    for level, row in addition.items():
        if level not in total:
            total[level] = row
            continue
        target = total[level]
        for key in ('matrices','verified','unresolved','rescued_by_later_start','stopped_searches'):
            target[key] += row[key]
        target['verified_by_start'] = [a+b for a,b in zip(target['verified_by_start'],row['verified_by_start'])]
        target['termination_reasons'] = dict(Counter(target['termination_reasons'])+Counter(row['termination_reasons']))
        if 'paired_sign_outcomes' in row:
            for outcome, count in row['paired_sign_outcomes'].items():
                target['paired_sign_outcomes'][outcome] += count


def recover(source, output):
    """Reconstruct an aggregate from committed chunks without altering the source."""
    if output.exists(): raise ValueError('recovery output exists')
    if output.resolve().parent != source.resolve():
        raise ValueError('recovery output must be alongside the referenced vectors')
    report = json.loads((source/'results.json').read_text())
    report.update(runs=[], chunks=[], summary={}, status='recovered')
    report.pop('active_chunk', None)
    for path in sorted(source.glob('chunk-*.json'), key=lambda p:int(p.stem.split('-')[1])):
        checkpoint = json.loads(path.read_text())
        if checkpoint['chunk']['chunk'] != len(report['chunks']):
            raise ValueError('Noncontiguous checkpoint sequence')
        for row in checkpoint['runs']:
            vector = (source/row['vector_file']).resolve()
            if source.resolve() not in vector.parents or hashlib.sha256(vector.read_bytes()).hexdigest()!=row['vector_sha256']:
                raise ValueError('Invalid checkpoint vector')
        report['chunks'].append(checkpoint['chunk'])
        report['runs'].extend(checkpoint['runs'])
        merge_summary(report['summary'], summarize(checkpoint['runs']))
    atomic_json(output, report)
    return report

def summarize(runs):
    summary = {}
    for level in sorted({r['level'] for r in runs}):
        rows = [r for r in runs if r['level']==level]
        solved = sum(r['status']=='eigenpair_verified' for r in rows)
        summary[level] = {'matrices':len(rows), 'verified':solved, 'unresolved':len(rows)-solved,
            'rescued_by_later_start': sum(r['status']=='eigenpair_verified' and
                r['attempts'][0]['status']!='eigenpair_verified' for r in rows),
            'verified_by_start': [sum(r['attempts'][j]['status']=='eigenpair_verified' for r in rows)
                                 for j in range(len(rows[0]['attempts']))],
            'termination_reasons': dict(Counter(a['solver']['reason'] for r in rows for a in r['attempts'])),
            'stopped_searches': sum(a['solver']['reason']=='run stopped' for r in rows for a in r['attempts'])}
        if rows[0]['attempts'][0]['kind'] == 'positive_random_normal':
            counts = dict.fromkeys(('both_verified', 'positive_only', 'negative_only',
                                    'neither_verified', 'incomplete'), 0)
            for row in rows:
                positive, negative = row['attempts']
                if any(a['solver']['reason']=='run stopped' for a in (positive, negative)):
                    outcome = 'incomplete'
                else:
                    flags = tuple(a['status']=='eigenpair_verified' for a in (positive, negative))
                    outcome = {(True,True):'both_verified', (True,False):'positive_only',
                               (False,True):'negative_only', (False,False):'neither_verified'}[flags]
                counts[outcome] += 1
            summary[level]['paired_sign_outcomes'] = counts
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu','gpu'], default='gpu')
    parser.add_argument('--levels', default='6,8')
    parser.add_argument('--matrices-per-level', type=int, default=256)
    parser.add_argument('--starts', type=int, default=2)
    parser.add_argument('--paired-signs', action='store_true',
                        help='Run both x0 and -x0 for every matrix; requires --starts 2')
    parser.add_argument('--support', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--maxiter', type=int, default=20)
    parser.add_argument('--gmres-restart', type=int, default=20)
    parser.add_argument('--gmres-cycles', type=int, default=3)
    parser.add_argument('--line-search', type=int, default=12)
    parser.add_argument('--tol', type=float, default=1e-9)
    parser.add_argument('--minutes', type=float, default=30)
    parser.add_argument('--progress-seconds', type=float, default=30)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try: args.levels = [int(v) for v in args.levels.split(',')]
    except ValueError: parser.error('levels must be comma-separated integers')
    if not args.levels or len(set(args.levels)) != len(args.levels) or any(n<1 or n>20 for n in args.levels):
        parser.error('levels must be distinct integers from 1 to 20')
    if args.backend=='cpu' and max(args.levels)>8: parser.error('CPU limited to level 8')
    if args.paired_signs and args.starts != 2: parser.error('--paired-signs requires --starts 2')
    if min(args.matrices_per_level,args.starts,args.support,args.batch_size,args.maxiter,
           args.gmres_restart,args.gmres_cycles,args.line_search)<1 or args.seed<0 or args.starts>args.batch_size:
        parser.error('invalid counts; starts must not exceed batch size')
    if not all(np.isfinite(v) for v in (args.tol,args.minutes,args.progress_seconds)) or min(args.tol,args.minutes)<=0 or args.progress_seconds<0:
        parser.error('invalid tolerance/time controls')
    if args.output.exists(): parser.error('output exists; choose a new directory')
    run(args)


if __name__ == '__main__':
    main()
