# SPDX-License-Identifier: Apache-2.0
"""Compare historical SciPy Newton with coordinated batches on identical inputs."""
import argparse
import hashlib
import json
import platform
import time
import threading
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import scipy
import CayleyKernel as CK
from scripts.paper_quadratic_frontier import Products, coefficient, newton
from scripts.batched_eigen_frontier import BatchProducts, solve_batch, verify


class CountedProducts(Products):
    """Count fixed CPU applications too; GPU fixed() already calls full()."""
    def __init__(self, dim, backend):
        super().__init__(dim, backend)
        self.fixed_calls = 0

    def fixed(self, a, left=True):
        original = super().fixed(a, left)
        if self.backend == 'gpu': return original
        def apply(x):
            self.fixed_calls += 1
            return original(x)
        return apply


def individual(coeff, initial, backend, tol, maxiter, restart, cycles, line_search, **_):
    P = CountedProducts(initial.shape[1], backend)
    check = BatchProducts(initial.shape[1], backend)
    results = []
    for i, start in enumerate(initial):
        a, b, c, d = (v[i] for v in coeff)
        B, D, RA = P.fixed(b), P.fixed(d), P.fixed(a, False)
        f = lambda x: P.full(x, B(x))+RA(x)-D(x)-c
        jac = lambda x,h: P.full(h,B(x))+P.full(x,B(h))+RA(h)-D(h)
        x, info = newton(f, jac, start, tol, maxiter, restart, cycles, line_search)
        lam, rows = verify(check, tuple(check.xp.asarray(v[i:i+1]) for v in coeff), check.xp.asarray(x[None]))
        rows = check.host(rows)[0]
        valid = bool(np.all(np.isfinite(rows)) and max(rows) <= 10*tol)
        results.append({'status': 'eigenpair_verified' if valid else 'unresolved', 'solver': info,
                        'row_residuals': [float(v) if np.isfinite(v) else None for v in rows], '_root': x, '_lambda': check.host(lam)[0]})
    calls = P.calls+P.fixed_calls+sum(check.batch_sizes.values())
    return results, {'multiply_calls': calls, 'multiply_rows': calls, 'batch_sizes': {1: calls},
                     'cpu_fixed_plan_applications': P.fixed_calls}


def sync(backend):
    if backend == 'gpu': CK.cp.cuda.get_current_stream().synchronize()



@contextmanager
def heartbeat(label, interval):
    """Elapsed-time notice only; never touches CUDA or solver state."""
    stop = threading.Event()
    started = time.perf_counter()
    def announce():
        while not stop.wait(interval):
            print(f'{label} RUNNING: {time.perf_counter()-started:.1f}s elapsed '
                  '(measurement not yet complete)', flush=True)
    thread = None
    if interval > 0:
        thread = threading.Thread(target=announce, daemon=True)
        thread.start()
    try:
        yield
    finally:
        stop.set()
        if thread is not None:
            thread.join()


def apply_profile(args):
    defaults = dict(level=3, matrices=16, starts=2, batch_size=32,
                    maxiter=25, gmres_restart=40, gmres_cycles=15, repeats=3)
    if args.screen:
        defaults.update(level=6, matrices=32, batch_size=64, maxiter=15,
                        gmres_restart=20, gmres_cycles=3, repeats=1)
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--level', type=int, default=None)
    parser.add_argument('--matrices', type=int, default=None)
    parser.add_argument('--starts', type=int, default=None)
    parser.add_argument('--seed', type=int, default=23613)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--maxiter', type=int, default=None)
    parser.add_argument('--gmres-restart', type=int, default=None)
    parser.add_argument('--gmres-cycles', type=int, default=None)
    parser.add_argument('--line-search', type=int, default=12)
    parser.add_argument('--tol', type=float, default=1e-9)
    parser.add_argument('--repeats', type=int, default=None)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--screen', action='store_true',
                        help='Short level-6 screen: 32 matrices, 2 starts, batch 64, Newton 15, '
                             'GMRES restart 20/cycles 3, one steady repeat; explicit options override')
    parser.add_argument('--progress-seconds', type=float, default=30,
                        help='Elapsed-time heartbeat interval; 0 disables (default: 30)')
    args = apply_profile(parser.parse_args())
    if not np.isfinite(args.progress_seconds) or args.progress_seconds < 0:
        parser.error('progress-seconds must be finite and nonnegative')
    if not 1 <= args.level <= 20 or min(args.matrices, args.starts, args.batch_size, args.maxiter,
            args.gmres_restart, args.gmres_cycles, args.line_search, args.repeats) < 1 or not np.isfinite(args.tol) or args.tol <= 0:
        parser.error('invalid dimensions or controls')
    if args.output.exists(): parser.error('output exists; choose a new directory')
    run_label = f'[{args.backend} level={args.level} dim={1 << args.level} batch={args.batch_size}]'
    print(f'{run_label} SETUP: matrices={args.matrices}, starts={args.starts}, '
          f'searches={args.matrices*args.starts}, initial_chunk={min(args.batch_size, args.matrices*args.starts)}, '
          f'Newton={args.maxiter}, GMRES={args.gmres_restart}x{args.gmres_cycles}, '
          f'steady_repeats={args.repeats}; output={args.output}', flush=True)
    if args.screen:
        print(f'{run_label} SCREEN: reduced budgets; compare solvers within this run, '
              'not success counts against earlier full-budget runs.', flush=True)
    setup = time.perf_counter()
    Products(1 << args.level, args.backend)
    sync(args.backend)
    device_setup = time.perf_counter()-setup
    setup = time.perf_counter()
    coeff, points, ids = [], [], []
    for matrix in range(args.matrices):
        seed = args.seed + 1000*args.level + matrix
        rng = np.random.default_rng(seed)
        co = [coefficient(rng, 1 << args.level) for _ in range(4)]
        for start in range(args.starts):
            coeff.append(co)
            points.append(np.zeros(1 << args.level) if start == 0 else rng.normal(size=1 << args.level)/np.sqrt(1 << args.level))
            ids.append({'matrix_id': f'level-{args.level}-seed-{seed}', 'seed': seed, 'start': start})
    coeff = tuple(np.stack(v) for v in zip(*coeff))
    points = np.stack(points)
    generation = time.perf_counter()-setup
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'inputs.npz', **dict(zip('abcd', coeff)), initial=points)
    config = dict(backend=args.backend, tol=args.tol, maxiter=args.maxiter, restart=args.gmres_restart,
                  cycles=args.gmres_cycles, line_search=args.line_search, batch_size=args.batch_size)
    report = {'configuration': {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
              'kernel_library': CK.library_path(), 'device_setup_seconds': device_setup,
              'input_generation_seconds': generation, 'inputs_sha256': hashlib.sha256((args.output/'inputs.npz').read_bytes()).hexdigest(),
              'searches': ids, 'measurements': [],
              'timing_scope': 'Solve, transfers, coefficient plans, diagnostics, and both-row verification. Excludes input generation and disk IO.',
              'work_policy': 'Every identical starting point is solved in both paths, including siblings of successful starts; unique matrix counts also reported.',
              'cpu_caveat': 'Individual uses historical signed-permutation fixed plans; batched CPU uses recursive oracle. CPU timing is not GPU evidence.',
              'source_sha256': {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                for name in ('benchmark_eigen_frontier.py', 'batched_eigen_frontier.py', 'paper_quadratic_frontier.py')}}
    if args.backend == 'gpu':
        report['gpu'] = str(CK.cp.cuda.runtime.getDeviceProperties(CK.cp.cuda.Device().id))
        report['cupy'] = CK.cp.__version__
    destination = args.output/'results.json'
    report['status'] = 'running'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    # First invocation of each solver is explicitly cold; shared CUDA context means
    # these are first-use costs, not independent fresh-process cold measurements.
    for repeat in range(-1, args.repeats):
        order = [('individual', individual), ('batched', solve_batch)]
        if repeat % 2 == 1 and repeat >= 0: order.reverse()
        for name, solver in order:
            phase = 'first_use' if repeat == -1 else f'steady_state {repeat+1}/{args.repeats}'
            label = f'{run_label} {name} {phase}'
            report['active_measurement'] = {'solver': name, 'repeat': repeat, 'phase': phase}
            destination.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print(f'{label} START', flush=True)
            try:
                with heartbeat(label, args.progress_seconds):
                    sync(args.backend)
                    started = time.perf_counter()
                    results, stats = solver(coeff, points, **config)
                    sync(args.backend)
                    elapsed = time.perf_counter()-started
            except KeyboardInterrupt:
                report['status'] = 'interrupted'
                destination.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                print(f'{label} INTERRUPTED: completed measurements preserved in {destination}; '
                      'partial measurement excluded.', flush=True)
                raise SystemExit(130) from None
            verified = {ids[i]['matrix_id'] for i,r in enumerate(results) if r['status'] == 'eigenpair_verified'}
            entry = {'solver': name, 'phase': 'first_use' if repeat == -1 else 'steady_state',
                     'repeat': repeat, 'end_to_end_seconds': elapsed, **stats,
                     'verified_searches': sum(r['status'] == 'eigenpair_verified' for r in results),
                     'unresolved_searches': sum(r['status'] != 'eigenpair_verified' for r in results),
                     'verified_matrices': len(verified), 'unresolved_matrices': args.matrices-len(verified),
                     'newton_iterations': [r['solver']['newton_steps'] for r in results],
                     'krylov_iterations': [r['solver']['krylov_iterations'] for r in results],
                     'row_residuals': [r['row_residuals'] for r in results],
                     'solver_diagnostics': [{**r['solver'], 'residuals': [float(v) if v is not None and np.isfinite(v) else None
                         for v in r['solver']['residuals']]} for r in results]}
            filename = f'{name}-{repeat}.npz'
            np.savez_compressed(args.output/filename, root=np.stack([r['_root'] for r in results]),
                                **{'lambda': np.stack([r['_lambda'] for r in results])})
            entry.update(vector_file=filename, vector_sha256=hashlib.sha256((args.output/filename).read_bytes()).hexdigest())
            report['measurements'].append(entry)
            report.pop('active_measurement', None)
            (args.output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print(f'{label} DONE: {elapsed:.6f}s, {entry["verified_searches"]} verified / '
                  f'{entry["unresolved_searches"]} unresolved starts; {stats["multiply_calls"]} calls; '
                  f'Newton={sum(entry["newton_iterations"])}, Krylov={sum(entry["krylov_iterations"])}', flush=True)
    report['steady_state_median_seconds'] = {
        name: float(np.median([m['end_to_end_seconds'] for m in report['measurements']
                              if m['solver'] == name and m['phase'] == 'steady_state']))
        for name in ('individual', 'batched')}
    report['status'] = 'complete'
    (args.output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    medians = report['steady_state_median_seconds']
    print(f'{run_label} COMPLETE: median individual={medians["individual"]:.6f}s, '
          f'batched={medians["batched"]:.6f}s; results={destination}', flush=True)


if __name__ == '__main__':
    main()
