# SPDX-License-Identifier: Apache-2.0
"""Coordinated real Newton/restarted GMRES searches; last axis is algebra."""
from collections import Counter
import numpy as np
import CayleyKernel as CK
from scripts.paper_quadratic_frontier import Products, reference_product


class BatchProducts:
    def __init__(self, dim, backend):
        Products(dim, backend)  # validate device before allocating
        if backend == 'cpu' and dim > 256:
            raise ValueError('CPU reference limited to dimension 256')
        self.dim, self.backend = dim, backend
        self.xp = CK.cp if backend == 'gpu' else np
        self.batch_sizes = Counter()

    def host(self, a):
        return CK.cp.asnumpy(a) if self.backend == 'gpu' else np.asarray(a)

    def full(self, a, b):
        if a.ndim != 2 or a.shape != b.shape or a.shape[1] != self.dim:
            raise ValueError('products require matching (batch_size, algebra_dimension) arrays')
        self.batch_sizes[len(a)] += 1
        if self.backend == 'gpu':
            return CK.multiply(a, b)
        # Intentionally slow, independent CPU oracle, never a GPU fallback.
        return np.stack([reference_product(u, v) for u, v in zip(a, b)])

    def stats(self):
        return {'multiply_calls': sum(self.batch_sizes.values()),
                'multiply_rows': sum(k*v for k, v in self.batch_sizes.items()),
                'batch_sizes': dict(sorted(self.batch_sizes.items()))}


def residual(P, coefficients, x):
    a, b, c, d = coefficients
    bx = P.full(b, x)
    return P.full(x, bx) + P.full(x, a) - P.full(d, x) - c, bx


def derivative(P, coefficients, x, bx, h):
    a, b, _, d = coefficients
    return P.full(h, bx) + P.full(x, P.full(b, h)) + P.full(h, a) - P.full(d, h)


def verify(P, coefficients, x):
    """Both rows in the paper convention: a+b*x=lambda, c+d*x=x*lambda."""
    xp = P.xp
    a, b, c, d = coefficients
    lam = a + P.full(b, x)
    one = xp.zeros_like(x); one[:, 0] = 1
    first = P.full(a, one) + P.full(b, x) - P.full(lam, one)
    second = P.full(c, one) + P.full(d, x) - P.full(x, lam)
    return lam, xp.stack((xp.linalg.norm(first, axis=1), xp.linalg.norm(second, axis=1)), axis=1)


class SearchStopped(Exception):
    """Cooperative stop carrying completed Krylov work for diagnostics."""
    def __init__(self, counts):
        self.counts = counts


def gmres_batch(P, coeff, x, bx, rhs, atol, restart=40, cycles=15, rtol=1e-7, should_stop=None):
    """Independent Arnoldi/Givens problems with coordinated Jacobian products.

    Compact Krylov-converged rows before Jacobian products; retain their small
    Hessenberg systems until the cycle ends. No per-row host solve or per-row multiply occurs.
    """
    xp = P.xp
    n, dim = rhs.shape
    restart = min(restart, dim)
    answer = xp.zeros_like(rhs)
    counts = xp.zeros(n, dtype=xp.int64)
    target = xp.maximum(atol, rtol*xp.linalg.norm(rhs, axis=1))
    active = xp.arange(n)
    for cycle in range(cycles):
        if not len(active):
            break
        co = tuple(v[active] for v in coeff)
        xx, bb = x[active], bx[active]
        r = rhs[active] if cycle == 0 else rhs[active]-derivative(P, co, xx, bb, answer[active])
        beta = xp.linalg.norm(r, axis=1)
        keep = beta > target[active]
        active, r, beta = active[keep], r[keep], beta[keep]
        if not len(active):
            break
        co = tuple(v[active] for v in coeff)
        xx, bb = x[active], bx[active]
        m = len(active)
        V = xp.zeros((m, restart+1, dim))
        H = xp.zeros((m, restart+1, restart))
        cs = xp.zeros((m, restart)); sn = xp.zeros_like(cs)
        g = xp.zeros((m, restart+1)); g[:, 0] = beta
        V[:, 0] = r / beta[:, None]
        live = xp.ones(m, dtype=bool)
        used = xp.zeros(m, dtype=xp.int64)
        for j in range(restart):
            if should_stop is not None and should_stop():
                raise SearchStopped(counts)
            live_ids = xp.flatnonzero(live)
            w = xp.zeros((m, dim), dtype=rhs.dtype)
            w[live_ids] = derivative(P, tuple(v[live_ids] for v in co),
                                     xx[live_ids], bb[live_ids], V[live_ids, j])
            # Twice modified Gram-Schmidt improves behavior near dependence.
            for _ in range(2):
                for k in range(j+1):
                    dot = xp.sum(w*V[:, k], axis=1)
                    H[:, k, j] += dot
                    w -= dot[:, None]*V[:, k]
            length = xp.linalg.norm(w, axis=1)
            H[:, j+1, j] = length
            V[:, j+1] = w/xp.where(length > 0, length, 1)[:, None]
            for k in range(j):
                u, v = H[:, k, j].copy(), H[:, k+1, j].copy()
                H[:, k, j] = cs[:, k]*u + sn[:, k]*v
                H[:, k+1, j] = -sn[:, k]*u + cs[:, k]*v
            u, v = H[:, j, j].copy(), H[:, j+1, j].copy()
            rho = xp.hypot(u, v)
            safe = xp.where(rho > 0, rho, 1)
            cs[:, j], sn[:, j] = u/safe, v/safe
            H[:, j, j] = rho
            g[:, j+1] = -sn[:, j]*g[:, j]
            g[:, j] *= cs[:, j]
            used += live
            counts[active] += live
            live &= (xp.abs(g[:, j+1]) > target[active]) & (length > 1e-30) & (rho > 0)
            if not bool(xp.any(live)):
                break
        y = xp.zeros((m, j+1))
        for k in range(j, -1, -1):
            value = g[:, k] - xp.sum(H[:, k, k+1:j+1]*y[:, k+1:], axis=1)
            diag = H[:, k, k]
            y[:, k] = xp.where((used > k) & (diag > 1e-30), value/xp.where(diag > 1e-30, diag, 1), 0)
        answer[active] += xp.sum(V[:, :j+1]*y[:, :, None], axis=1)
        # True linear residual determines whether another restart is needed.
    return answer, counts


def solve_batch(coefficients, initial, backend='cpu', batch_size=32, tol=1e-9,
                maxiter=25, restart=40, cycles=15, line_search=12, products=None, should_stop=None):
    """Solve a same-dimension group, preserving original row order and diagnostics."""
    if min(batch_size, restart, cycles, line_search) < 1 or maxiter < 0 or not np.isfinite(tol) or tol <= 0:
        raise ValueError('invalid solver controls')
    initial = np.asarray(initial, dtype=np.float64)
    if initial.ndim != 2 or initial.shape[1] < 2 or initial.shape[1] & (initial.shape[1]-1):
        raise ValueError('initial must have shape (searches, 2**level)')
    if len(coefficients) != 4 or any(np.shape(v) != initial.shape for v in coefficients):
        raise ValueError('four coefficient arrays must match initial shape')
    P = products or BatchProducts(initial.shape[1], backend)
    xp = P.xp
    results = []
    for begin in range(0, len(initial), batch_size):
        end = min(begin+batch_size, len(initial))
        co = tuple(xp.asarray(v[begin:end], dtype=xp.float64) for v in coefficients)
        x = xp.asarray(initial[begin:end]).copy()
        size = len(x)
        ids = xp.arange(size)
        histories = [[] for _ in range(size)]
        reasons = ['iteration budget']*size
        iterations = np.zeros(size, dtype=int)
        corrections = xp.zeros(size, dtype=xp.int64)
        krylov = xp.zeros(size, dtype=xp.int64)
        backtracks = xp.zeros(size, dtype=xp.int64)
        for iteration in range(maxiter+1):
            if not len(ids):
                break
            if should_stop is not None and should_stop():
                for i in P.host(ids): reasons[i] = 'run stopped'
                break
            active_co = tuple(v[ids] for v in co)
            f, bx = residual(P, active_co, x[ids])
            norms = xp.linalg.norm(f, axis=1)
            # One small diagnostic transfer per Newton iteration, not per product.
            host_ids, host_norms = P.host(ids), P.host(norms)
            for i, norm in zip(host_ids, host_norms):
                histories[i].append(float(norm) if np.isfinite(norm) else None)
                iterations[i] = iteration
                if not np.isfinite(norm): reasons[i] = 'nonfinite residual'
                elif norm <= tol: reasons[i] = 'residual tolerance'
            keep = xp.isfinite(norms) & (norms > tol)
            ids, f, bx, norms = ids[keep], f[keep], bx[keep], norms[keep]
            if not len(ids) or iteration == maxiter:
                break
            active_co = tuple(v[ids] for v in co)
            corrections[ids] += 1
            try:
                step, counts = gmres_batch(P, active_co, x[ids], bx, -f, tol/10, restart, cycles,
                                           should_stop=should_stop)
            except SearchStopped as stopped:
                krylov[ids] += stopped.counts
                for i in P.host(ids): reasons[i] = 'run stopped'
                break
            krylov[ids] += counts
            pending = xp.arange(len(ids))
            accepted = xp.zeros(len(ids), dtype=bool)
            stopped = False
            for power in range(line_search):
                if not len(pending): break
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                row = ids[pending]
                candidate = x[row] + (0.5**power)*step[pending]
                trial, _ = residual(P, tuple(v[row] for v in co), candidate)
                improved = xp.linalg.norm(trial, axis=1) < norms[pending]
                x[row[improved]] = candidate[improved]
                accepted[pending[improved]] = True
                backtracks[row] += 1
                pending = pending[~improved]
            if stopped:
                for i in P.host(ids): reasons[i] = 'run stopped'
                break
            for i in P.host(ids[~accepted]): reasons[i] = 'line search stalled'
            ids = ids[accepted]
        lam, rows = verify(P, co, x)
        hx, hl, hr, hk, hb, hc = map(P.host, (x, lam, rows, krylov, backtracks, corrections))
        for i in range(size):
            verified = bool(np.all(np.isfinite(hr[i])) and np.max(hr[i]) <= 10*tol)
            results.append({'status': 'eigenpair_verified' if verified else 'unresolved',
                            'eigenpair_residual': float(np.max(hr[i])) if np.all(np.isfinite(hr[i])) else None,
                            'row_residuals': [float(v) if np.isfinite(v) else None for v in hr[i]],
                            'solver': {'converged': reasons[i] == 'residual tolerance', 'reason': reasons[i],
                                       'residuals': histories[i], 'iterations': int(iterations[i]),
                                       'newton_steps': int(hc[i]), 'krylov_iterations': int(hk[i]), 'line_search_evaluations': int(hb[i])},
                            '_root': hx[i], '_lambda': hl[i]})
    return results, P.stats()


def starting_points(saved_x, starts, seed):
    """Historical retry start sequence, shared by both implementations."""
    rng = np.random.default_rng(seed)
    scale = max(1.0, float(np.linalg.norm(saved_x)))
    for i in range(starts):
        if i == 0:
            yield -saved_x, 'negative_saved_candidate'
        elif i == 1:
            yield np.zeros_like(saved_x), 'zero'
        else:
            direction = rng.normal(size=len(saved_x))
            direction /= np.linalg.norm(direction)
            radius = scale*(0.25, 1.0, 4.0)[(i-2)//2 % 3]
            yield (saved_x+radius*direction, 'perturbed_saved_candidate') if i % 2 == 0 else (radius*direction, 'random_direction')


def decode(record):
    result = []
    for name in 'abcd':
        v = np.zeros(record['dimension'])
        for index, value in record['coefficients'][name].items():
            v[int(index)] = value
        result.append(v)
    return result


def retry_group(records, saved, backend, starts, seed, resume_saved=False, progress=None, **controls):
    """One start per unresolved matrix per round; stop launching starts on success.

    Groups must have a common dimension. Start RNGs are matrix-local and retain
    the historical retry sequence regardless of batching/compaction order.
    """
    if not records: return [], {}
    coefficients = tuple(np.stack(v) for v in zip(*(decode(r) for r in records)))
    P = BatchProducts(records[0]['dimension'], backend)
    if progress: progress('checking saved candidates')
    best, _ = solve_batch(coefficients, np.stack(saved), backend, products=P,
                          **{**controls, 'maxiter': 0})
    generators = [starting_points(x, starts, seed+r['seed']) for r, x in zip(records, saved)]
    attempts = [[] for _ in records]
    if resume_saved:
        pending = [i for i, r in enumerate(best) if r['status'] != 'eigenpair_verified']
        if pending:
            if progress: progress(f'continuing {len(pending)} saved candidates')
            resumed, _ = solve_batch(tuple(v[pending] for v in coefficients),
                                     np.stack([saved[i] for i in pending]), backend, products=P, **controls)
            for i, r in zip(pending, resumed):
                attempts[i].append({'start': -1, 'kind': 'continued_saved_candidate',
                    'initial_norm': float(np.linalg.norm(saved[i])),
                    'eigenpair_residual': r['eigenpair_residual'], 'solver': r['solver']})
                if r['eigenpair_residual'] is not None and (best[i]['eigenpair_residual'] is None or
                        r['eigenpair_residual'] < best[i]['eigenpair_residual']):
                    best[i] = r
            if progress: progress(f'continuation complete: {sum(r["status"] == "eigenpair_verified" for r in best)}/{len(best)} matrices verified')
    for start in range(starts):
        if controls.get('should_stop') is not None and controls['should_stop'](): break
        pending = [i for i, r in enumerate(best) if r['status'] != 'eigenpair_verified']
        if not pending: break
        if progress: progress(f'fresh start {start+1}/{starts}: {len(pending)} unresolved matrices')
        points = [next(generators[i]) for i in pending]
        results, _ = solve_batch(tuple(v[pending] for v in coefficients),
                                 np.stack([p[0] for p in points]), backend, products=P, **controls)
        for i, r, (initial, label) in zip(pending, results, points):
            attempts[i].append({'start': start, 'kind': label, 'initial_norm': float(np.linalg.norm(initial)),
                                'eigenpair_residual': r['eigenpair_residual'], 'solver': r['solver']})
            if r['eigenpair_residual'] is not None and (best[i]['eigenpair_residual'] is None or
                    r['eigenpair_residual'] < best[i]['eigenpair_residual']):
                best[i] = r
    for r, original, history in zip(best, records, attempts):
        r.update(level=original['level'], dimension=original['dimension'], seed=original['seed'],
                 matrix_id=original.get('matrix_id', f"level-{original['level']}-seed-{original['seed']}"),
                 restart_seed=seed+original['seed'], coefficients=original['coefficients'], attempts=history)
    return best, P.stats()


def add_controls(parser, default_solver=None):
    parser.add_argument('--solver', choices=['individual', 'batched'], default=default_solver,
                        help='Eigen searches default to batched; individual retains the historical reference')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gmres-restart', type=int, default=40)
    parser.add_argument('--gmres-cycles', type=int, default=15)
    parser.add_argument('--line-search', type=int, default=12)


def controls(args):
    return dict(batch_size=args.batch_size, tol=args.tol, maxiter=args.maxiter,
                restart=args.gmres_restart, cycles=args.gmres_cycles, line_search=args.line_search)
