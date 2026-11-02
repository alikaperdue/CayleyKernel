# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Multi-GPU batch splitting for CayleyKernel.

This module provides :func:`visible_devices` and :func:`split_batch` for
distributing a large batch of Cayley-Dickson products across multiple
visible CUDA devices.

Design notes
------------
* Only CuPy's ``cp.cuda.Device`` context manager and ``cp.cuda.Stream``
  are used; no external dependency (torch.distributed, mpi4py, …) is
  required.
* The leading batch axis (``axis=0``) is sliced into ``len(devices)``
  contiguous, roughly-equal chunks with ``np.array_split``.
* Each chunk is copied to its target device, the kernel is launched, and
  the CPU moves on to the next device **without** synchronising.  This
  lets the GPU work on different devices overlap.
* :func:`gather_into` synchronises each device and copies its result
  chunk back onto the caller's device, preserving original batch order.

See ``docs/multi-gpu.md`` for the full usage guide, memory-budget
guidance, and a worked example.
"""

import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

from . import (  # noqa: E402  (imported after the guarded cupy import)
    _as_cupy,
    _broadcast_shapes,
    _launch,
    _result_dtype,
    _select_kernel,
    is_power_of_two,
    loaded,
)

__all__ = ["visible_devices", "split_batch"]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def visible_devices():
    """Return a list of visible CUDA device IDs.

    Wraps ``cp.cuda.runtime.getDeviceCount()``.  Returns an empty list
    when CuPy or the CUDA runtime is unavailable.

    Returns
    -------
    list[int]
        ``[0, 1, …, N-1]`` for N visible devices, or ``[]``.
    """
    if cp is None:
        return []
    try:
        n = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        return []
    return list(range(n))


def split_batch(a, b, devices=None, axis=0, out=None, stream=None):
    """Distribute a batch of Cayley-Dickson products across N CUDA devices.

    The leading batch axis (``axis=0``) is split into ``len(devices)``
    contiguous, roughly-equal chunks.  Each chunk is launched on its own
    device via a ``cp.cuda.Device`` context manager and a dedicated
    ``cp.cuda.Stream`` so that per-device work can overlap.  All devices
    are synchronised before the result is gathered back onto the caller's
    current device, preserving the original batch order.

    Parameters
    ----------
    a, b : array_like
        Operands.  ``a.shape[-1]`` and ``b.shape[-1]`` must be equal and
        a power of two.  Leading axes are batch axes; only ``axis=0`` is
        split.
    devices : list[int] or None
        CUDA device IDs to use.  ``None`` means all visible devices.
    axis : int, default 0
        The batch axis to split.  Only ``0`` is supported.
    out : cupy.ndarray or None
        Pre-allocated output buffer **on the caller's device**.  Must have
        a shape equal to the broadcast result shape and a dtype equal to
        the promoted dtype of ``a`` and ``b``.
    stream : cupy.cuda.Stream or int or None
        Accepted for API symmetry with ``multiply``; ignored.  Per-device
        streams are created internally.

    Returns
    -------
    cupy.ndarray
        The reassembled result on the caller's current device.

    Raises
    ------
    ImportError
        If CuPy is not installed.
    RuntimeError
        If the CUDA kernel library has not been loaded.
    ValueError
        If ``axis`` is not 0, if batch sizes are incompatible, or if the
        algebra dimension is not a power of two.
    TypeError
        If ``devices`` is not a list/tuple of ints, or if ``out`` is not
        a CuPy array.
    """
    if cp is None:
        raise ImportError("split_batch requires CuPy; install with: pip install cupy-cuda12x")
    if not loaded():
        raise RuntimeError("split_batch requires the CUDA kernel; build it first: make kernel")

    # --- resolve devices ----------------------------------------------------
    if devices is None:
        devices = visible_devices()
    if not isinstance(devices, (list, tuple)):
        raise TypeError(
            "devices must be a list/tuple of ints, or None (got %s)" % type(devices).__name__
        )
    if len(devices) < 1:
        raise ValueError("devices must contain at least one device ID")

    # --- validate axis ------------------------------------------------------
    if axis != 0:
        raise ValueError("Only axis=0 is supported in split_batch (got axis=%d)" % axis)

    # --- normalise inputs to CuPy on the caller's device --------------------
    # Note: any arrays the caller passes (a, b, out) are on the caller's
    # current device; ``out.device.id`` is the destination device we use
    # in gather_into() to copy each per-device chunk back.
    a = _as_cupy(a)
    b = _as_cupy(b, ref_dtype=a.dtype)

    if a.ndim < 1 or b.ndim < 1:
        raise ValueError(
            "Inputs must have at least one axis; the last axis is the "
            "Cayley-Dickson algebra dimension (2^n)."
        )

    # --- unify dtype ---------------------------------------------------------
    out_dtype = _result_dtype(a.dtype, b.dtype)
    if a.dtype != out_dtype:
        a = a.astype(out_dtype)
    if b.dtype != out_dtype:
        b = b.astype(out_dtype)

    kernel_pair = _select_kernel(out_dtype)

    # --- validate algebra dimension ------------------------------------------
    dim = a.shape[-1]
    if b.shape[-1] != dim:
        raise ValueError(
            "Algebra dimension mismatch: a.shape[-1] (%d) != b.shape[-1] (%d)" % (dim, b.shape[-1])
        )
    if not is_power_of_two(dim):
        raise ValueError("Algebra dimension (last axis = %d) must be a power of two (2^n)." % dim)

    # --- broadcast batch axes ------------------------------------------------
    batch_shape = _broadcast_shapes([a.shape[:-1], b.shape[:-1]])
    out_shape = tuple(batch_shape) + (dim,)
    if a.shape != out_shape:
        a = cp.broadcast_to(a, out_shape)
    if b.shape != out_shape:
        b = cp.broadcast_to(b, out_shape)
    a = cp.ascontiguousarray(a)
    b = cp.ascontiguousarray(b)

    batch_total = int(a.shape[0])
    if batch_total == 0:
        return cp.zeros(out_shape, dtype=out_dtype)

    # --- allocate caller-side output ----------------------------------------
    if out is not None:
        if not isinstance(out, cp.ndarray):
            raise TypeError("out must be a cupy.ndarray when provided")
        if out.dtype != out_dtype:
            raise TypeError(
                "out.dtype (%s) must equal the promoted dtype (%s). "
                "The CUDA kernel is dtype-homogeneous." % (out.dtype, out_dtype)
            )
        if tuple(out.shape) != out_shape:
            raise ValueError(
                "out.shape %s does not match the broadcast result shape %s."
                % (tuple(out.shape), out_shape)
            )
        o = cp.ascontiguousarray(out)
    else:
        o = cp.zeros(out_shape, dtype=out_dtype)

    # --- split the batch axis into contiguous chunks -------------------------
    n_dev = len(devices)
    index_chunks = np.array_split(np.arange(batch_total), n_dev)

    # --- Phase 1: queue work on each device (CPU does not block) --------------
    # Each entry: (start, end, dev_id, stream, c_dev)
    launches = []

    for i, dev in enumerate(devices):
        idx = index_chunks[i]
        if len(idx) == 0:
            continue
        start = int(idx[0])
        end = int(idx[-1]) + 1
        chunk_size = end - start

        with cp.cuda.Device(dev):
            s = cp.cuda.Stream(non_blocking=True)
            with s:
                # Copy the relevant chunk to this device.  .copy() copies
                # to the currently active device (i.e. `dev`), performing a
                # D2D or H2D transfer as needed.
                a_dev = a[start:end].copy()
                b_dev = b[start:end].copy()
                c_dev = cp.empty((chunk_size, dim), dtype=out_dtype)
                # _launch synchronises the current stream (s) before
                # calling the kernel, ensuring the copies above are complete.
                # The kernel itself is enqueued on the device's default
                # stream and runs asynchronously from the CPU's point of
                # view.
                _launch(kernel_pair, a_dev, b_dev, c_dev, dim, chunk_size)
            # Deliberately NOT calling s.synchronize() here: the CPU
            # proceeds immediately to the next device so that GPU work on
            # different devices can overlap.

            launches.append((start, end, dev, s, c_dev))

    # --- Phase 2: synchronise all devices, then gather into the caller's array
    gather_into(o, launches)

    return o


# ---------------------------------------------------------------------------
# internal helper
# ---------------------------------------------------------------------------


def gather_into(out, launches):
    """Synchronise each device in *launches* and copy its result into *out*.

    Parameters
    ----------
    out : cupy.ndarray
        Destination array on the caller's device.  Must be large enough to
        hold all chunks.
    launches : list of tuples
        Each tuple is ``(start, end, dev_id, stream, c_dev)`` where
        ``c_dev`` is the per-device result array (still on ``dev_id``)
        and ``out[start:end]`` is the corresponding slice to fill.

    This helper is the "synchronise only at the end" step: it calls
    ``cp.cuda.runtime.deviceSynchronize()`` for every device before
    reading its result, ensuring all GPU work is complete.
    """
    out_dev = out.device.id
    for start, end, dev_id, _stream, c_dev in launches:
        # Wait for all work on this device to finish before reading c_dev.
        with cp.cuda.Device(dev_id):
            cp.cuda.runtime.deviceSynchronize()
        # Copy the chunk into the caller-side output array.  CuPy performs
        # the D2D (or H2D) transfer automatically.
        with cp.cuda.Device(out_dev):
            out[start:end] = c_dev
