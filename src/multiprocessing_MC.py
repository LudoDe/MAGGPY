import os
import emcee
import pickle
import platform
import numpy as np

from pathlib import Path
from typing import Callable, List
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool


def _get_executor_kwargs(workers: int = None):
    """Platform-aware kwargs for ProcessPoolExecutor."""
    n_cpu = max(1, os.cpu_count() - 1)
    if workers is not None:
        n_cpu = min(n_cpu, workers)

    kwargs = {'max_workers': n_cpu}
    if platform.system().lower() == 'darwin':
        import multiprocessing as mp
        kwargs['mp_context'] = mp.get_context('spawn')
    return kwargs


def run_mcmc_parallel(
        log_probability: Callable,
        initial_pos: np.ndarray,
        max_n: int,
        backend: emcee.backends.HDFBackend = None,
        workers: int = None,
        parallel: bool = True,
        blobs_dtype: List = None,
        moves: list = None,
        progress: bool = True):
    """
    Run MCMC sampling with optional multiprocessing and automatic fallback.

    Automatically resumes from the backend if it already contains iterations
    (walker positions are loaded from the backend, not from ``initial_pos``).

    Parameters
    ----------
    log_probability : callable
        Log probability function.  Can return ``(logp, *blobs)``.
    initial_pos : array, shape (nwalkers, ndim)
        Initial walker positions.  Ignored when resuming from a backend.
    max_n : int
        Total number of iterations (including any already completed).
    backend : emcee.backends.HDFBackend, optional
        HDF5 backend for incremental saves.
    workers : int, optional
        Number of parallel workers.  ``None`` → ``cpu_count - 1``.
    parallel : bool
        Use multiprocessing.  Falls back to serial on failure.
    blobs_dtype : list, optional
        Dtype spec for blobs.  Defaults to MAGGPY's 5-float layout.
    moves : list, optional
        emcee move strategy.  Defaults to 60 % Stretch + 40 % DESnooker.
    progress : bool
        Show tqdm progress bar.

    Returns
    -------
    sampler : emcee.EnsembleSampler or None
    """
    # ── Resume logic ─────────────────────────────────────────────────────────
    start_pos = initial_pos
    effective_steps = max_n

    if backend is not None and Path(backend.filename).exists():
        try:
            already_done = backend.iteration
            if already_done > 0:
                effective_steps = max_n - already_done
                start_pos = None        # emcee reads last state from backend
                print(f"Resuming from iteration {already_done}/{max_n}")
        except Exception:
            pass

    if effective_steps <= 0:
        print("MCMC sampling already completed.")
        return None

    # ── Defaults ─────────────────────────────────────────────────────────────
    if moves is None:
        moves = [
            (emcee.moves.StretchMove(),    0.60),
            (emcee.moves.DESnookerMove(),  0.40),
        ]
    if blobs_dtype is None:
        blobs_dtype = [
            ("rate",      float),
            ("l_epeak",   float),
            ("l_t90",     float),
            ("l_pflux",   float),
            ("l_fluence", float),
        ]

    nwalkers, ndim = initial_pos.shape

    # ── Helper: build sampler & run ──────────────────────────────────────────
    def _run(pool=None):
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            pool=pool, backend=backend,
            moves=moves, blobs_dtype=blobs_dtype,
        )
        sampler.run_mcmc(start_pos, nsteps=effective_steps, progress=progress)
        acc = np.mean(sampler.acceptance_fraction)
        mode = "parallel" if pool is not None else "serial"
        print(f"✓ MCMC complete ({mode}). Mean acceptance: {acc:.3f}")
        return sampler

    # ── Serial path ──────────────────────────────────────────────────────────
    if not parallel:
        print(f"Running MCMC on a single core ({effective_steps} steps)...")
        return _run()

    # ── Parallel path with automatic fallback ────────────────────────────────
    try:
        executor_kwargs = _get_executor_kwargs(workers)
        n_workers = executor_kwargs['max_workers']
        print(f"Running MCMC with {n_workers} workers ({effective_steps} steps)...")
        with ProcessPoolExecutor(**executor_kwargs) as pool:
            return _run(pool)

    except (BrokenProcessPool, RuntimeError, pickle.PicklingError, TypeError) as e:
        print(f"\n⚠ Multiprocessing failed: {e}")
        print("  Falling back to single-core execution...\n")
        return _run()