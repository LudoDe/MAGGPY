import os
import sys
import emcee
import pickle
import signal
import warnings
import platform
import numpy as np

from pathlib import Path
from typing import Callable, Optional, List
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool


def get_platform_specific_executor(workers: int = None):
    """Returns the appropriate executor configuration based on the platform."""
    system = platform.system().lower()
    n_processes = max(1, os.cpu_count() - 1)
    if workers is not None:
        n_processes = min(n_processes, workers)

    if system == 'darwin':  # macOS requires spawn
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        return {'max_workers': n_processes, 'mp_context': ctx}
    elif system == 'windows':
        return {'max_workers': n_processes}
    else:  # Linux - use fork (works with notebook-defined functions)
        return {'max_workers': n_processes}
    
def _run_serial(
        log_probability: Callable,
        initial_pos: np.ndarray,
        effective_steps: int,
        backend: emcee.backends.Backend,
        blobs_dtype: List,
        moves: list,
        progress: bool,
        chunk_size: int):
    """Run MCMC on a single core (no multiprocessing)."""
    nwalkers, ndim = initial_pos.shape
    sampler = None
    interrupted = False

    original_sigint = signal.getsignal(signal.SIGINT)

    def graceful_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n⚠ Interrupt received. Finishing current chunk...")

    try:
        signal.signal(signal.SIGINT, graceful_handler)

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            backend=backend,
            moves=moves,
            blobs_dtype=blobs_dtype,
        )

        remaining = effective_steps
        current_pos = initial_pos

        while remaining > 0 and not interrupted:
            steps_this_chunk = min(chunk_size, remaining)
            try:
                state = sampler.run_mcmc(
                    current_pos,
                    nsteps=steps_this_chunk,
                    progress=progress,
                )
                current_pos = None
            except KeyboardInterrupt:
                interrupted = True
                break
            remaining -= steps_this_chunk

        if interrupted:
            iteration = backend.iteration if backend else "N/A"
            print(f"✓ Stopped gracefully at iteration {iteration}")
        else:
            acceptance = np.mean(sampler.acceptance_fraction)
            print(f"✓ MCMC complete (serial). Mean acceptance fraction: {acceptance:.3f}")

        return sampler

    except Exception as e:
        warnings.warn(f"MCMC error: {str(e)}")
        raise

    finally:
        signal.signal(signal.SIGINT, original_sigint)


def _run_parallel(
        log_probability: Callable,
        initial_pos: np.ndarray,
        effective_steps: int,
        backend: emcee.backends.Backend,
        workers: int,
        blobs_dtype: List,
        moves: list,
        progress: bool,
        chunk_size: int):
    """Run MCMC with ProcessPoolExecutor."""
    nwalkers, ndim = initial_pos.shape
    executor_kwargs = get_platform_specific_executor(workers=workers)
    sampler = None
    interrupted = False

    original_sigint = signal.getsignal(signal.SIGINT)

    def graceful_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n⚠ Interrupt received. Finishing current chunk...")

    try:
        signal.signal(signal.SIGINT, graceful_handler)

        with ProcessPoolExecutor(**executor_kwargs) as executor:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                log_probability,
                pool=executor,
                backend=backend,
                moves=moves,
                blobs_dtype=blobs_dtype,
            )

            remaining = effective_steps
            current_pos = initial_pos

            while remaining > 0 and not interrupted:
                steps_this_chunk = min(chunk_size, remaining)
                try:
                    state = sampler.run_mcmc(
                        current_pos,
                        nsteps=steps_this_chunk,
                        progress=progress,
                    )
                    current_pos = None
                except KeyboardInterrupt:
                    interrupted = True
                    break
                remaining -= steps_this_chunk

        if interrupted:
            iteration = backend.iteration if backend else "N/A"
            print(f"✓ Stopped gracefully at iteration {iteration}")
        else:
            acceptance = np.mean(sampler.acceptance_fraction)
            print(f"✓ MCMC complete. Mean acceptance fraction: {acceptance:.3f}")

        return sampler

    except BrokenProcessPool as e:
        print(f"\n⚠ Process pool error: {e}")
        if backend:
            print(f"  State saved at iteration {backend.iteration}")
        return sampler

    except Exception as e:
        warnings.warn(f"MCMC error: {str(e)}")
        raise

    finally:
        signal.signal(signal.SIGINT, original_sigint)


def run_mcmc_parallel(
        log_probability: Callable,
        initial_pos: np.ndarray,
        max_n: int,
        backend: emcee.backends.Backend = None,
        workers: int = None,
        parallel: bool = True,
        blobs_dtype: List = None,
        moves: list = None,
        progress: bool = True,
        chunk_size: int = 100):
    """
    Run MCMC sampling with optional multiprocessing and automatic fallback.

    When ``parallel=True`` (default), uses a ProcessPoolExecutor for
    multi-core sampling.  If the pool fails — typically because the
    log-probability function cannot be pickled (common on macOS with
    ``spawn`` context) — the sampler automatically retries on a single
    core so the notebook can still run.

    Set ``parallel=False`` to skip multiprocessing entirely (recommended
    if you encounter pickling errors on macOS).

    Parameters
    ----------
    log_probability : callable
        Log probability function to sample. Can return (logp, *blobs).
    initial_pos : array
        Initial positions for walkers (shape: (nwalkers, ndim))
    max_n : int
        Maximum number of iterations
    backend : emcee.Backend, optional
        Backend for storing chain.
    workers : int, optional
        Number of workers. If None, uses cpu_count - 1.
    parallel : bool
        If True (default), use multiprocessing. If False, run on a
        single core. When True and multiprocessing fails, automatically
        falls back to serial execution.
    blobs_dtype : list, optional
        Dtype for blobs storage. If None, uses MAGGPY default (5 floats).
    moves : list, optional
        emcee move strategy.
    progress : bool
        Show progress bar.
    chunk_size : int
        Steps per chunk for responsive interrupts (default: 100).

    Returns
    -------
    sampler : emcee.EnsembleSampler or None
    """
    # Determine effective number of steps
    effective_steps = max_n
    if backend is not None and Path(backend.filename).exists():
        try:
            if backend.iteration > 0:
                effective_steps = max_n - backend.iteration
                print(f"Resuming from iteration {backend.iteration}/{max_n}")
        except Exception:
            pass  # Backend might be empty

    if effective_steps <= 0:
        print("MCMC sampling already completed.")
        return None

    if moves is None:
        moves = [
            (emcee.moves.StretchMove(), 0.60),
            (emcee.moves.DESnookerMove(), 0.40),
        ]

    # Default blobs for MAGGPY (5 values after logp)
    if blobs_dtype is None:
        blobs_dtype = [
            ("rate", float),
            ("l_epeak", float),
            ("l_t90", float),
            ("l_pflux", float),
            ("l_fluence", float),
        ]

    common_kwargs = dict(
        log_probability=log_probability,
        initial_pos=initial_pos,
        effective_steps=effective_steps,
        backend=backend,
        blobs_dtype=blobs_dtype,
        moves=moves,
        progress=progress,
        chunk_size=chunk_size,
    )

    if not parallel:
        print(f"Running MCMC on a single core ({effective_steps} steps)...")
        return _run_serial(**common_kwargs)

    # --- Parallel mode with automatic fallback ---
    try:
        n_workers = get_platform_specific_executor(workers=workers)['max_workers']
        print(f"Running MCMC with {n_workers} workers ({effective_steps} steps)...")
        return _run_parallel(workers=workers, **common_kwargs)

    except (BrokenProcessPool, RuntimeError, pickle.PicklingError, TypeError) as e:
        print(f"\n⚠ Multiprocessing failed: {e}")
        print("  Falling back to single-core execution...\n")
        # Reset backend if it was partially written
        return _run_serial(**common_kwargs)