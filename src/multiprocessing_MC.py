import os
import emcee
import pickle
import signal
import warnings
import platform
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool


def get_platform_specific_executor(workers: int = None):
    """Returns executor kwargs tuned for the current platform."""
    system = platform.system().lower()
    n_processes = max(1, os.cpu_count() - 1)
    if workers is not None:
        n_processes = min(n_processes, workers)

    if system == "darwin":
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        return {"max_workers": n_processes, "mp_context": ctx}
    else:
        return {"max_workers": n_processes}


@contextmanager
def _interrupt_context():
    """Context manager that sets a SIGINT handler and yields a mutable flag dict."""
    original = signal.getsignal(signal.SIGINT)
    interrupted = {"flag": False}

    def _handler(signum, frame):
        interrupted["flag"] = True
        print("\n⚠ Interrupt received. Finishing current chunk...")

    signal.signal(signal.SIGINT, _handler)
    try:
        yield interrupted
    finally:
        signal.signal(signal.SIGINT, original)


def _run_loop(sampler, initial_pos, effective_steps, progress, chunk_size, backend, interrupted):
    """Run the sampler in chunks, honoring interrupts."""
    remaining = effective_steps
    current_pos = initial_pos

    while remaining > 0 and not interrupted["flag"]:
        steps_this_chunk = min(chunk_size, remaining)
        try:
            sampler.run_mcmc(current_pos, nsteps=steps_this_chunk, progress=progress)
            current_pos = None
        except KeyboardInterrupt:
            interrupted["flag"] = True
            break
        remaining -= steps_this_chunk

    if interrupted["flag"]:
        iteration = backend.iteration if backend else "N/A"
        print(f"✓ Stopped gracefully at iteration {iteration}")
    else:
        acceptance = np.mean(sampler.acceptance_fraction)
        print(f"✓ MCMC complete. Mean acceptance fraction: {acceptance:.3f}")

    return sampler


def _run_serial(
    log_probability: Callable,
    initial_pos: np.ndarray,
    effective_steps: int,
    backend: emcee.backends.Backend,
    blobs_dtype: List,
    moves: list,
    progress: bool,
    chunk_size: int,
):
    """Run MCMC on a single core."""
    nwalkers, ndim = initial_pos.shape

    try:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            backend=backend,
            moves=moves,
            blobs_dtype=blobs_dtype,
        )

        with _interrupt_context() as interrupted:
            return _run_loop(
                sampler,
                initial_pos,
                effective_steps,
                progress,
                chunk_size,
                backend,
                interrupted,
            )

    except Exception as e:
        warnings.warn(f"MCMC error: {str(e)}")
        raise


def _run_parallel(
    log_probability: Callable,
    initial_pos: np.ndarray,
    effective_steps: int,
    backend: emcee.backends.Backend,
    workers: int,
    blobs_dtype: List,
    moves: list,
    progress: bool,
    chunk_size: int,
):
    """Run MCMC using a ProcessPoolExecutor."""
    nwalkers, ndim = initial_pos.shape
    executor_kwargs = get_platform_specific_executor(workers=workers)

    try:
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

            with _interrupt_context() as interrupted:
                return _run_loop(
                    sampler,
                    initial_pos,
                    effective_steps,
                    progress,
                    chunk_size,
                    backend,
                    interrupted,
                )

    except BrokenProcessPool as e:
        print(f"\n⚠ Process pool error: {e}")
        if backend:
            print(f"  State saved at iteration {backend.iteration}")
        return None

    except Exception as e:
        warnings.warn(f"MCMC error: {str(e)}")
        raise


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
    chunk_size: int = 100,
):
    """Run MCMC sampling with optional multiprocessing and automatic fallback."""
    effective_steps = max_n
    if backend is not None and Path(backend.filename).exists():
        try:
            if backend.iteration > 0:
                effective_steps = max_n - backend.iteration
                print(f"Resuming from iteration {backend.iteration}/{max_n}")
        except Exception:
            pass

    if effective_steps <= 0:
        print("MCMC sampling already completed.")
        return None

    if moves is None:
        moves = [(emcee.moves.StretchMove(), 0.60), (emcee.moves.DESnookerMove(), 0.40)]

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

    try:
        n_workers = get_platform_specific_executor(workers=workers)["max_workers"]
        print(f"Running MCMC with {n_workers} workers ({effective_steps} steps)...")
        sampler = _run_parallel(workers=workers, **common_kwargs)
        if sampler is None:
            print("  Falling back to single-core execution...\n")
            return _run_serial(**common_kwargs)
        return sampler

    except (BrokenProcessPool, RuntimeError, pickle.PicklingError, TypeError) as e:
        print(f"\n⚠ Multiprocessing failed: {e}")
        print("  Falling back to single-core execution...\n")
        return _run_serial(**common_kwargs)
