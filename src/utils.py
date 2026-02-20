import time
import numpy as np
from functools  import wraps
from itertools  import zip_longest
from typing     import Any, Callable, Tuple
from scipy      import integrate, interpolate

def timer_decorator(func: Callable) -> Callable:
    """
    Decorator to time function execution.

    Returns:
        A tuple containing the original function's result and the elapsed time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[Any, float]:
        start_time  = time.perf_counter()
        result      = func(*args, **kwargs)
        end_time    = time.perf_counter()
        return result, end_time - start_time
    return wrapper

def running_avg(arr: np.ndarray, window_size: int = 1600) -> np.ndarray:
    """
    Calculate the running average of an array using a sliding window.

    Parameters:
        arr (np.ndarray): Input array.
        window_size (int): Size of the averaging window.

    Returns:
        np.ndarray: Array of running averages computed only over non-NaN elements.
    """
    valid_values = arr[~np.isnan(arr)]
    avg = np.convolve(valid_values, np.ones(window_size) / window_size, mode='valid')
    return avg