"""

Author: Camila Losada
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal


def moving_average(data: np.ndarray, win: int, step: int = 1) -> np.ndarray:
    """Computes a centered moving average over the last axis of an N-dimensional array.

    This implementation preserves edges by treating boundaries as "shrinking windows"
    (effectively ignoring padding) and ignores NaNs in the input data.

    Args:
        data (np.ndarray): Input array of any dimension.
        win (int): The size of the moving window.
        step (int, optional): The step size for the moving average.
            Defaults to 1.

    Returns:
        np.ndarray: The smoothed array. The shape is the same as the input, though the
            last dimension size will vary based on the step size.
    """
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(float)

    # Calculate asymmetric padding to ensure output size matches input size.
    pad_left = win // 2
    pad_right = win - 1 - pad_left

    pads = [(0, 0)] * (data.ndim - 1) + [(pad_left, pad_right)]
    data_padded = np.pad(data, pad_width=pads, mode="constant", constant_values=np.nan)

    windows = sliding_window_view(data_padded, window_shape=win, axis=-1)
    result = np.nanmean(windows, axis=-1)

    return result[..., ::step]


def define_kernel(w_size: float, w_std: float, fs: int):
    kernel = signal.windows.gaussian(M=w_size * fs, std=w_std * fs)
    kernel = kernel / sum(kernel)  # area of the kernel must be one
    return kernel


def convolve_signal(
    arr: np.ndarray,
    fs: int = 1000,
    w_size: float = 0.1,
    w_std: float = 0.015,
    axis: int = 1,
):
    # define kernel for convolution
    kernel = define_kernel(w_size, w_std, fs=fs)
    conv = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"), axis=axis, arr=arr
    )
    return conv * fs
