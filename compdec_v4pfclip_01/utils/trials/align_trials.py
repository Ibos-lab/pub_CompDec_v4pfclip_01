"""

Author: Camila Losada
"""

import numpy as np


def indep_roll(arr: np.ndarray, shifts: np.ndarray, axis: int = 1) -> np.ndarray:
    """Apply an independent roll for each dimensions of a single axis.
    Args:
        arr (np.ndarray): Array of any shape.
        shifts (np.ndarray): How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.
        axis (int, optional): Axis along which elements are shifted. Defaults to 1.

    Returns:
        np.ndarray: shifted array.
    """
    arr = np.swapaxes(arr, axis, -1)  # Move the target axis to the last position
    all_idcs = np.ogrid[[slice(0, n) for n in arr.shape]]  # Create grid indices
    shifts[shifts < 0] += arr.shape[-1]  # Convert to a positive shift
    new_indices = all_idcs[-1] - shifts[:, np.newaxis]
    result = arr[tuple(all_idcs[:-1]) + (new_indices,)]
    arr = np.swapaxes(result, -1, axis)
    return arr


def fill_from_idx(arr, indices, fill_value, direction="after"):
    """
    Fill elements in an n-dimensional NumPy array along the last axis either before or after given indices.

    Args:
        arr (np.ndarray): Input n-dimensional array to be modified (ndim >= 1).
        indices (np.ndarray): Array of starting indices for the last axis, with shape arr.shape[:-1].
                             Each index must be valid (0 <= indices[...] < arr.shape[-1]).
        fill_value (scalar): Value to fill in the specified positions (e.g., np.nan, 0).
        direction (str): Direction to fill, either 'after' (default) or 'before'.

    Returns:
        np.ndarray: The modified array with specified positions filled.

    Raises:
        ValueError: If indices shape mismatches arr.shape[:-1], indices are invalid, or direction is invalid.
        TypeError: If arr is not a NumPy array or indices is not array-like.
    """
    # Input validation
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a NumPy array")
    if arr.ndim < 1:
        raise ValueError("arr must have at least 1 dimension")

    indices = np.asarray(indices)
    if indices.shape != arr.shape[:-1]:
        raise ValueError(
            f"indices shape {indices.shape} must match arr.shape[:-1] {arr.shape[:-1]}"
        )
    if not np.all((0 <= indices) & (indices <= arr.shape[-1])):
        raise ValueError(f"indices must be between 0 and {arr.shape[-1]-1}")

    if direction not in ["after", "before"]:
        raise ValueError("direction must be 'after' or 'before'")

    # Convert array to compatible dtype if necessary
    if np.isnan(fill_value) and not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(float)

    # Create index arrays for all dimensions
    idx = [np.arange(s) for s in arr.shape[:-1]]
    last_dim_indices = []

    for idx_tuple in np.ndindex(arr.shape[:-1]):
        start = indices[idx_tuple]
        if direction == "after":
            range_indices = np.arange(start, arr.shape[-1])
        else:  # direction == 'before'
            range_indices = np.arange(0, start)
        last_dim_indices.append(range_indices)

    last_dim_indices = np.concatenate(last_dim_indices)

    # Create indices for the first n-1 dimensions, repeated for each last dimension index
    idx_grids = np.meshgrid(*idx, indexing="ij")
    idx_grids = [g.ravel() for g in idx_grids]

    # Calculate lengths for repetition
    if direction == "after":
        lengths = arr.shape[-1] - indices.ravel()
    else:
        lengths = indices.ravel()

    idx_grids = [np.repeat(g, lengths) for g in idx_grids]

    full_idx = idx_grids + [last_dim_indices]
    arr[tuple(full_idx)] = fill_value

    return arr
