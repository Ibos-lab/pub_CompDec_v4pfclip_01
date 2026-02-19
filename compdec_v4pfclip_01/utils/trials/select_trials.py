"""

Author: Camila Losada
"""

import numpy as np
from typing import Dict, List, Tuple
from compdec_v4pfclip_01.utils.config_task import EVENTS_B1


def get_sp_by_sample(
    sp: np.ndarray, sample_id: np.ndarray, samples: List = [11, 15, 51, 55, 0]
) -> Tuple[Dict, Dict]:
    """Group spike data by sample IDs and return corresponding data and indices.

    Args:
        sp (np.ndarray): Spike data array, where each row corresponds to a trial.
        sample_id (np.ndarray): Array of sample IDs for each trial, same length as sp.
        samples (List, optional): List of sample IDs to process. Defaults to [11, 15, 51, 55, 0].

    Returns:
        Tuple[Dict, Dict]: Two dictionaries:
            - sp_samples: Maps sample IDs (as strings) to arrays of spike data.
            - idx_samples: Maps sample IDs (as strings) to arrays of trial indices.
            For sample IDs with no trials, returns np.array([np.nan]).
    """
    sp_samples = {}
    idx_samples = {}
    for s_id in samples:
        samp_idx = np.where(sample_id == s_id)[0]
        if samp_idx.shape[0] > 0:  # Check number of trials
            sp_samples[str(s_id)] = sp[samp_idx]
            idx_samples[str(s_id)] = samp_idx
        else:
            sp_samples[str(s_id)] = np.array([np.nan])
            idx_samples[str(s_id)] = np.array([np.nan])
    return sp_samples, idx_samples


def remove_missing_trials(
    neu,
    ntr=3,
    code_end=EVENTS_B1["end_trial"],
    offset=1500,
    start_idx=1000,
    ret_mask=False,
    update=True,
):
    """
    Remove trials in NeuronData object based on concatenated spike samples.

    Trials with zero spikes in (ntr) concatenated segments are marked invalid.
    Updates the neu object's attributes with filtered data.

    Args:
        neu (object): Neural data object with attributes: sp_samples, code_numbers, code_samples, etc.
        ntr (int): Number of trials to concatenate (default: 3).
        code_end (int): Code number indicating trial end (default: EVENTS_B1["end_trial"]).
        offset (int): Offset to add to code_samples for end index (default: 1500).
        start_idx (int): Start index for spike samples (default: 1000).

    Returns:
        None: Updates neu object attributes in-place via neu.edit_attributes.

    Raises:
        ValueError: If idx_end length mismatches total trials.
    """
    # Input data
    sp = neu.sp_samples
    total_tr = sp.shape[0]
    signal = np.full(total_tr, True)
    # Find end indices
    idx_code_end = np.where(neu.code_numbers == code_end)
    idx_end = (neu.code_samples[idx_code_end]).astype(int) + offset
    # Validate idx_end length
    if len(idx_end) < total_tr:
        raise ValueError(
            f"idx_end length ({len(idx_end)}) does not match total trials ({total_tr})"
        )
    # Filter trials
    i = 0
    while i <= total_tr - ntr:
        try:
            tr_concat = np.concatenate(
                [sp[i + k, start_idx : idx_end[i + k]] for k in range(ntr)]
            )
            total_sp = np.sum(tr_concat)
            if total_sp == 0:
                signal[i : i + ntr] = False
                i += ntr
            else:
                i += 1
        except IndexError as e:
            print(f"Index error at trial {i}: {e}")
            break
    # Update attributes
    if update:
        new_values = {}
        attr = [
            "sp_samples",
            "block",
            "trial_error",
            "code_samples",
            "code_numbers",
            "position",
            "pos_code",
            "sample_id",
            "test_stimuli",
            "test_distractor",
        ]
        for iattr in attr:
            if hasattr(neu, iattr):
                new_values[iattr] = getattr(neu, iattr)[signal]
            else:
                print(f"Warning: Attribute {iattr} not found in neu")
        # Apply updates
        neu.edit_attributes(new_values)
    if ret_mask:
        return signal
