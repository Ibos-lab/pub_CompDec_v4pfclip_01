"""

Author: Camila Losada
"""

import h5py
import numpy as np
from pathlib import Path
import logging


class LfpData:
    def __init__(
        self,
        date_time: str,
        subject: str,
        area: str,
        experiment: str,
        recording: str,
        # -----signal------
        lfps: np.ndarray,
        ch_row: np.ndarray,
        ch_col: np.ndarray,
        # -------bhv-------
        block: np.ndarray,
        trial_error: np.ndarray,
        code_samples: np.ndarray,
        code_numbers: np.ndarray,
        position: np.ndarray,
        pos_code: np.ndarray,
        sample_id: np.ndarray,
        test_stimuli: np.ndarray,
        test_distractor: np.ndarray,
        # ------extra------
        filt: bool,
        f_lp: int,
        f_hp: int,
    ):
        """Initialize the class.

        Args:
            date_time (str): date and time of the recording session.
            subject (str):  name of the subject.
            area (str): recorded area.
            experiment (str): experiment number.
            recording (str): recording number.
            lfps (np.ndarray): array of shape (trials x ch x time) containing the lfp values at each ms.

            ------ bhv ---------
            block (np.ndarray): array of shape (trials) containing:
                                - 1 when is a DMTS trial.
                                - 2 when is a saccade task trial.
            trial_error (np.ndarray): array of shape (trials) containing:
                                - 0 when is a correct trial.
                                - n != 0 when is an incorrect trial. Each number correspond to different errors.
            code_samples (np.ndarray): array of shape (trials, events) containing the timestamp of the events
                                        (timestamps correspond to sp_sample index).
            code_numbers (np.ndarray): array of shape (trials, events) containing the codes of the events.
            position (np.ndarray): array of shape (trials, 2) containing the position of the stimulus.
            pos_code (np.ndarray): array of shape (trials) containing the position code of the stimulus.
                                    - for block 1: 1 is for 'in', -1 is for 'out' the receptive field.
                                    - for block 2: codes from 120 to 127 corresponding to the 8 target positions.
            sample_id (np.ndarray): array of shape (trials) containing the sample presented in each trial of block 1:
                                    - 0: neutral sample
                                    - 11: orientation 1, color 1.
                                    - 51: orientation 5, color 1.
                                    - 15: orientation 1, color 5.
                                    - 55: orientation 5, color 5.
            test_stimuli (np.ndarray): array of shape (trials,n_test_stimuli) containing the id of the test stimuli.
                                    As in sample_id, first number correspond to orientation and second to color.
            test_distractor (np.ndarray): array of shape (trials,n_test_stimuli) containing the id of the test distractor.
                                    As in sample_id, first number correspond to orientation and second to color.
        """

        self.date_time = date_time
        self.subject = subject
        self.area = area
        self.experiment = experiment
        self.recording = recording
        # -----signal--------
        self.lfps = lfps
        self.ch_row = ch_row
        self.ch_col = ch_col
        # -------bhv-------
        self.block = block
        self.trial_error = trial_error
        self.code_samples = code_samples
        self.code_numbers = code_numbers
        self.position = position
        self.pos_code = pos_code
        self.sample_id = sample_id
        self.test_stimuli = test_stimuli
        self.test_distractor = test_distractor
        # ------extra------
        self.filt = filt
        self.f_lp = f_lp
        self.f_hp = f_hp

    def to_python_hdf5(self, save_path: Path):
        """Save data in hdf5 format."""
        # Prepare a dictionary of attributes to save
        save_keys = {
            "lfps": (self.lfps, np.int64),
            "code_samples": (self.code_samples, np.int32),
            "code_numbers": (self.code_numbers, np.int16),
            "sample_id": (self.sample_id, np.int8),
            "test_distractor": (self.test_distractor, np.int8),
            "test_stimuli": (self.test_stimuli, np.int8),
        }
        exclude_keys = save_keys.keys()
        metadata = {
            "date_time",
            "subject",
            "area",
            "experiment",
            "recording",
        }
        # save the data
        with h5py.File(save_path, "w") as f:
            group = f.create_group("data")
            # Save defined datasets with specified dtypes and compression
            for key, (value, idtype) in save_keys.items():
                # Replace np.nan with -2
                value = value.copy()
                value[np.isnan(value)] = -2
                group.create_dataset(
                    key, data=value, dtype=idtype, compression="gzip"
                )
            # Save remaining attributes from self.__dict__
            for key, value in self.__dict__.items():
                if key not in exclude_keys:
                    if key in metadata:
                        group.attrs[key] = value
                    else:
                        compression = "gzip"
                        if np.ndim(value) == 0:
                            value = np.array(value)
                            compression = None
                        group.create_dataset(
                            key,
                            value.shape,
                            data=value,
                            compression=compression,
                        )

    @classmethod
    def from_python_hdf5(cls, load_path: Path):
        """Load data from a file in hdf5 format from Python."""
        lfp_data = {}
        with h5py.File(load_path, "r") as f:
            group = f["data"]
            lfp_data["date_time"] = group.attrs["date_time"]
            lfp_data["subject"] = group.attrs["subject"]
            lfp_data["area"] = group.attrs["area"]
            lfp_data["experiment"] = group.attrs["experiment"]
            lfp_data["recording"] = group.attrs["recording"]
            # Define the keys that require special processing.
            special_keys = [
                "lfps",
                "code_samples",
                "code_numbers",
                "sample_id",
                "test_distractor",
                "test_stimuli",
            ]
            # Iterate over all datasets in the group.
            for key, value in group.items():
                # arr = np.array(value, dtype=np.float32)
                arr = np.array(value)
                if key in special_keys:
                    arr = arr.astype(np.float64)
                    arr[arr == -2] = np.nan
                lfp_data[key] = arr
        return cls(**lfp_data)

    @staticmethod
    def indep_roll(
        arr: np.ndarray, shifts: np.ndarray, axis: int = 1
    ) -> np.ndarray:
        """Apply an independent roll for each dimensions of a single axis.
        Args:
            arr (np.ndarray): Array of any shape.
            shifts (np.ndarray): How many shifting to use for each dimension. Shape: `(arr.shape[axis],)`.
            axis (int, optional): Axis along which elements are shifted. Defaults to 1.

        Returns:
            np.ndarray: shifted array.
        """
        arr = np.swapaxes(arr, axis, -1)
        all_idcs = np.ogrid[[slice(0, n) for n in arr.shape]]
        # Convert to a positive shift
        shifts[shifts < 0] += arr.shape[-1]
        all_idcs[-1] = all_idcs[-1] - shifts[:, np.newaxis]
        result = arr[tuple(all_idcs)]
        arr = np.swapaxes(result, -1, axis)
        return arr
