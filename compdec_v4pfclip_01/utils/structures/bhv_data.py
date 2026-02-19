"""

Author: Camila Losada
"""

import h5py
import numpy as np
from pathlib import Path


class BhvData:
    def __init__(
        self,
        # both
        block: np.ndarray,
        iti: np.ndarray,
        position: np.ndarray,
        reward_plus: np.ndarray,
        trial_error: np.ndarray,
        delay_time: np.ndarray,
        fix_time: np.ndarray,
        fix_window_radius: np.ndarray,
        idletime3: np.ndarray,
        rand_delay_time: np.ndarray,
        reward_dur: np.ndarray,
        wait_for_fix: np.ndarray,
        # sacc
        fix_post_sacc_blank: np.ndarray,
        max_reaction_time: np.ndarray,
        stay_time: np.ndarray,
        fix_fp_t_time: np.ndarray,
        fix_fp_post_t_time: np.ndarray,
        fix_fp_pre_t_time: np.ndarray,
        fix_close: np.ndarray,
        fix_far: np.ndarray,
        closeexc: np.ndarray,
        excentricity: np.ndarray,
        farexc: np.ndarray,
        # dmts
        eye_ml: np.ndarray,
        condition: np.ndarray,
        code_numbers: np.ndarray,
        code_times: np.ndarray,
        stim_match: np.ndarray,
        pos_code: np.ndarray,
        stim_total: np.ndarray,
        test_distractor: np.ndarray,
        test_stimuli: np.ndarray,
        sample_time: np.ndarray,
        test_time: np.ndarray,
        sample_id: np.ndarray,
        **kwargs,
    ):
        """BhvData contains behavioral data.

        Args:
            block (np.ndarray): shape: (trials).
            iti (np.ndarray): duration of the intertrial interval. shape: (trials).
            position (np.ndarray): position of the stimulus.  shape: (trials, 2).
            reward_plus (np.ndarray): the amount of reward if more was given. shape: (trials).
            trial_error (np.ndarray): if 0: correct trial else: code of the error. shape: (trials).
            delay_time (np.ndarray): duration of the delay. shape: (trials).
            fix_time (np.ndarray): duration of the fixation. shape: (trials).
            fix_window_radius (np.ndarray): shape: (trials).
            idletime3 (np.ndarray): #TODO
            rand_delay_time (np.ndarray): range of the delay variation. shape: (trials).
            reward_dur (np.ndarray): duration of the reward. shape: (trials).
            wait_for_fix (np.ndarray): max time to fixate before the trial starts. shape: (trials).
            sacc_code (np.ndarray): shape: (trials).
            fix_post_sacc_blank (np.ndarray):
            max_reaction_time (np.ndarray): max time the monkey has to do the sacc
            stay_time (np.ndarray): post sacc fix. shape: (trials).
            fix_fp_t_time (np.ndarray): fixation fix point target time
            fix_fp_post_t_time (np.ndarray):#TODO
            fix_fp_pre_t_time (np.ndarray):#TODO
            fix_close (np.ndarray):#TODO
            fix_far (np.ndarray): scaling #TODO
            closeexc (np.ndarray):#TODO
            excentricity (np.ndarray):#TODO
            farexc (np.ndarray):#TODO
            eye_ml (np.ndarray):#TODO
            condition (np.ndarray): condition in the txt file. shape: (trials).
            code_numbers (np.ndarray):array of shape (trials, events) containing the codes of the events.
            code_samples (np.ndarray):array of shape (trials, events) containing the timestamp of the events
            code_times (np.ndarray): exact time when each event ocurred during the trial. shape: (trials, events).
            stim_match (np.ndarray):#TODO
            pos_code (np.ndarray): position of the sample stimulus. shape: (trials, 2).
            stim_total (np.ndarray):#TODO
            test_distractor (np.ndarray): orientation, color
            test_stimuli (np.ndarray): orientation, color
            sample_time (np.ndarray):#TODO
            test_time (np.ndarray):#TODO
            sample_id (np.ndarray):#TODO
        """

        self.block = block
        self.code_numbers = code_numbers
        self.code_times = code_times
        self.condition = condition
        self.eye_ml = eye_ml
        self.fix_fp_t_time = fix_fp_t_time
        self.fix_fp_post_t_time = fix_fp_post_t_time
        self.fix_fp_pre_t_time = fix_fp_pre_t_time
        self.fix_close = fix_close
        self.fix_far = fix_far
        self.iti = iti
        self.stim_match = stim_match
        self.pos_code = pos_code
        self.position = position
        self.reward_plus = reward_plus
        self.test_distractor = test_distractor
        self.test_stimuli = test_stimuli
        self.sample_id = sample_id
        self.stim_total = stim_total
        self.trial_error = trial_error
        self.closeexc = closeexc
        self.delay_time = delay_time
        self.excentricity = excentricity
        self.farexc = farexc
        self.fix_post_sacc_blank = fix_post_sacc_blank
        self.fix_time = fix_time
        self.fix_window_radius = fix_window_radius
        self.idletime3 = idletime3
        self.max_reaction_time = max_reaction_time
        self.rand_delay_time = rand_delay_time
        self.reward_dur = reward_dur
        self.sample_time = sample_time
        self.stay_time = stay_time
        self.test_time = test_time
        self.wait_for_fix = wait_for_fix
        # # Attach any additional attributes
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @classmethod
    def from_python_hdf5(cls, load_path: Path):
        """Load data from a file in hdf5 format from Python."""
        bhv_data = {}
        with h5py.File(load_path, "r") as f:
            group = f["data"]
            bhv_data["date_time"] = group.attrs["date_time"]
            bhv_data["subject"] = group.attrs["subject"]
            bhv_data["experiment"] = group.attrs["experiment"]
            bhv_data["recording"] = group.attrs["recording"]
            for key, value in group.items():
                bhv_data[key] = value[:]

        # Convert specific keys to float32 and replace -2 with np.nan
        for key in [
            "code_numbers",
            "code_samples",
            "code_times",
            "sample_id",
            "test_distractor",
            "test_stimuli",
            "stim_total",
        ]:
            bhv_data[key] = bhv_data[key].astype(np.float32)
            bhv_data[key][bhv_data[key] == -2] = np.nan
        return cls(**bhv_data)

    def to_python_hdf5(self, save_path: Path):
        """Save data in hdf5 format."""
        # Prepare a dictionary of attributes to save
        save_keys = {
            "code_numbers": (self.code_numbers, np.int16),
            "code_samples": (self.code_samples, np.int32),
            "code_times": (self.code_times, np.float32),
            "test_distractor": (self.test_distractor, np.int8),
            "test_stimuli": (self.test_stimuli, np.int8),
            "sample_id": (self.sample_id, np.int8),
            "stim_total": (self.stim_total, np.int8),
        }
        metadata = {"date_time", "subject", "experiment", "recording"}
        exclude_keys = save_keys.keys()
        with h5py.File(save_path, "w") as f:
            group = f.create_group("data")
            # Save defined datasets with specified dtypes and compression
            for key, (value, idtype) in save_keys.items():
                # Replace np.nan with -2
                value = value.copy()
                value[np.isnan(value)] = -2
                group.create_dataset(key, data=value, dtype=idtype, compression="gzip")
            # Save remaining attributes from self.__dict__
            for key, value in self.__dict__.items():
                if key not in exclude_keys:
                    if key in metadata:
                        group.attrs[key] = value
                    else:
                        group.create_dataset(key, data=value)
