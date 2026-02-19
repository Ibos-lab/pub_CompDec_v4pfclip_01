"""

Author: Camila Losada
"""

import h5py
import logging
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from compdec_v4pfclip_01.utils import config_task
from compdec_v4pfclip_01.utils.trials import align_trials

logger = logging.getLogger(__name__)


class NeuronData:
    def __init__(
        self,
        date_time: str,
        subject: str,
        area: str,
        experiment: str,
        recording: str,
        # --------sp-------
        sp_samples: np.ndarray,
        cluster_id: int,
        cluster_ch: int,
        ch_row: int,
        ch_col: int,
        cluster_group: str,
        cluster_number: int,
        cluster_array_pos: int,
        cluster_depth: int,
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
        **kwargs,
    ):
        """Initialize the class.

        This class contains information about each cluster.

        Args:
            date_time (str): date and time of the recording session
            subject (str):  name of the subject
            area (str): area recorded
            experiment (str): experiment number
            recording (str): recording number
            ------ sp ---------
            sp_samples (np.ndarray): array of shape (trials x time) containing the number of spikes at each ms in each trial.
            cluster_id (int): kilosort cluster ID.
            cluster_ch (int): electrode channel that recorded the activity of the cluster.
            ch_row (int): # TODO compleate docstring
            ch_col (int):
            cluster_group (str): "good" if it is a neuron or "mua" if it is a multi unit activity.
            cluster_number (int): number of good or mua.
            cluster_array_pos (int): position of the cluster in SpikeDate.sp_samples.
            cluster_depth (int): depth of the cluster.
            ------ bhv ---------
            block (np.ndarray): array of shape (trials) containing:
                                - 1 when is a DMTS trial
                                - 2 when is a saccade task trial
            trial_error (np.ndarray): array of shape (trials) containing:
                                - 0 when is a correct trial
                                - n != 0 when is an incorrect trial. Each number correspond to different errors
            code_samples (np.ndarray): array of shape (trials, events) containing the timestamp of the events
                                        (timestamps correspond to sp_sample index).
            code_numbers (np.ndarray): array of shape (trials, events) containing the codes of the events.
            position (np.ndarray): array of shape (trials, 2) containing the position of the stimulus.
            pos_code (np.ndarray): array of shape (trials) containing the position code of the stimulus.
                                    - for block 1: 1 is for 'in', -1 is for 'out' the receptive field
                                    - for block 2: codes from 120 to 127 corresponding to the 8 target positions.
            sample_id (np.ndarray): array of shape (trials) containing the sample presented in each trial of block 1:
                                    - 0: neutral sample
                                    - 11: orientation 1, color 1
                                    - 51: orientation 5, color 1
                                    - 15: orientation 1, color 5
                                    - 55: orientation 5, color 5
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
        # --------sp-------
        self.sp_samples = sp_samples
        self.cluster_id = cluster_id
        self.cluster_ch = cluster_ch
        self.ch_row = ch_row
        self.ch_col = ch_col
        self.cluster_group = cluster_group
        self.cluster_number = cluster_number
        self.cluster_array_pos = cluster_array_pos
        self.cluster_depth = cluster_depth
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
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def edit_attributes(self, new_values: Dict):
        for attr_name, attr_value in zip(
            new_values.keys(), new_values.values()
        ):
            setattr(self, attr_name, attr_value)

    @classmethod
    def from_python_hdf5(cls, load_path: Path):
        """Load data from a file in hdf5 format from Python."""
        # load the data and create class object
        neu_data = {}
        with h5py.File(load_path, "r") as f:
            group = f["data"]
            neu_data["date_time"] = group.attrs["date_time"]
            neu_data["subject"] = group.attrs["subject"]
            neu_data["area"] = group.attrs["area"]
            neu_data["experiment"] = group.attrs["experiment"]
            neu_data["recording"] = group.attrs["recording"]
            neu_data["cluster_group"] = group.attrs["cluster_group"]
            # Define the keys that require special processing.
            special_keys = [
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
                    arr = arr.astype(np.float32)
                    arr[arr == -2] = np.nan
                neu_data[key] = arr

        return cls(**neu_data)

    def to_python_hdf5(self, save_path: Path):
        """Save data in hdf5 format."""
        # Prepare a dictionary of attributes to save
        save_keys = {
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
            "cluster_group",
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

    def get_neuron_id(self):
        nid = f"{self.date_time}_{self.subject}_{self.area}_e{self.experiment}_r{self.recording}_{self.cluster_group}{self.cluster_number}"
        return nid

    def _align_on(
        self,
        select_block: int,
        event: str,
        time_before: int,
        error_type: int,
        stim_loc: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align spike data on a specific event and return the aligned data along with a trial mask.

        Args:
            select_block (int): Block number to select trials from.
            event (str): Event to align on.
            time_before (int): Time before the event to include in the alignment.
            error_type (int): Type of error trials to include (e.g., 0 for correct trials).
            stim_loc (str): Position code for selecting trials.
                                - block 1: Must be one of "in", "out", "ipsi", or "contra".

        Raises:
            KeyError: If stim_loc is not one of "in", "out", "ipsi", or "contra".
            ValueError: If select_block is not valid.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Aligned spike data.
                - Mask for selecting trials from the original data.
        """
        # Determine the event code based on the block number
        if select_block == 1:
            code = config_task.EVENTS_B1[event]
            # Check stim_loc value
            if stim_loc == "in":
                stim_loc_int, rf_loc = 1, self.rf_loc
            elif stim_loc == "out":
                stim_loc_int, rf_loc = -1, self.rf_loc
            elif stim_loc == "contra":
                stim_loc_int, rf_loc = 1, self.pos_code
            elif stim_loc == "ipsi":
                stim_loc_int, rf_loc = -1, self.pos_code
            else:
                raise KeyError(
                    "Invalid stim_loc value: %s. Must be one of 'in', 'out', 'ipsi', or 'contra'."
                    % stim_loc
                )
            # Create mask to select trials based on position, error type, and block
            mask = (rf_loc == stim_loc_int) & (self.block == select_block)
            if error_type is not None:
                mask = mask & (self.trial_error == error_type)

        elif select_block == 2:
            code = config_task.EVENTS_B2[event]
            mask = self.block == select_block
            if error_type is not None:
                mask = mask & (self.trial_error == error_type)
        else:
            raise ValueError(
                f"Invalid block value: {select_block}. Must be 1 or 2."
            )

        sp_samples_m = self.sp_samples[mask]
        # Find event occurrences in the code_numbers matrix
        code_mask = np.where(self.code_numbers[mask] == code, True, False)
        # Check if the event occurred in each trial
        trials_mask = np.any(code_mask, axis=1)
        # Get the sample indices where the event occurred and shift by time_before
        shifts = self.code_samples[mask][code_mask]
        shifts = (shifts - time_before).astype(int)
        # Align spike data based on the calculated shifts
        align_sp = align_trials.indep_roll(
            arr=sp_samples_m[trials_mask], shifts=-shifts, axis=1
        )
        # Create mask for selecting the trials from the original matrix size
        tr = np.arange(self.sp_samples.shape[0])
        complete_mask = np.isin(tr, tr[mask][trials_mask])

        return align_sp, complete_mask

    def get_align_on(
        self,
        params: List,
        inplace: bool = False,
        delete_att: List = None,
        # rfloc: str = None,
        # rf_loc_df: pd.DataFrame = None,
    ):
        """Read, align, and add spiking activity to the NeuronData object.

        Args:
            params (List[dict]): List of dictionaries containing the following keys:
                - 'loc': str, location code ('in', 'out', 'ipsi', 'contra')
                - 'event': str, event name (e.g., 'sample_on')
                - 'time_before': int, time before event
                - 'time_after': int, time after event
                - 'select_block': int, block number
                - 'error_type': int,
            delete_att (List[str], optional): List of attribute names to delete. Defaults to None.
            rfloc (str, optional): A string specifying the receptive field location
                ('contra' or 'ipsi'). Defaults to None.
            rf_loc_df (pd.DataFrame, optional): A DataFrame containing neuron IDs ('nid')
                and their corresponding receptive field locations ('contra' or 'ipsi'). Defaults to None.
        Returns:
            NeuronData: The modified NeuronData object with added spiking activity.
        """
        aligned = []
        # if rfloc is not None or rf_loc_df is not None:
        #     self.add_rf_loc(rfloc=rfloc, rf_loc_df=rf_loc_df)
        for it in params:
            # Alignment and extraction of spike and mask data
            sp, mask = self._align_on(
                select_block=it["select_block"],
                event=it["event"],
                time_before=it["time_before"],
                error_type=it["error_type"],
                stim_loc=it["loc"],
            )
            endt = it["time_before"] + it["time_after"]
            # Set name based on the event and rf/stimulus location
            if it["select_block"] == 1:
                att_name = (
                    f"{config_task.EVENTS_B1_SHORT[it['event']]}_{it['loc']}"
                )
            elif it["select_block"] == 2:
                att_name = (
                    f"{config_task.EVENTS_B2_SHORT[it['event']]}_{it['loc']}"
                )

            sp_trimmed = np.array(sp[:, :endt], dtype=np.int8)
            mask_bool = np.array(mask, dtype=bool)
            t_before = np.array(it["time_before"], dtype=np.int32)

            if inplace:
                setattr(self, f"sp_{att_name}", sp_trimmed)
                setattr(self, f"mask_{att_name}", mask_bool)
                setattr(self, f"time_before_{att_name}", t_before)
            else:
                aligned.append(
                    {
                        f"sp_{att_name}": sp_trimmed,
                        f"mask_{att_name}": mask_bool,
                        f"time_before_{att_name}": t_before,
                    }
                )
        # Delete specified attributes if delete_att is provided
        if delete_att:
            for iatt in delete_att:
                if hasattr(self, iatt):
                    setattr(self, iatt, np.array([]))
                else:
                    logger.warning(
                        f"Warning: Attribute '{iatt}' does not exist and cannot be deleted."
                    )

        return self if inplace else aligned

    def add_rf_loc(self, rfloc: str = None, rf_loc_df: pd.DataFrame = None):
        """Adds receptive field position information to the NeuronData object.

        Args:
            rfloc (str, optional): A string specifying the receptive field location
                ('contra' or 'ipsi'). Defaults to None.
            rf_loc_df (pd.DataFrame, optional): A DataFrame containing neuron IDs ('nid')
                and their corresponding receptive field locations ('contra' or 'ipsi'). Defaults to None.

        Raises:
            ValueError: If neither `rfloc` nor `rf_loc_df` is provided.
            ValueError: If both `rfloc` and `rf_loc_df` are provided simultaneously.
            IndexError: If the neuron ID ('nid') is not found in `rf_loc_df`.
            ValueError: If rf_loc is not 'ipsi' or 'contra'.

        Returns:
            NeuronData: The modified NeuronData object.
        """
        if rfloc is None and rf_loc_df is None:
            raise ValueError("Receptive field information must be provided.")

        if rfloc is not None and rf_loc_df is not None:
            raise ValueError(
                "Receptive field information must be provided in only one argument."
            )

        if rf_loc_df is not None:
            nid = self.get_neuron_id()
            # Filter the DataFrame and check if any results are found
            filtered_rf_loc = rf_loc_df[rf_loc_df["nid"] == nid]
            if filtered_rf_loc.empty:
                raise IndexError(f"No rf_loc found for neuron ID {nid}")
            rfloc = filtered_rf_loc["rf_loc"].values[0]
        pos_code = self.pos_code
        rf_loc = np.zeros(pos_code.shape, dtype=np.int8)
        if rfloc == "ipsi":
            rf_loc[pos_code == 1] = -1  # out
            rf_loc[pos_code == -1] = 1  # in
        elif rfloc == "contra":
            rf_loc[pos_code == 1] = 1
            rf_loc[pos_code == -1] = -1
        else:
            raise ValueError('rfloc must be "ipsi" or "contra"')
        setattr(self, "rf_loc", rf_loc)
