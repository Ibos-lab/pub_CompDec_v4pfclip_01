"""

Author: Camila Losada
"""

import numpy as np
from pathlib import Path
import h5py


class SpikeData:
    def __init__(
        self,
        date_time: str,
        subject: str,
        area: str,
        experiment: str,
        recording: str,
        # -----sp-----
        sp_samples: np.ndarray,
        clusters_id: np.ndarray,
        clusters_ch: np.ndarray,
        ch_row: np.ndarray,
        ch_col: np.ndarray,
        clusters_group: np.ndarray,
        clusters_depth: np.ndarray,
    ):
        """Initialize the class.

        This class contains information about all the clusters recoded in one area in one session.

        Args:
            date_time (str): date and time of the recording session
            subject (str): name of the subject
            area (str): area recorded
            experiment (str): experiment number
            recording (str): recording number
            ------ sp ---------
            sp_samples (np.ndarray): array of shape (neurons x time) containing the number of spikes at each ms.
            clusters_id (np.ndarray): array of shape (neurons,1) containing the kilosort cluster ID.
            clusters_ch (np.ndarray): array of shape (neurons,1) containing the electrode channel that recorded the activity of each cluster.
            clusters_group (np.ndarray): array of shape (neurons,1) containing "good" when is a neuron or "mua" when is multi unit activity.
            clusters_depth (np.ndarray): array of shape (neurons,1) containing the de depth of each cluster.
        """

        self.date_time = date_time
        self.subject = subject
        self.area = area
        self.experiment = experiment
        self.recording = recording
        self.sp_samples = sp_samples
        self.clusters_id = clusters_id
        self.clusters_ch = clusters_ch
        self.ch_row = ch_row
        self.ch_col = ch_col
        self.clusters_group = clusters_group
        self.clusters_depth = clusters_depth

    def to_python_hdf5(self, save_path: Path):
        """Save data in HDF5 format."""
        with h5py.File(save_path, "w") as f:
            group = f.create_group("data")
            group.attrs["date_time"] = self.date_time
            group.attrs["subject"] = self.subject
            group.attrs["area"] = self.area
            group.attrs["experiment"] = self.experiment
            group.attrs["recording"] = self.recording
            group.create_dataset(
                "sp_samples",
                self.sp_samples.shape,
                compression="gzip",
                data=self.sp_samples,
            )
            group.create_dataset(
                "clusters_id",
                self.clusters_id.shape,
                data=self.clusters_id,
                compression="gzip",
            )
            group.create_dataset(
                "clusters_ch",
                self.clusters_ch.shape,
                data=self.clusters_ch,
                compression="gzip",
            )
            group.create_dataset(
                "ch_row",
                self.ch_row.shape,
                data=self.ch_row,
                compression="gzip",
            )
            group.create_dataset(
                "ch_col",
                self.ch_col.shape,
                data=self.ch_col,
                compression="gzip",
            )
            group.create_dataset(
                "clusters_group",
                self.clusters_group.shape,
                data=self.clusters_group,
                compression="gzip",
            )
            group.create_dataset(
                "clusters_depth",
                self.clusters_depth.shape,
                data=self.clusters_depth,
                compression="gzip",
            )

    @classmethod
    def from_python_hdf5(cls, load_path: Path):
        """Load data from an HDF5 file in Python."""
        with h5py.File(load_path, "r") as f:
            #  get data
            group = f["data"]
            date_time = group.attrs["date_time"]
            subject = group.attrs["subject"]
            area = group.attrs["area"]
            experiment = group.attrs["experiment"]
            recording = group.attrs["recording"]
            sp_samples = group["sp_samples"][:]
            clusters_id = group["clusters_id"][:]
            clusters_ch = group["clusters_ch"][:]
            ch_row = group["ch_row"][:]
            ch_col = group["ch_col"][:]
            clusters_group = group["clusters_group"][:].astype(str)
            clusters_depth = group["clusters_depth"][:]

        # create class object and return
        trials_data = {
            "date_time": date_time,
            "subject": subject,
            "area": area,
            "experiment": experiment,
            "recording": recording,
            "sp_samples": sp_samples,
            "clusters_id": clusters_id,
            "clusters_ch": clusters_ch,
            "ch_row": ch_row,
            "ch_col": ch_col,
            "clusters_group": clusters_group,
            "clusters_depth": clusters_depth,
        }
        return cls(**trials_data)
