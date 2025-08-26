import numpy as np
import torch
from torch.utils.data import Dataset

class HighSimDataset(Dataset):
    """
    DataLoader for the preprocessed highsim dataset.
    Feature indices in the raw .npz file:
    0: platoon_id, 1: gap, 2: speed, 3: speed_difference,
    4: acceleration, 5: pet, 6: ssdd, 7: vehicle_length,
    8: preceding_vehicle_length, 9: preceding_vehicle_speed
    """
    def __init__(self, data_path, data_name, in_length, out_length):
        """
        Args:
            data_path (str): Path to the .npz data file.
            data_name (str): The key for the data split (e.g., 'train_data').
            in_length (int): Length of the input history sequence.
            out_length (int): Length of the output prediction sequence.
        """
        data = np.load(data_path)
        self.data = data[data_name]

        # --- Input Features (History) ---
        # Selected features: [gap, speed, speed_diff, accel, ssdd, veh_len, pre_veh_len]
        hist_indices = [1, 2, 3, 4, 6, 7, 8]
        self.hist = self.data[:, :, :in_length, hist_indices]

        # --- Ground Truth Labels (Future) ---
        # Selected features: [gap, speed, ssdd]
        fut_indices = [1, 2, 6]
        self.fut = self.data[:, :, in_length : in_length + out_length, fut_indices]

        # --- External Input (Leader's Future Velocity) ---
        # Selected feature: [preceding_vehicle_speed]
        nextv_indices = [9]
        self.nextv = self.data[:, :1, in_length - 1 : in_length + out_length, nextv_indices]

    def __len__(self):
        return len(self.hist)

    def __getitem__(self, idx):
        history_data = self.hist[idx]
        future_data = self.fut[idx]
        leader_velocity_data = self.nextv[idx]

        return (
            torch.tensor(history_data, dtype=torch.float32),
            torch.tensor(future_data, dtype=torch.float32),
            torch.tensor(leader_velocity_data, dtype=torch.float32),
        )
