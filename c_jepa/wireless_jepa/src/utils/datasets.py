from torch.utils.data import Dataset
import torch
import numpy as np


class ChannelDataset(Dataset):
    def __init__(self, csi_time_domain, groundtruth_positions):

        self.csi_time_domain = csi_time_domain
        self.groundtruth_positions = groundtruth_positions

    def __getitem__(self, index):
        return (
            self.csi_time_domain[index],
            self.groundtruth_positions[index],
            index,
        )

    def __len__(self):
        return len(self.csi_time_domain)
    
class PowerDataset(Dataset):
    def __init__(self, chart, power):

        self.chart = chart
        self.power = power

    def __getitem__(self, index):
        return (
            self.chart[index],
            self.power[index],
        )

    def __len__(self):
        return len(self.chart)

class JEPA_ChannelDataset(Dataset):
    def __init__(self, csi_time_domain, latent_dynamics, seq_len, window_len):
        self.csi_time_domain = csi_time_domain
        self.latent_dynamics = latent_dynamics
        self.seq_len = seq_len
        self.window_len = window_len

        def get_indices(data, window_size, step_size):
            stop = len(data)
            idxs = []
            s = 0
            e = window_size
            while e <= stop:
                idxs.append((s, e))
                s += step_size
                e += step_size
            return idxs

        self.indices = get_indices(
            self.csi_time_domain,
            self.seq_len,
            self.window_len
        )

    def __getitem__(self, index):
        s, e = self.indices[index]
        return (
            self.csi_time_domain[s:e],
            self.latent_dynamics[s:e-1],
        )

    def __len__(self):
        return len(self.indices)
