import torch
import numpy as np
from torch.utils.data import Dataset
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
from dreamerv2.training.config_ import RacingCarConfig
from gazebo_env import GazeboEnv
from gazebo_wrappers import ImageEnv, OneHotAction
from dreamerv2.training.config_ import RacingCarConfig

class PTImageDataset(Dataset):
    """
    For autoencoder training (single frames)
    """
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.images = data["srcimage"]  # (N,84,84,3)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        return self._to_tensor(img)

    def _to_tensor(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.ndim == 3:
            img = img.permute(2, 0, 1)
        img = img.float()
        if img.max() > 1:
            img = img / 255.0
        return img


# class PTSequenceDataset(Dataset):
#     """
#     For temporal prediction (frame_t -> frame_t+1)
#     """
#     def __init__(self, pt_path):
#         data = torch.load(pt_path)
#         self.images = data["srcimage"]

#     def __len__(self):
#         return len(self.images) - 1

#     def __getitem__(self, idx):
#         img_t  = self._to_tensor(self.images[idx])
#         img_t1 = self._to_tensor(self.images[idx + 1])
#         return img_t, img_t1

#     def _to_tensor(self, img):
#         if isinstance(img, np.ndarray):
#             img = torch.from_numpy(img)
#         if img.ndim == 3:
#             img = img.permute(2, 0, 1)
#         img = img.float()
#         if img.max() > 1:
#             img = img / 255.0
#         return img

class PTSequenceDataset(Dataset):
    def __init__(self, pt_path, horizon=4):
        data = torch.load(pt_path)
        self.images = data["srcimage"]
        self.horizon = horizon

    def __len__(self):
        return len(self.images) - self.horizon

    def __getitem__(self, idx):
        seq = []
        for t in range(self.horizon + 1):
            img = self.images[idx + t]
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            if img.max() > 1:
                img = img / 255.0
            seq.append(img)
        return torch.stack(seq)  # (H+1, 3, 84, 84)