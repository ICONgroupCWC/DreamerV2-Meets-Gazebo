import torch
import torch.nn as nn
import numpy as np
import collections


class FeatureLayer(nn.Module):
    def __init__(self):
        super(FeatureLayer, self).__init__()

    def forward(self, csi):
        # autocorrelations = torch.einsum("damt,dbnt->dtabmn", csi, torch.conj(csi))
        # return torch.nn.functional.normalize(
        #     torch.flatten(
        #         torch.stack(
        #             [torch.real(autocorrelations), torch.imag(autocorrelations)], dim=-1
        #         ),
        #         start_dim=1,
        #         end_dim=-1,
        #     )
        # )
        return torch.nn.functional.normalize(
            torch.flatten(
                torch.stack(
                    [torch.real(csi), torch.imag(csi)], dim=-1
                ),
                start_dim=1,
                end_dim=-1,
            )
        )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            collections.OrderedDict(
                [
                    ("feature", FeatureLayer()),
                    # ("faltten", nn.Flatten()),
                    # ("fc1", nn.Linear(2 * 16 * 64 * 16, 1024)),
                    ("fc1", nn.Linear(768, 1024)),
                    ("relu1", nn.ReLU()),
                    ("bn1", nn.BatchNorm1d(1024)),
                    ("fc2", nn.Linear(1024, 512)),
                    ("relu2", nn.ReLU()),
                    ("bn2", nn.BatchNorm1d(512)),
                    ("fc3", nn.Linear(512, 256)),
                    ("relu3", nn.ReLU()),
                    ("bn3", nn.BatchNorm1d(256)),
                    ("fc4",     nn.Linear(256, 128)),
                    ("relu4",   nn.ReLU()),
                    ("bn4",     nn.BatchNorm1d(128)),
                    ("fc5", nn.Linear(128, 64)),
                    ("relu5", nn.ReLU()),
                    ("bn5", nn.BatchNorm1d(64)),
                    ("out", nn.Linear(64, 2)),
                ]
            )
        )

    def forward(self, csi):
        return self.encoder(csi)


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(Predictor, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # self.norm = nn.BatchNorm1d(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        # self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim, device=x.device
        ).requires_grad_()
        # out, h = self.gru(self.norm(x.permute(0,2,1)).permute(0,2,1), h0.detach())
        out, h = self.gru(x, h0.detach())
        out = self.fc(out)
        return out, h


# class PowerPredictor(nn.Module):
#     def __init__(self):
#         super(PowerPredictor, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(2, 32),
#             nn.ReLU(),
#             nn.Linear(32, 32),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16,1)
#         )
    
#     def forward(self, x):
#         return self.net(x)

class PowerPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)