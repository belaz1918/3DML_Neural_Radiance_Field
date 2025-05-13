"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()

        # TODO
        # In __init__:
        
        self.feat_dim = feat_dim
        self.input_pos_dim = pos_dim
        self.input_view_dir_dim = view_dir_dim

        # Input layer
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_pos_dim, feat_dim))  # 1st layer

        # Layers 2-4: 256 -> 256
        for _ in range(3):
            self.layers.append(nn.Linear(feat_dim, feat_dim))

        # 5th layer: (256+pos_dim) -> 256, because of skip connection
        self.layers.append(nn.Linear(feat_dim + self.input_pos_dim, feat_dim))

        # Layers 6-9: 256 -> 256
        for _ in range(4):
            self.layers.append(nn.Linear(feat_dim, feat_dim))

        # Sigma (density) head
        self.sigma_head = nn.Linear(feat_dim, 1)

        # Feature for color head
        self.feature_head = nn.Linear(feat_dim, feat_dim)

        # Color (RGB) head: takes feature + view_dir encoding as input
        self.color_layer1 = nn.Linear(feat_dim + self.input_view_dir_dim, 128)
        self.color_layer2 = nn.Linear(128, 3)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # raise NotImplementedError("Task 1")

    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """

        # TODO
        # In forward:
        # TODO
        x = pos
        for i, layer in enumerate(self.layers):
            if i == 4:  # after 4th layer, do skip connection
                x = torch.cat([x, pos], dim=-1)
            x = self.relu(layer(x))

        sigma = self.relu(self.sigma_head(x))      # (num_sample, 1)
        feat = self.feature_head(x)                # (num_sample, feat_dim)

        # Concatenate feature with view_dir encoding
        h = torch.cat([feat, view_dir], dim=-1)
        h = self.relu(self.color_layer1(h))
        rgb = self.sigmoid(self.color_layer2(h))   # (num_sample, 3)

        return sigma, rgb

        # raise NotImplementedError("Task 1")
