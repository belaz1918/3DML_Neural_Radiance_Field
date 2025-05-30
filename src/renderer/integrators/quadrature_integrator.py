"""
Integrator implementing quadrature rule.
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
from torch_nerf.src.renderer.integrators.integrator_base import IntegratorBase


class QuadratureIntegrator(IntegratorBase):
    """
    Numerical integrator which approximates integral using quadrature.
    """

    @jaxtyped
    @typechecked
    def integrate_along_rays(
        self,
        sigma: Float[torch.Tensor, "num_ray num_sample"],
        radiance: Float[torch.Tensor, "num_ray num_sample 3"],
        delta: Float[torch.Tensor, "num_ray num_sample"],
    ) -> Tuple[Float[torch.Tensor, "num_ray 3"], Float[torch.Tensor, "num_ray num_sample"]]:
        """
        Computes quadrature rule to approximate integral involving in volume rendering.
        Pixel colors are computed as weighted sums of radiance values collected along rays.

        For details on the quadrature rule, refer to 'Optical models for
        direct volume rendering (IEEE Transactions on Visualization and Computer Graphics 1995)'.

        Args:
            sigma: Density values sampled along rays.
            radiance: Radiance values sampled along rays.
            delta: Distance between adjacent samples along rays.

        Returns:
            rgbs: Pixel colors computed by evaluating the volume rendering equation.
            weights: Weights used to determine the contribution of each sample to the final pixel color.
                A weight at a sample point is defined as a product of transmittance and opacity,
                where opacity (alpha) is defined as 1 - exp(-sigma * delta).
        """
        # TODO
        # Compute alpha (opacity) for each sample
        alpha = 1.0 - torch.exp(-sigma * delta)  # (num_ray, num_sample)

        # Compute transmittance T_i = exp(-sum_{j=1}^{i-1} sigma_j * delta_j)
        # Shifted cumulative sum for exclusive sum (so T_1 = 1)
        sigma_delta = sigma * delta  # (num_ray, num_sample)
        # Pad with zeros at the start for exclusive sum
        sigma_delta_shifted = torch.cat([torch.zeros_like(sigma_delta[:, :1]), sigma_delta[:, :-1]], dim=-1)
        transmittance = torch.exp(-torch.cumsum(sigma_delta_shifted, dim=-1))  # (num_ray, num_sample)

        # Compute weights: w_i = T_i * alpha_i
        weights = transmittance * alpha  # (num_ray, num_sample)

        # Weighted sum of radiance for each ray
        rgbs = torch.sum(weights.unsqueeze(-1) * radiance, dim=1)  # (num_ray, 3)

        return rgbs, weights

        # HINT: Look up the documentation of 'torch.cumsum'.
        # raise NotImplementedError("Task 3")
