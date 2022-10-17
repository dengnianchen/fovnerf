from .__common__ import *

__all__ = ["density2energy", "energy2alpha", "density2alpha", "VolumnRenderer"]


def density2energy(densities: Tensor, dists: Tensor, raw_noise_std: float = 0) -> Tensor:
    """
    Calculate energies from densities inferred by model.

    :param densities `Tensor(N...)`: model's output densities
    :param dists `Tensor(N...)`: integration times
    :param raw_noise_std `float`: the noise std used to egularize network during training (prevents 
                                  floater artifacts), defaults to 0, means no noise is added
    :return `Tensor(N...)`: energies which block light rays
    """
    if raw_noise_std > 0:
        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        densities = densities + torch.normal(0.0, raw_noise_std, densities.shape,
                                             device=densities.device)
    return torch.relu(densities) * dists


def energy2alpha(energies: Tensor) -> Tensor:
    """
    Convert energies to alphas.

    :param energies `Tensor(N...)`: energies (calculated from densities)
    :return `Tensor(N...)`: alphas
    """
    return 1.0 - torch.exp(-energies)


def density2alpha(densities: Tensor, dists: Tensor, raw_noise_std: float = 0) -> Tensor:
    """
    Calculate alphas from densities inferred by model.

    :param densities `Tensor(N...)`: model's output densities
    :param dists `Tensor(N...)`: integration times
    :param raw_noise_std `float`: the noise std used to regularize network during training (prevents 
                                  floater artifacts), defaults to 0, means no noise is added
    :return `Tensor(N...)`: alphas
    """
    return energy2alpha(density2energy(densities, dists, raw_noise_std))


class VolumnRenderer(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, samples: Samples, rgbd: Tensor, *outputs: str,
                 white_bg: bool, raw_noise_std: float) -> RenderOut:
        """
        Perform volumn rendering.

        :param samples `Samples(B, P)`: samples
        :param rgbd `Tensor(B, P, C+1)`: colors and densities
        :param outputs `str...`: items should be contained in the result dict.
                Optional values include "color", "depth", "colors", "densities", "alphas", "weights"
        :return `RenderOut`: render result
        """
        alphas = density2alpha(rgbd[..., -1], samples.dists, raw_noise_std)  # (B, P)
        weights = (alphas * torch.cumprod(union(1, 1. - alphas + 1e-10), -1)[..., :-1])[..., None]
        output_fn = {
            "color": lambda: torch.sum(weights * rgbd[..., :-1], -2) + (1. - torch.sum(weights, -2)
                                                                        if white_bg else 0.),
            "depth": lambda: torch.sum(weights * samples.depths[..., None], -2),
            "colors": lambda: rgbd[..., :-1],
            "densities": lambda: rgbd[..., -1:],
            "alphas": lambda: alphas[..., None],
            "weights": lambda: weights
        }
        return RenderOut({key: output_fn[key]() for key in outputs if key in output_fn})
