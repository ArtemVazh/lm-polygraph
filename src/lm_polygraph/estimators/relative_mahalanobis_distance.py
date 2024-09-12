import os
import numpy as np
import torch

from typing import Dict

from .estimator import Estimator
from .mahalanobis_distance import (
    compute_inv_covariance,
    mahalanobis_distance_with_known_centroids_sigma_inv,
    MahalanobisDistanceSeq,
    create_cuda_tensor_from_numpy,
)


def save_array(array, filename):
    with open(filename, "wb") as f:
        np.save(f, array)


def load_array(filename):
    with open(filename, "rb") as f:
        array = np.load(f)
    return array


class RelativeMahalanobisDistanceSeq(Estimator):
    """
    Ren et al. (2023) showed that it might be useful to adjust the Mahalanobis distance score by subtracting
    from it the other Mahalanobis distance MD_0(x) computed for some large general purpose dataset covering many domain.
    RMD(x) = MD(x) - MD_0(x)
    """

    def __init__(
        self,
        embeddings_type: str = "decoder",
        parameters_path: str = None,
        normalize: bool = False,
        hidden_layer: int = -1,
        device: str = "cuda",
        storage_device: str = "cuda",
    ):
        self.hidden_layer = hidden_layer
        if self.hidden_layer == -1:
            self.hidden_layer_name = ""
        else:
            self.hidden_layer_name = f"_{self.hidden_layer}"
        super().__init__(
            [
                f"embeddings{self.hidden_layer_name}",
                f"train_embeddings{self.hidden_layer_name}",
                f"background_train_embeddings{self.hidden_layer_name}",
            ],
            "sequence",
        )
        self.centroid_0 = None
        self.sigma_inv_0 = None
        self.parameters_path = parameters_path
        self.embeddings_type = embeddings_type
        self.normalize = normalize
        self.min = 1e100
        self.max = -1e100
        self.device = device
        self.storage_device = storage_device

        self.MD = MahalanobisDistanceSeq(
            embeddings_type,
            parameters_path,
            normalize=False,
            hidden_layer=self.hidden_layer,
            device=device, 
            storage_device=storage_device
        )
        self.is_fitted = False

        if self.parameters_path is not None:
            self.full_path = f"{self.parameters_path}/rmd_{self.embeddings_type}{self.hidden_layer_name}"
            os.makedirs(self.full_path, exist_ok=True)
            if os.path.exists(f"{self.full_path}/centroid_0.pt"):
                self.centroid_0 = torch.load(
                    f"{self.full_path}/centroid_0.pt", weights_only=False
                ).to(self.storage_device)
                self.sigma_inv_0 = torch.load(
                    f"{self.full_path}/sigma_inv_0.pt", weights_only=False
                ).to(self.storage_device)
                self.max = load_array(f"{self.full_path}/max_0.npy")
                self.min = load_array(f"{self.full_path}/min_0.npy")
                self.is_fitted = True

    def __str__(self):
        return f"RelativeMahalanobisDistanceSeq_{self.embeddings_type}{self.hidden_layer_name}"

    def __call__(self, stats: Dict[str, np.ndarray], save_data: bool = True) -> np.ndarray:
        # take the embeddings
        embeddings = create_cuda_tensor_from_numpy(
            stats[f"embeddings_{self.embeddings_type}{self.hidden_layer_name}"]
        )

        # since we want to adjust resulting reasure on baseline MD on train part
        # we have to compute average train centroid and inverse cavariance matrix
        # to obtain MD_0

        if not self.is_fitted:
            centroid_key = f"rmd_centroid{self.hidden_layer_name}"
            if (centroid_key in stats.keys()): # to reduce number of stored centroid for multiple methods used the same data
                self.centroid_0 = stats[centroid_key]
            else:
                background_train_embeddings = create_cuda_tensor_from_numpy(
                    stats[
                        f"background_train_embeddings_{self.embeddings_type}{self.hidden_layer_name}"
                    ]
                )
                self.centroid_0 = background_train_embeddings.mean(axis=0)
                if self.storage_device == "cpu":
                    self.centroid_0 = self.centroid_0.cpu()
                if self.parameters_path is not None:
                    torch.save(self.centroid_0, f"{self.full_path}/centroid_0.pt")
                if save_data:
                    stats[centroid_key] = self.centroid_0

        if not self.is_fitted:
            covariance_key = f"rmd_covariance{self.hidden_layer_name}"
            if (covariance_key in stats.keys()): # to reduce number of stored centroid for multiple methods used the same data
                self.sigma_inv_0 = stats[covariance_key]
            else:
                background_train_embeddings = create_cuda_tensor_from_numpy(
                    stats[
                        f"background_train_embeddings_{self.embeddings_type}{self.hidden_layer_name}"
                    ]
                )
                self.sigma_inv_0, _ = compute_inv_covariance(
                    self.centroid_0.unsqueeze(0), background_train_embeddings
                )
                if self.storage_device == "cpu":
                    self.sigma_inv_0 = self.sigma_inv_0.cpu()
                if self.parameters_path is not None:
                    torch.save(self.sigma_inv_0, f"{self.full_path}/sigma_inv_0.pt")
                if save_data:
                    stats[covariance_key] = self.sigma_inv_0
            self.is_fitted = True

        # compute MD_0

        if self.device == "cuda" and self.storage_device == "cpu":
            if embeddings.shape[0] < 20:
                # force compute on cpu, since for a small number of embeddings it will be faster than move to cuda 
                dists_0 = (
                    mahalanobis_distance_with_known_centroids_sigma_inv(
                        self.centroid_0.float(),
                        None,
                        self.sigma_inv_0.float(),
                        embeddings.cpu().float(),
                    )[:, 0]
                    .cpu()
                    .detach()
                    .numpy()
                )
            else:
                dists_0 = (
                    mahalanobis_distance_with_known_centroids_sigma_inv(
                        self.centroid_0.cuda().float(),
                        None,
                        self.sigma_inv_0.cuda().float(),
                        embeddings.float(),
                    )[:, 0]
                    .cpu()
                    .detach()
                    .numpy()
                )
        elif self.device == "cuda" and self.storage_device == "cuda":
            dists_0 = (
                    mahalanobis_distance_with_known_centroids_sigma_inv(
                        self.centroid_0.float(),
                        None,
                        self.sigma_inv_0.float(),
                        embeddings.float(),
                    )[:, 0]
                    .cpu()
                    .detach()
                    .numpy()
                )
        else:
            raise NotImplementedError

        # compute original MD

        md = self.MD(stats)

        # RMD calculation

        dists = md - dists_0
        if self.max < dists.max():
            self.max = dists.max()
            if self.parameters_path is not None:
                save_array(self.max, f"{self.full_path}/max_0.npy")
        if self.min > dists.min():
            self.min = dists.min()
            if self.parameters_path is not None:
                save_array(self.min, f"{self.full_path}/min_0.npy")

        if self.normalize:
            dists = np.clip(
                (self.max - dists) / (self.max - self.min), a_min=0, a_max=1
            )

        return dists
