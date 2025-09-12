import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from transformers import set_seed, AutoConfig

from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline

from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator
from .common import cross_val_hp, TrainerMLP
from .energy import fft_energy_ratio, layerwise_total_energy
from .dimension_reduction import layerwise_norms
from .curvatures import angle_curvature, curvature_menger, curvature_arc_chord, curvature_second_derivative
from .transitions import layerwise_angles, layerwise_sigmas


class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer for aggregating token embeddings.

    Used in the SHEEPS method from https://aclanthology.org/2024.findings-acl.260.pdf
    """

    def __init__(self, embedding_size):
        super().__init__()
        self.attn = nn.Linear(embedding_size, 1)

    def forward(self, x, mask=None):
        attn_logits = self.attn(x)
        if mask is not None:
            attn_logits[mask] = -float("inf")
        attn_weights = torch.softmax(attn_logits, dim=1)
        return (x * attn_weights).sum(dim=1)


class MLP(nn.Module):
    """
    MLP classifier with attention pooling for sequence-level uncertainty estimation.
    """

    def __init__(self, n_features: int = 4096):
        super().__init__()
        self.pooling = AttentionPooling(n_features)
        self.output = nn.Linear(n_features, 2)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x, mask, eval: bool = False, regression: bool = False):
        x = self.pooling(x, mask)
        x = self.output(x)
        if eval:
            return self.activation(x)[:, 1]
        return x


class LayerSheeps(Estimator):
    """
    LayerSHEEPS: Sequence-level uncertainty estimation using attention-pooled token embeddings
    from a single layer, as described in https://aclanthology.org/2024.findings-acl.260.pdf

    This estimator fits an MLP with attention pooling on token embeddings from a specific layer,
    using a cross-entropy loss to predict hallucination labels.
    """

    def __init__(
        self,
        embeddings_type: str = "decoder",
        layer: int = -1,
        device: str = "cuda",
        metric_thr: float = 0.3,
    ):
        super().__init__(
            ["token_embeddings", "train_token_embeddings", "train_metrics"], "sequence"
        )
        self.layer = layer
        self.layer_name = "" if self.layer == -1 else f"_{self.layer}"
        self.embeddings_type = embeddings_type
        self.device = device
        self.metric_thr = metric_thr
        self.is_fitted = False
        self.params = {
            "n_epochs": [5, 10, 20],
            "batch_size": [32, 64],
            "lr": [1e-1, 1e-2, 1e-3],
            "n_features": [4096],
        }

        self.loss_fn = nn.CrossEntropyLoss()
        self.model_init = lambda param: TrainerMLP(
            n_epochs=param[0],
            batch_size=param[1],
            lr=param[2],
            n_features=param[3],
            device=self.device,
            loss_fn=self.loss_fn,
            model=MLP,
        )

    def __str__(self):
        return f"LayerSheeps_{self.embeddings_type}{self.layer_name}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute sequence-level uncertainty scores using attention-pooled token embeddings
        from a single layer, as in SHEEPS (https://aclanthology.org/2024.findings-acl.260.pdf).

        Args:
            stats (Dict[str, np.ndarray]): Dictionary containing token embeddings and metrics.

        Returns:
            np.ndarray: Array of uncertainty scores for each sequence.
        """
        if not self.is_fitted:
            train_metrics = stats["train_metrics"]
            train_greedy_tokens = stats["train_greedy_tokens"]

            train_metrics = (train_metrics < self.metric_thr).astype(int)
            train_embeddings = stats[
                f"train_token_embeddings_{self.embeddings_type}{self.layer_name}"
            ]
            k = 0
            aggregated_embeddings, lens = [], []
            for tokens in train_greedy_tokens:
                aggregated_embeddings.append(
                    torch.tensor(np.array(train_embeddings[k : k + len(tokens)]))
                )
                lens.append(len(tokens))
                k += len(tokens)
            aggregated_embeddings = pad_sequence(
                aggregated_embeddings, batch_first=True, padding_value=0
            )
            attention_mask = np.zeros(
                (aggregated_embeddings.shape[0], aggregated_embeddings.shape[1])
            )

            for i, l in enumerate(lens):
                attention_mask[i, l:] = 1
            attention_mask = torch.tensor(attention_mask).int()

            self.params["n_features"] = [aggregated_embeddings.shape[-1]]
            best_params = cross_val_hp(
                aggregated_embeddings,
                train_metrics,
                self.model_init,
                self.params,
                regression=False,
                mask=attention_mask,
                estimator_name=self.__str__(),
            )
            self.ue_predictor = self.model_init(best_params)

            self.ue_predictor.fit(aggregated_embeddings, train_metrics, attention_mask)
            self.is_fitted = True
        # Inference
        embeddings = stats[f"token_embeddings_{self.embeddings_type}{self.layer_name}"]
        greedy_tokens = stats["greedy_tokens"]
        k = 0
        aggregated_embeddings, lens = [], []
        for tokens in greedy_tokens:
            aggregated_embeddings.append(
                torch.tensor(np.array(embeddings[k : k + len(tokens)]))
            )
            lens.append(len(tokens))
            k += len(tokens)
        aggregated_embeddings = pad_sequence(
            aggregated_embeddings, batch_first=True, padding_value=0
        )
        attention_mask = np.zeros(
            (aggregated_embeddings.shape[0], aggregated_embeddings.shape[1])
        )
        for i, l in enumerate(lens):
            attention_mask[i, l:] = 1

        attention_mask = torch.tensor(attention_mask).int()
        ue = self.ue_predictor.predict(aggregated_embeddings, attention_mask)

        return ue


class Sheeps(Estimator):
    """
    Implements the SHEEPS method from "Do Androids Know They’re Only Dreaming of Electric Sheep?"
    (https://aclanthology.org/2024.findings-acl.260.pdf)

    This estimator fits a meta-classifier (logistic regression) on the outputs of LayerSHEEPS
    from multiple layers, using attention-pooled token embeddings, to predict hallucination
    or uncertainty at the sequence level.
    """

    def __init__(
        self,
        embeddings_type: str = "decoder",
        layers: List[int] = None,
        device: str = "cuda",
        metric_thr: float = 0.3,
        dev_size: float = 0.5,
        model_name: str = None,
        with_dynamic: bool = False, 
        cache_dir: str = None,
    ):
        self.model_name = model_name
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        if layers is None:
            self.layers = (
                list(range(self.model_config.num_hidden_layers))
                if hasattr(self.model_config, "num_hidden_layers")
                else list(range(self.model_config.text_config.num_hidden_layers))
            )
        else:
            self.layers = layers
        self.embeddings_type = embeddings_type
        self.device = device
        self.metric_thr = metric_thr
        self.dev_size = dev_size
        self.is_fitted = False
        self.layersheeps = [
            LayerSheeps(
                embeddings_type=embeddings_type,
                layer=layer,
                device=device,
                metric_thr=metric_thr,
            )
            for layer in self.layers
        ]
        self.with_dynamic = with_dynamic
        if self.with_dynamic:
            self.dynamic_functions = {
                "angle_curvature": lambda x: angle_curvature(x, degrees=False),
                "curvature_menger": lambda x: curvature_menger(x),
                "curvature_arc_chord": lambda x: curvature_arc_chord(x),
                "curvature_second_derivative": lambda x: curvature_second_derivative(x),
                "angle_curvatulayerwise_anglesre": lambda x: layerwise_angles(x),
                "layerwise_sigmas": lambda x: layerwise_sigmas(x),
                #### energy
                "layerwise_norms": lambda x: layerwise_norms(x, norm=2),
                "layerwise_total_energy": lambda x: layerwise_total_energy(x),
                "fft_energy_ratio_k5": lambda x: fft_energy_ratio(x, top_k=5),
            }

        super().__init__(
            ["token_embeddings", "train_token_embeddings", "train_metrics"], "sequence"
        )
        if self.with_dynamic:
            self.ue_predictor = Pipeline([
                ('scaler', PowerTransformer()),
                ('logreg', LogisticRegressionCV(max_iter=1000, tol=1e-4, cv=10))
            ])
        else:
            self.ue_predictor = LogisticRegressionCV()
            
        self.cache_dir = cache_dir
            

    def __str__(self):
        if self.with_dynamic:
            return f"DynamicSheeps_{self.embeddings_type}"
        return f"Sheeps_{self.embeddings_type}"

    def __call__(self, stats: Dict[str, np.ndarray], folder_name: str = "", eval_idx: int = None) -> np.ndarray:
        if not self.is_fitted:
            set_seed(42)
            train_metrics_raw = stats["train_metrics"]
            train_greedy_tokens = stats["train_greedy_tokens"]

            train_metrics = (train_metrics_raw < self.metric_thr).astype(int)
            train_idx, dev_idx = train_test_split(
                np.arange(len(train_greedy_tokens)),
                test_size=self.dev_size,
                random_state=42,
            )            
            train_sheeps = []
            for layer in self.layers:
                # Prepare stats for this layer
                layer_name = "" if layer == -1 else f"_{layer}"
                train_embeddings = stats[
                    f"train_token_embeddings_{self.embeddings_type}{layer_name}"
                ]
                k = 0
                aggregated_embeddings = []
                for tokens in train_greedy_tokens:
                    aggregated_embeddings.append(train_embeddings[k : k + len(tokens)])
                    k += len(tokens)
                train_stats = {
                    "train_greedy_tokens": [train_greedy_tokens[k] for k in train_idx],
                    "greedy_tokens": [train_greedy_tokens[k] for k in dev_idx],
                    "train_metrics": train_metrics_raw[train_idx],
                    f"train_token_embeddings_{self.embeddings_type}{layer_name}": [
                        emb for k in train_idx for emb in aggregated_embeddings[k]
                    ],
                    f"token_embeddings_{self.embeddings_type}{layer_name}": [
                        emb for k in dev_idx for emb in aggregated_embeddings[k]
                    ],
                }
                score = self.layersheeps[layer](train_stats).reshape(-1)
                train_sheeps.append(score)
                
            if self.with_dynamic:
                final_embeddings = []
                for layer in self.layers:
                    layer_name = "" if layer == -1 else f"_{layer}"
                    train_embeddings = stats[
                        f"train_token_embeddings_{self.embeddings_type}{layer_name}"
                    ]
                    k = 0
                    layer_embeddings = []
                    for tokens in train_greedy_tokens:
                        layer_embeddings.append(train_embeddings[k + len(tokens) - 1 : k + len(tokens)][0]) ## last token
                        k += len(tokens)
                    layer_embeddings = np.array(layer_embeddings) # [B, D]
                    final_embeddings.append(layer_embeddings)
                final_embeddings = np.array(final_embeddings) # [L, B, D]
                final_embeddings = torch.tensor(final_embeddings.transpose(1, 0, 2)[dev_idx])
                
                train_dynamics = []
                for func_name in self.dynamic_functions.keys():
                    features = self.dynamic_functions[func_name](final_embeddings)
                    if self.cache_dir:
                        np.save(f'{self.cache_dir}/{folder_name}/train_{func_name}.npy', features)
                    for feature in features.T:
                        train_dynamics.append(feature)
                train_dynamics = np.array(train_dynamics).T
                if self.cache_dir:
                    np.save(f'{self.cache_dir}/{folder_name}/train_dynamics.npy', train_dynamics)
                self.pca = KernelPCA(n_components=10, kernel='rbf')
                train_pca_dynamics = self.pca.fit_transform(train_dynamics)
                
            train_sheeps = np.array(train_sheeps).T
            if self.with_dynamic:
                train_sheeps = np.hstack([train_sheeps, train_pca_dynamics])
            if self.cache_dir:
                np.save(f'{self.cache_dir}/{folder_name}/train_features.npy', train_sheeps)
                np.save(f'{self.cache_dir}/{folder_name}/train_targets.npy', train_metrics[dev_idx])
            self.ue_predictor.fit(train_sheeps, train_metrics[dev_idx])
            self.is_fitted = True

        eval_scores = []
        for layer in self.layers:
            score = self.layersheeps[layer](stats).reshape(-1)
            eval_scores.append(score)
        
        greedy_tokens = stats["greedy_tokens"]
        if self.with_dynamic:
            final_embeddings = []
            for layer in self.layers:
                layer_name = "" if layer == -1 else f"_{layer}"
                embeddings = stats[
                    f"token_embeddings_{self.embeddings_type}{layer_name}"
                ]
                k = 0
                layer_embeddings = []
                for tokens in greedy_tokens:
                    layer_embeddings.append(embeddings[k + len(tokens) - 1 : k + len(tokens)][0])
                    k += len(tokens)
                layer_embeddings = np.array(layer_embeddings) # [B, D]
                final_embeddings.append(layer_embeddings)
            final_embeddings = np.array(final_embeddings) # [L, B, D]
            final_embeddings = torch.tensor(final_embeddings.transpose(1, 0, 2))
                       
            eval_dynamics = []     
            for func_name in self.dynamic_functions.keys():
                features = self.dynamic_functions[func_name](final_embeddings)
                if self.cache_dir:
                    np.save(f'{self.cache_dir}/{folder_name}/eval_{func_name}_{eval_idx}.npy', features)
                for feature in features.T:
                    eval_dynamics.append(feature)
            eval_dynamics = np.array(eval_dynamics).T
            if self.cache_dir:
                np.save(f'{self.cache_dir}/{folder_name}/eval_dynamics_{eval_idx}.npy', eval_dynamics)
            eval_pca_dynamics = self.pca.transform(eval_dynamics)
                    
        eval_scores = np.array(eval_scores).T
        if self.with_dynamic:
            eval_scores = np.hstack([eval_scores, eval_pca_dynamics])
        eval_scores[np.isnan(eval_scores)] = 0
        if self.cache_dir:
            np.save(f'{self.cache_dir}/{folder_name}/eval_features_{eval_idx}.npy', eval_scores)
        ue = self.ue_predictor.predict_proba(eval_scores)[:, 1]
        return ue
