import numpy as np
import torch
import os
import joblib
from torch import nn
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from transformers import set_seed, AutoConfig
from pathlib import Path

from typing import Dict, List

from lm_polygraph.estimators.estimator import Estimator
from .common import TrainerMLP
from .sheeps import MLP, LayerSheeps


class Intra(Estimator):
    """
    Implements the INTRA (Intrinsic Truthfulness Assessment) method for uncertainty estimation.

    INTRA is a two-stage approach that assesses truthfulness by training layer-specific classifiers
    and then aggregating their predictions through quantile normalization and ridge regression.

    Method Overview:
    ----------------
    1. **Layer Selection**: Flexible selection of transformer layers (middle, first/last quarters,
       even/odd layers, uniform sampling, fractional selection, or exact layer specification)
    2. **Stage 1 - Layer Models**: Train MLP classifiers with attention pooling on selected layers
       using the SHEEPS architecture to extract layer-wise uncertainty signals
    3. **Stage 2 - Aggregation**: Apply quantile normalization to layer probabilities and train
       a ridge regression meta-classifier for final truthfulness scoring

    Layer Selection Strategies:
    -------------------------
    - **middle** (default): Middle third of layers (recommended in original paper)
    - **first_half/second_half**: First/last 50% of layers
    - **first_quarter/last_quarter**: First/last 25% of layers
    - **middle_half**: Middle 50% (layers 25% to 75%)
    - **even_layers/odd_layers**: Even/odd-numbered layers only
    - **uniform**: Evenly spaced layers with n_layers parameter
    - **fraction**: Fraction of layers with layer_fraction parameter
    - **exact**: Direct layer list specification via layers parameter

    Based on the INTRA methodology for intrinsic truthfulness assessment using
    layer-specific representations and robust aggregation.

    References:
    -----------
    Original implementation follows the INTRA two-stage training approach:
    1. Layer-specific classifier training
    2. Quantile normalization + ridge regression aggregation

    Usage:
    ------
    ```python
    # Default: middle layers selection
    intra = Intra(model_name='gpt2')

    # Specific layer selection
    intra = Intra(model_name='gpt2', layers=[0, 4, 8])

    # Even layers only
    intra = Intra(model_name='gpt2', layer_selection='even_layers')

    # 25% of layers evenly spaced
    intra = Intra(model_name='gpt2', layer_selection='fraction', layer_fraction=0.25)

    # Train and predict
    scores = intra(train_stats)  # Training phase
    predictions = intra(test_stats)  # Inference phase

    # Checkpoint functionality
    checkpoint_path = intra.save_checkpoint('./intra_checkpoint')
    loaded_intra = Intra.load_checkpoint('./intra_checkpoint')
    ```
    """

    def __init__(
        self,
        embeddings_type: str = "decoder",
        layers: List[int] = None,
        layer_selection: str = "middle",
        n_layers: int = None,
        layer_fraction: float = None,
        device: str = "cuda",
        metric_thr: float = 0.3,
        dev_size: float = 0.5,
        model_name: str = None,
        cache_dir: str = None,
    ):
        self.model_name = model_name
        self.model_config = AutoConfig.from_pretrained(self.model_name)

        # Get total number of layers
        num_layers = (
            self.model_config.num_hidden_layers
            if hasattr(self.model_config, "num_hidden_layers")
            else self.model_config.text_config.num_hidden_layers
        )
        # Layer selection strategy
        if layers is not None:
            # Use exact layer numbers provided
            self.layers = layers
        else:
            # Apply layer selection strategy
            if layer_selection == "middle":
                # Middle third (default from paper)
                start_layer = num_layers // 3
                end_layer = 2 * num_layers // 3
                self.layers = list(range(start_layer, end_layer))
            elif layer_selection == "first_half":
                # First half of layers
                end_layer = num_layers // 2
                self.layers = list(range(end_layer))
            elif layer_selection == "second_half":
                # Second half of layers
                start_layer = num_layers // 2
                self.layers = list(range(start_layer, num_layers))
            elif layer_selection == "first_quarter":
                # First quarter
                end_layer = num_layers // 4
                self.layers = list(range(end_layer))
            elif layer_selection == "last_quarter":
                # Last quarter
                start_layer = 3 * num_layers // 4
                self.layers = list(range(start_layer, num_layers))
            elif layer_selection == "middle_half":
                # Middle half (layers 25% to 75%)
                start_layer = num_layers // 4
                end_layer = 3 * num_layers // 4
                self.layers = list(range(start_layer, end_layer))
            elif layer_selection == "even_layers":
                # Even-numbered layers only
                self.layers = [i for i in range(num_layers) if i % 2 == 0]
            elif layer_selection == "odd_layers":
                # Odd-numbered layers only
                self.layers = [i for i in range(num_layers) if i % 2 == 1]
            elif layer_selection == "uniform":
                # Uniformly sample layers
                if n_layers is None:
                    n_layers = max(1, num_layers // 4)  # Default to 25% of layers
                indices = np.linspace(0, num_layers - 1, n_layers, dtype=int)
                self.layers = indices.tolist()
            elif layer_selection == "fraction":
                # Use a fraction of layers (evenly spaced)
                if layer_fraction is None:
                    layer_fraction = 0.33  # Default to 33% of layers
                n_layers = max(1, int(num_layers * layer_fraction))
                indices = np.linspace(0, num_layers - 1, n_layers, dtype=int)
                self.layers = indices.tolist()
            else:
                raise ValueError(f"Unknown layer_selection strategy: {layer_selection}")

        print(f"Selected {len(self.layers)} layers using '{layer_selection}' strategy: {self.layers}")

        # Store layer selection parameters for checkpointing
        self.layer_selection = layer_selection
        self.n_layers = n_layers
        self.layer_fraction = layer_fraction

        self.embeddings_type = embeddings_type
        self.device = device
        self.metric_thr = metric_thr
        self.dev_size = dev_size
        self.cache_dir = cache_dir

        # Per-layer MLP models (reuse SHEEPS architecture)
        self.layer_models = {}

        # Aggregation components
        self.quantile_transformer = QuantileTransformer()
        self.ridge_regressor = RidgeCV(alphas=(1e+2, 1e+3, 1e+4, 1e+5), cv=5)  # Reduced CV folds for smaller datasets

        # Training parameters (same as SHEEPS)
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

        self.is_fitted = False

        super().__init__(
            ["token_embeddings", "train_token_embeddings", "train_metrics"], "sequence"
        )

    def __str__(self):
        return f"Intra_{self.layer_selection}_{self.embeddings_type}"

    def _create_layer_model(self, layer: int):
        """Create a SHEEPS-style layer model for the specified layer."""
        return LayerSheeps(
            embeddings_type=self.embeddings_type,
            layer=layer,
            device=self.device,
            metric_thr=self.metric_thr,
        )

    def _get_layer_embeddings(self, stats: Dict[str, np.ndarray], layer: int, is_train: bool = True):
        """Extract embeddings for a specific layer."""
        layer_name = "" if layer == -1 else f"_{layer}"
        prefix = "train_token_embeddings_" if is_train else "token_embeddings_"
        key = f"{prefix}{self.embeddings_type}{layer_name}"
        return stats[key]

    def _prepare_layer_stats(self, stats: Dict[str, np.ndarray], layer: int, train_idx=None, dev_idx=None):
        """Prepare statistics dictionary for a single layer."""
        layer_name = "" if layer == -1 else f"_{layer}"

        if train_idx is not None and dev_idx is not None:
            # Training phase with train/dev split
            train_greedy_tokens = stats["train_greedy_tokens"]
            train_metrics_raw = stats["train_metrics"]
            train_metrics = (train_metrics_raw < self.metric_thr).astype(int)

            # Extract embeddings for this layer
            train_embeddings = self._get_layer_embeddings(stats, layer, is_train=True)

            # Prepare aggregated embeddings
            k = 0
            aggregated_embeddings = []
            for tokens in train_greedy_tokens:
                aggregated_embeddings.append(train_embeddings[k : k + len(tokens)])
                k += len(tokens)

            # Create layer-specific stats
            layer_stats = {
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

            return layer_stats, train_metrics[dev_idx]
        else:
            # Inference phase
            return stats

    def _train_layer_models(self, stats: Dict[str, np.ndarray], train_idx, dev_idx):
        """Stage 1: Train MLP models for each layer."""
        print(f"Training layer models for layers: {self.layers}")

        layer_probabilities = []

        for layer in self.layers:
            print(f"  Training layer {layer}...")

            # Create layer model
            layer_model = self._create_layer_model(layer)

            # Prepare layer-specific statistics
            layer_stats, dev_labels = self._prepare_layer_stats(stats, layer, train_idx, dev_idx)

            # Train the layer model
            layer_model(layer_stats)

            # Get probabilities on development set
            layer_probs = layer_model(layer_stats)
            layer_probabilities.append(layer_probs)

            # Store the trained model
            self.layer_models[layer] = layer_model

            print(f"    Layer {layer} trained successfully")

        # Stack probabilities from all layers
        return np.column_stack(layer_probabilities), dev_labels

    def _train_aggregation_regressor(self, layer_probabilities: np.ndarray, dev_labels: np.ndarray):
        """Stage 2: Train quantile normalization and ridge regression."""
        print("Training aggregation regressor...")

        # Apply quantile normalization
        print("  Applying quantile normalization...")
        X_normalized = self.quantile_transformer.fit_transform(layer_probabilities)

        # Train ridge regression
        print("  Training ridge regression...")
        self.ridge_regressor.fit(X_normalized, dev_labels)

        print("  Aggregation regressor trained successfully")

    def __call__(self, stats: Dict[str, np.ndarray], folder_name: str = "", eval_idx: int = None, batch_size: int = 32) -> np.ndarray:
        """
        Compute intrinsic truthfulness scores using the INTRA method.

        Args:
            stats: Dictionary containing embeddings and metrics
            folder_name: Cache folder name
            eval_idx: Evaluation index for caching
            batch_size: Batch size for processing

        Returns:
            np.ndarray: Truthfulness scores for each sequence
        """
        if not self.is_fitted:
            set_seed(42)
            print("=== INTRA Training Phase ===")

            train_greedy_tokens = stats["train_greedy_tokens"]
            train_metrics_raw = stats["train_metrics"]
            train_metrics = (train_metrics_raw < self.metric_thr).astype(int)

            # Split data for two-stage training
            train_idx, dev_idx = train_test_split(
                np.arange(len(train_greedy_tokens)),
                test_size=self.dev_size,
                random_state=42,
            )

            # Stage 1: Train layer models
            print("\n--- Stage 1: Training Layer Models ---")
            layer_probabilities, dev_labels = self._train_layer_models(stats, train_idx, dev_idx)

            # Stage 2: Train aggregation regressor
            print("\n--- Stage 2: Training Aggregation Regressor ---")
            self._train_aggregation_regressor(layer_probabilities, dev_labels)

            # Cache training results
            if self.cache_dir:
                os.makedirs(f'{self.cache_dir}/{folder_name}', exist_ok=True)
                np.save(f'{self.cache_dir}/{folder_name}/intra_layer_probs.npy', layer_probabilities)
                np.save(f'{self.cache_dir}/{folder_name}/intra_train_labels.npy', dev_labels)

            self.is_fitted = True
            print("INTRA training completed successfully")

        # Inference Phase
        print("=== INTRA Inference Phase ===")

        # Get probabilities from all trained layer models
        layer_probabilities = []
        for layer in self.layers:
            print(f"  Computing probabilities for layer {layer}...")

            # Prepare layer stats for inference
            layer_stats = self._prepare_layer_stats(stats, layer)

            # Get probabilities from this layer
            layer_probs = self.layer_models[layer](layer_stats)
            layer_probabilities.append(layer_probs)

        # Stack layer probabilities
        X_features = np.column_stack(layer_probabilities)

        # Apply quantile normalization
        X_normalized = self.quantile_transformer.transform(X_features)

        # Get final truthfulness scores
        truthfulness_scores = self.ridge_regressor.predict(X_normalized)

        # Cache inference results (simplified without eval_idx dependency)
        if self.cache_dir:
            os.makedirs(f'{self.cache_dir}/{folder_name}', exist_ok=True)
            np.save(f'{self.cache_dir}/{folder_name}/intra_features.npy', X_normalized)

        print(f"INTRA inference completed for {len(truthfulness_scores)} sequences")
        return truthfulness_scores

    def save_checkpoint(self, checkpoint_dir: str):
        """
        Save INTRA model checkpoint including layer models and aggregation components.

        Args:
            checkpoint_dir: Directory to save checkpoint

        Returns:
            str: Path to saved checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Saving INTRA checkpoint to: {checkpoint_dir}")

        # Save main configuration
        config = {
            'model_name': self.model_name,
            'layers': self.layers,
            'embeddings_type': self.embeddings_type,
            'device': str(self.device).split(':')[0],
            'metric_thr': self.metric_thr,
            'dev_size': self.dev_size,
            'cache_dir': str(self.cache_dir) if self.cache_dir else None,
            'is_fitted': self.is_fitted,
            'layer_selection': getattr(self, 'layer_selection', 'middle'),
            'n_layers': getattr(self, 'n_layers', None),
            'layer_fraction': getattr(self, 'layer_fraction', None)
        }
        joblib.dump(config, Path(checkpoint_dir) / 'main_config.pkl')
        print("Main configuration saved")

        # Save layer models (extract critical components to avoid lambda issues)
        for layer_idx, layer in enumerate(self.layers):
            layer_model = self.layer_models[layer]

            # Create saveable layer state
            layer_state = {
                'layer': layer,
                'embeddings_type': layer_model.embeddings_type,
                'device': layer_model.device,
                'metric_thr': layer_model.metric_thr,
                'layer_name': layer_model.layer_name,
                'is_fitted': layer_model.is_fitted,
                'params': layer_model.params,
            }

            # Save the model weights if fitted
            if layer_model.is_fitted and hasattr(layer_model, 'ue_predictor'):
                # Save the MLP model state dict
                mlp_model = layer_model.ue_predictor.model
                state_dict = {k: v.cpu() for k, v in mlp_model.state_dict().items()}
                torch.save(state_dict, Path(checkpoint_dir) / f'layer_{layer_idx}_weights.pt')
                layer_state['n_features'] = mlp_model.pooling.attn.in_features

            joblib.dump(layer_state, Path(checkpoint_dir) / f'layer_{layer_idx}_state.pkl')
            print(f"Saved layer {layer} model state")

        # Save aggregation components
        joblib.dump(self.quantile_transformer, Path(checkpoint_dir) / 'quantile_transformer.pkl')
        joblib.dump(self.ridge_regressor, Path(checkpoint_dir) / 'ridge_regressor.pkl')
        print("Aggregation components saved")

        # Final summary
        checkpoint_size = sum(f.stat().st_size for f in Path(checkpoint_dir).glob('*') if f.is_file())
        abs_path = os.path.abspath(checkpoint_dir)
        print(f"INTRA checkpoint save complete. Size: {checkpoint_size/1e6:.2f} MB")
        print(f"Location: {abs_path}")

        return abs_path

    @classmethod
    def load_checkpoint(cls, checkpoint_dir: str, target_device=None):
        """
        Load INTRA model from checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoint files
            target_device: Target device for loaded model

        Returns:
            Intra: Loaded model instance
        """
        config = joblib.load(Path(checkpoint_dir) / 'main_config.pkl')
        device = target_device or config['device']
        print(f"Loading INTRA checkpoint to {device}...")

        # Create fresh instance with exact layer configuration
        new_model = cls(
            model_name=config['model_name'],
            layers=config['layers'],  # Use exact layers from checkpoint
            embeddings_type=config['embeddings_type'],
            device=device,
            metric_thr=config['metric_thr'],
            dev_size=config['dev_size'],
            cache_dir=config['cache_dir']
        )

        # Restore layer selection parameters for compatibility
        new_model.layer_selection = config.get('layer_selection', 'middle')
        new_model.n_layers = config.get('n_layers', None)
        new_model.layer_fraction = config.get('layer_fraction', None)
        new_model.is_fitted = True

        # Load layer models
        for layer_idx, layer in enumerate(new_model.layers):
            state_path = Path(checkpoint_dir) / f'layer_{layer_idx}_state.pkl'
            weights_path = Path(checkpoint_dir) / f'layer_{layer_idx}_weights.pt'

            if state_path.exists():
                # Load layer state
                layer_state = joblib.load(state_path)

                # Create fresh LayerSheeps model
                from .sheeps import LayerSheeps
                layer_model = LayerSheeps(
                    embeddings_type=layer_state['embeddings_type'],
                    layer=layer_state['layer'],
                    device=device,
                    metric_thr=layer_state['metric_thr']
                )
                layer_model.is_fitted = layer_state['is_fitted']
                layer_model.params = layer_state['params']

                # Load model weights if available
                if weights_path.exists():
                    from .sheeps import MLP
                    mlp_model = MLP(n_features=layer_state['n_features'])
                    state_dict = torch.load(weights_path, map_location=device)
                    mlp_model.load_state_dict(state_dict)
                    mlp_model.to(device)
                    mlp_model.eval()

                    # Create inference wrapper (same as in SHEEPS)
                    class IntraWrapper:
                        def __init__(self, model, device):
                            self.model = model
                            self.device = device

                        def predict(self, x, mask):
                            """Replicate LayerSheeps inference behavior"""
                            self.model.eval()
                            with torch.no_grad():
                                x = x.to(self.device)
                                mask = mask.to(self.device)

                                attn_logits = self.model.pooling.attn(x)
                                attn_logits = attn_logits.masked_fill(mask.unsqueeze(-1).bool(), -1e9)
                                attn_weights = torch.softmax(attn_logits, dim=1)
                                pooled = (x * attn_weights).sum(dim=1)
                                logits = self.model.output(pooled)
                                probs = torch.softmax(logits, dim=1)

                                return probs[:, 1].cpu().numpy()

                    layer_model.ue_predictor = IntraWrapper(mlp_model, device)
                    print(f"Loaded layer {layer} model with weights")
                else:
                    print(f"Warning: Layer {layer} weights not found")

                new_model.layer_models[layer] = layer_model
                print(f"Loaded layer {layer} model")
            else:
                print(f"Warning: Layer {layer} state not found")

        # Load aggregation components
        new_model.quantile_transformer = joblib.load(Path(checkpoint_dir) / 'quantile_transformer.pkl')
        new_model.ridge_regressor = joblib.load(Path(checkpoint_dir) / 'ridge_regressor.pkl')
        print("Aggregation components loaded")

        print("INTRA checkpoint loaded successfully")
        return new_model
