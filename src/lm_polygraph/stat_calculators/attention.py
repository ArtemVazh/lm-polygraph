import torch
import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel


class AttentionCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * attention masks across the model (if applicable)
    """

    def __init__(self, n_top_attention: int = 10):
        super().__init__(
            [
                "attention_weights",
                
                "lookback_ratios",
                "train_lookback_ratios",
                
                "attention_features",
                "train_attention_features",  
                                
                "attention_max_features_values",
                "train_attention_max_features_values",
                
                "attention_max_features_token",
                "train_attention_max_features_token",
                
                "attention_features_values",
                "train_attention_features_values",
            ],
            ["attentions_all"],
        )
        self.n_top_attention = n_top_attention

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the statistics of probabilities at each token position in the generation.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'attention' (List[List[np.array]]): attention maps at each token, if applicable to the model,
        """
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        
        attentions = dependencies["attentions_all"]
        cut_sequences = dependencies["greedy_tokens"]
        
        attn_features = []
        attention_weights = []
        lookback_ratios = []
        attn_features_max_tokens = []
        attn_features_max_values = []
        attn_features_values = []
        for i in range(len(texts)):
            c = len(cut_sequences[i])
            attn_mask = np.zeros(
                shape=(
                    model.model.config.num_attention_heads
                    * model.model.config.num_hidden_layers,
                    c,
                    c,
                )
            )
            
            for j in range(1, c):
                start_idx = -j
                end_idx = attentions[j][0].shape[-1]
                if attentions[0][0].shape[-1] == attentions[j][0].shape[-1]:
                    # for gemma-2
                    start_idx = attentions[0][0].shape[-2] # prompt len
                    end_idx = start_idx + j
                attn_mask[:, j, :j] = (
                    torch.vstack(
                        [
                            attentions[j][layer][0][head][0][start_idx:end_idx]
                            for layer in range(len(attentions[j]))
                            for head in range(len(attentions[j][layer][0]))
                        ]
                    )
                    .cpu()
                    .numpy()
                )
            for j in range(c):
                lookback_ratios_token = []
                
                start_idx = -j
                end_idx = attentions[j][0].shape[-1]
                if attentions[0][0].shape[-1] == attentions[j][0].shape[-1]:
                    # for gemma-2
                    start_idx = attentions[0][0].shape[-2] # prompt len
                    end_idx = start_idx + j
                    
                for layer in range(len(attentions[j])):
                    for head in range(len(attentions[j][layer][0])):
                        if j == 0:
                            attention_on_new = 0
                            attention_on_context = attentions[j][layer][0][head][0].mean().item()
                        else:
                            attention_on_new = attentions[j][layer][0][head][0][start_idx:end_idx].mean().item()
                            attention_on_context = attentions[j][layer][0][head][0][:start_idx].mean().item()
                        lookback_ratio = attention_on_context / (attention_on_new + attention_on_context)
                        lookback_ratios_token.append(lookback_ratio)
                lookback_ratios.append(lookback_ratios_token)

            max_attention = attn_mask.max(0)
            current_top_n = min(self.n_top_attention, max_attention.shape[1])
            topk = torch.topk(torch.tensor(max_attention), k=current_top_n, dim=1)
            attn_features_max_values_s = []
            attn_features_max_tokens_s = []
            attn_features_values_s = []

            attention_weights.append(max_attention)
            for j in range(1, c):
                attn_features.append(attn_mask[:, j, j - 1])
                attn_features_max_values_i = []
                attn_features_max_tokens_i = []
                attn_features_values_i = []
                for k in range(min(j, current_top_n)):
                    attn_features_max_values_i.append(attn_mask[:, j, topk.indices[j][k].item()])                    
                    attn_features_max_tokens_i.append(topk.indices[j][k].item())
                    attn_features_values_i.append(attn_mask[:, j, j - k - 1]) 

                attn_features_max_values_s.append(attn_features_max_values_i)
                attn_features_max_tokens_s.append(attn_features_max_tokens_i)
                attn_features_values_s.append(attn_features_values_i)

        attn_features_max_values.append(attn_features_max_values_s)
        attn_features_max_tokens.append(attn_features_max_tokens_s)
        attn_features_values.append(attn_features_values_s)
        
        attention_weights = np.array(attention_weights)
        attn_features = np.array(attn_features)
        lookback_ratios = np.array(lookback_ratios)

        result_dict = {
            "attention_weights": attention_weights,
            "lookback_ratios": lookback_ratios,
            
            "attention_features": attn_features,
            "attention_max_features_token": attn_features_max_tokens,
            # "attention_max_features_values": attn_features_max_values,
            "attention_features_values": attn_features_values,
        }

        return result_dict
