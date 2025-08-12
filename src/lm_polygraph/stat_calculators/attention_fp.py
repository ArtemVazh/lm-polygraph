import gc
import torch
import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel
import traceback


class AttentionForwardPassCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * attention masks across the model (if applicable)
    """

    def __init__(self):
        super().__init__(
            [                
                "forwardpass_attention_weights",
            ],
            ["greedy_tokens"],
        )
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
        device = model.device()
        batch = {k: v.to(device) for k, v in batch.items()}
        
        cut_sequences = dependencies["greedy_tokens"]
        
        forwardpass_attention_weights = []

        for i in range(len(texts)):
            input_ids = torch.cat([batch['input_ids'].to(device), torch.tensor([cut_sequences[i]]).to(device)], axis=1)
            with torch.inference_mode():
                forwardpass_attentions = model.model(input_ids[:, -2048:], output_attentions=True, use_cache=False).attentions
                forwardpass_attentions_cpu = tuple(attention.to("cpu") for attention in forwardpass_attentions)
                forwardpass_attentions_np = torch.cat(forwardpass_attentions_cpu).float().numpy()
            forwardpass_attention_weights.append(forwardpass_attentions_np)
            del forwardpass_attentions, forwardpass_attentions_cpu
            gc.collect()
            torch.cuda.empty_cache()
            
        forwardpass_attention_weights = np.array(forwardpass_attention_weights)

        result_dict = {
            "forwardpass_attention_weights": forwardpass_attention_weights
        }

        return result_dict
