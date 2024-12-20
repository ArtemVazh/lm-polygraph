import torch
import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel


class BlackboxGreedyTextsCalculator(StatCalculator):
    """
    Calculates generation texts for Blackbox model (lm_polygraph.BlackboxModel).
    """

    def __init__(self):
        super().__init__(["greedy_texts", "train_greedy_texts", "train_target_texts", "background_train_greedy_texts", "background_train_target_texts"], [])

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: BlackboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates generation texts for Blackbox model on the input batch.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with List[List[float]] generation texts at 'greedy_texts' key.
        """
        with torch.no_grad():
            sequences = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                n=1,
            )

        return {"greedy_texts": sequences}


class GreedyProbsCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * generation texts
    * tokens of the generation texts
    * probabilities distribution of the generated tokens
    * attention masks across the model (if applicable)
    * embeddings from the model
    """

    def __init__(self, n_alternatives: int = 10):
        super().__init__(
            [
                "input_tokens",
                "greedy_log_probs",
                "greedy_tokens",
                "greedy_tokens_alternatives",
                "greedy_texts",
                "greedy_log_likelihoods",
                "train_greedy_log_likelihoods",
                # "embeddings",
                # "token_embeddings",
                "embeddings_all",
                # "attention_features",
                # "attention_weights",
                # "train_embeddings_all",
                "train_attention_features",
                # "train_greedy_texts",
                # "train_greedy_tokens",
                "train_target_texts",
                "train_input_texts",
                "train_greedy_tokens_alternatives",
                # "train_attention_max_features",
                # "train_attention_max_features_values",
                # "train_attention_max_features_token",
                # "train_attention_all"
                # "attention_all",
                # "attention_max_features",
                # "attention_max_features_values",
                # "attention_max_features_token",
            ],
            [],
        )
        self.n_alternatives = n_alternatives

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
                - 'input_tokens' (List[List[int]]): tokenized input texts,
                - 'greedy_log_probs' (List[List[np.array]]): logarithms of autoregressive
                        probability distributions at each token,
                - 'greedy_texts' (List[str]): model generations corresponding to the inputs,
                - 'greedy_tokens' (List[List[int]]): tokenized model generations,
                - 'attention' (List[List[np.array]]): attention maps at each token, if applicable to the model,
                - 'greedy_log_likelihoods' (List[List[float]]): log-probabilities of the generated tokens.
        """
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        with torch.no_grad():
            out = model.generate(
                **batch,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=2,
                output_attentions=False,
                output_hidden_states=True,
                num_return_sequences=1,
                suppress_tokens=(
                    []
                    if model.generation_parameters.allow_newlines
                    else [
                        t
                        for t in range(len(model.tokenizer))
                        if "\n" in model.tokenizer.decode([t])
                    ]
                ),
            )
            logits = torch.stack(out.scores, dim=1)
            
            # attentions = out.attentions
            sequences = out.sequences

        cut_logits = []
        cut_sequences = []
        cut_texts = []
        cut_alternatives = []
        for i in range(len(texts)):
            if model.model_type == "CausalLM":
                idx = batch["input_ids"].shape[1]
                seq = sequences[i, idx:].cpu()
            else:
                seq = sequences[i, 1:].cpu()
            length, text_length = len(seq), len(seq)
            for j in range(len(seq)):
                if seq[j] == model.tokenizer.eos_token_id:
                    length = j + 1
                    text_length = j
                    break
            cut_sequences.append(seq[:length].tolist())
            cut_texts.append(model.tokenizer.decode(seq[:text_length]))
            cut_logits.append(logits[i, :length, :].cpu().numpy())
            cut_alternatives.append([[] for _ in range(length)])
            for j in range(length):
                lt = logits[i, j, :].cpu().numpy()
                best_tokens = np.argpartition(lt, -self.n_alternatives)
                ln = len(best_tokens)
                best_tokens = best_tokens[ln - self.n_alternatives : ln]
                for t in best_tokens:
                    cut_alternatives[-1][j].append((t.item(), lt[t].item()))
                cut_alternatives[-1][j].sort(
                    key=lambda x: x[0] == cut_sequences[-1][j],
                    reverse=True,
                )
        # attn_features = []
        # attention_all = []
        # attn_features_max = []
        # attn_features_max_tokens = []
        # attn_features_max_values = []
        # attn = []
        # for i in range(len(texts)):
        #     c = len(cut_sequences[i])
        #     attn_mask = np.zeros(
        #         shape=(
        #             model.model.config.num_attention_heads
        #             * model.model.config.num_hidden_layers,
        #             c,
        #             c,
        #         )
        #     )
        #     for j in range(1, c):
        #         attn_mask[:, j, :j] = (
        #             torch.vstack(
        #                 [
        #                     attentions[j][layer][0][head][0][-j:]
        #                     for layer in range(len(attentions[j]))
        #                     for head in range(len(attentions[j][layer][0]))
        #                 ]
        #             )
        #             .cpu()
        #             .numpy()
        #         )

            # attn.append(attn_mask.transpose(1,2,0))
            # top_n = min(3, attn_mask.max(0).shape[1])
            # topk = torch.topk(torch.tensor(attn_mask.max(0)), k=top_n, dim=1)
            # attn_features_max_values_s = []
            # attention_max_features_token_s = []

            # attention_all.append(attn_mask.max(0))
            # for j in range(1, c):
            #     attn_features.append(attn_mask[:, j, j - 1])
                # attn_features_max_values_i = []
                # attention_max_features_token_i = []
                # for k in range(top_n):
                #     attn_features_max.append(attn_mask[:, j, topk.indices[j][k].item()])
                #     attn_features_max_values_i.append(topk.values[j][k].item())
                #     attention_max_features_token_i.append(topk.indices[j][k].item())

                # attn_features_max_values_s.append(attn_features_max_values_i)
                # attention_max_features_token_s.append(attention_max_features_token_i)

        # attn_features_max_values.append(attn_features_max_values_s)
        # attn_features_max_tokens.append(attention_max_features_token_s)
        # attention_all = np.array(attention_all)

        # attn_features = np.array(attn_features)

        ll = []
        for i in range(len(texts)):
            log_probs = cut_logits[i]
            tokens = cut_sequences[i]
            assert len(tokens) == len(log_probs)
            ll.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

        if model.model_type == "CausalLM":
            embeddings_dict = {
                "embeddings_all_decoder": out.hidden_states,
            }
        elif model.model_type == "Seq2SeqLM":
            embeddings_dict = {
                "embeddings_all_encoder": out.encoder_hidden_states,
                "embeddings_all_decoder": out.decoder_hidden_states,
            }
        else:
            raise NotImplementedError

        # if model.model_type == "CausalLM":
        #     embeddings_dict = {
        #         "embeddings_decoder": embeddings_decoder,
        #         "token_embeddings_decoder": token_embeddings_decoder,
        #     }
        # elif model.model_type == "Seq2SeqLM":
        #     embeddings_dict = {
        #         "embeddings_encoder": embeddings_encoder,
        #         "embeddings_decoder": embeddings_decoder,
        #     }
        # else:
        #     raise NotImplementedError

        result_dict = {
            "input_tokens": batch["input_ids"].to("cpu").tolist(),
            "greedy_log_probs": cut_logits,
            "greedy_tokens": cut_sequences,
            "greedy_tokens_alternatives": cut_alternatives,
            "greedy_texts": cut_texts,
            "greedy_log_likelihoods": ll,
            # "attention_features": attn_features,
            # "attention_weights": attention_all,
            # "attention_max_features": attn_features_max,
            # "attention_max_features_token": attn_features_max_tokens,
            # "attention_max_features_values": attn_features_max_values,
            # "attention_all": attn
        }
        result_dict.update(embeddings_dict)

        return result_dict
