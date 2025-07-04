import torch
import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from .embeddings import get_embeddings_from_output
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel


class BlackboxSamplingGenerationCalculator(StatCalculator):
    """
    Calculates several sampled texts for Blackbox model (lm_polygraph.BlackboxModel).
    """

    def __init__(self, samples_n: int = 5):
        """
        Parameters:
            samples_n (int): number of samples to generate per input text. Default: 10
        """
        self.samples_n = samples_n
        super().__init__(["sample_texts"], [])

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: BlackboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates sampled texts for Blackbox model on the input batch.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with List[List[str]] sampled texts at 'sample_texts' key.
        """
        if isinstance(model, BlackboxModel):
            samples = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                n=self.samples_n,
            )
        else:
            samples_n = getattr(model.generation_parameters, "samples_n", self.samples_n)
            samples = [[] for _ in range(len(texts))]
            out = model.generate_texts(
                input_texts=texts,
                max_new_tokens=max_new_tokens,
                min_length=2,
                do_sample=True,
                num_beams=1,
                num_return_sequences=samples_n,
            )
            for i in range(len(texts)):
                for j in range(self.samples_n):
                    samples[i].append(out[i * samples_n + j])

        return {
            "sample_texts": samples,
        }


def _gen_samples(n_samples, model, batch, **kwargs):
    batch_size = len(batch["input_ids"])
    logits, sequences, embeddings, attentions = (
        [[] for _ in range(batch_size)],
        [[] for _ in range(batch_size)],
        [],
        [[] for _ in range(batch_size)],
    )
    with torch.no_grad():
        for k in range(n_samples):
            out = model.generate(**batch, **kwargs)
            cur_logits = torch.stack(out.scores, dim=1)
            if model.model_type == "CausalLM":
                embeddings.append(
                    {
                        "sample_embeddings_all_decoder": out.hidden_states,
                    }
                )
            elif model.model_type == "Seq2SeqLM":
                embeddings.append(
                    {
                        "sample_embeddings_all_encoder": out.encoder_hidden_states,
                        "sample_embeddings_all_decoder": out.decoder_hidden_states,
                    }
                )
            for i in range(batch_size):
                sequences[i].append(out.sequences[i])
                logits[i].append(cur_logits[i])
                attentions[i].append(out.attentions)
    sequences = [s for sample_seqs in sequences for s in sample_seqs]
    return sequences, sum(logits, []), embeddings, attentions


class SamplingGenerationCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * sampled texts
    * tokens of the sampled texts
    * probabilities of the sampled tokens generation
    """

    def __init__(self, samples_n: int = 10):
        """
        Parameters:
            samples_n (int): number of samples to generate per input text. Default: 10
        """
        self.samples_n = samples_n
        super().__init__(
            [
                "sample_log_probs",
                "sample_tokens",
                "sample_texts",
                "sample_log_likelihoods",
                "sample_embeddings_all",
                "sample_attention_all",
            ],
            [],
        )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the statistics of sampling texts.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'sample_texts' (List[List[str]]): `samples_n` texts for each input text in the batch,
                - 'sample_tokens' (List[List[List[float]]]): tokenized 'sample_texts',
                - 'sample_log_probs' (List[List[float]]): sum of the log probabilities at each token of the sampling generation.
                - 'sample_log_likelihoods' (List[List[List[float]]]): log probabilities at each token of the sampling generation.
        """
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        samples_n = getattr(model.generation_parameters, "samples_n", self.samples_n)
        sequences, logits, embeddings, attentions = _gen_samples(
            samples_n,
            model,
            batch,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_attentions=True,
            max_new_tokens=max_new_tokens,
            min_new_tokens=2,
            do_sample=True,
            num_beams=1,
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

        log_probs = [[] for _ in range(len(texts))]
        tokens = [[] for _ in range(len(texts))]
        texts = [[] for _ in range(len(texts))]
        log_likelihoods = [[] for _ in range(len(texts))]
        if model.model_type == "Seq2SeqLM":
            sequences = [seq[1:] for seq in sequences]
        for i in range(len(logits)):
            log_prob, ll, toks = 0, [], []
            inp_size = (
                len(batch["input_ids"][int(i / samples_n)])
                if model.model_type == "CausalLM"
                else 0
            )
            for j in range(len(sequences[i]) - inp_size):
                cur_token = sequences[i][j + inp_size].item()
                log_prob += logits[i][j][cur_token].item()
                if cur_token == model.tokenizer.eos_token_id:
                    break
                ll.append(logits[i][j][cur_token].item())
                toks.append(cur_token)

            log_likelihoods[int(i / samples_n)].append(ll)
            log_probs[int(i / samples_n)].append(log_prob)
            tokens[int(i / samples_n)].append(toks)
            texts[int(i / samples_n)].append(model.tokenizer.decode(toks))

        return {
            "sample_log_likelihoods": log_likelihoods,
            "sample_log_probs": log_probs,
            "sample_tokens": tokens,
            "sample_texts": texts,
            "sample_embeddings_all": embeddings,
            "sample_attention_all": attentions,
        }


class OutputWrapper:
    hidden_states = None
    encoder_hidden_states = None
    decoder_hidden_states = None


class SamplingGenerationEmbeddingsCalculator(StatCalculator):
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at input texts batch calculates:
    * sampled texts
    * tokens of the sampled texts
    * probabilities of the sampled tokens generation
    """

    def __init__(self, samples_n: int = 10, hidden_layers: List[int] = [-1]):
        """
        Parameters:
            samples_n (int): number of samples to generate per input text. Default: 10
        """
        self.samples_n = samples_n
        self.hidden_layers = hidden_layers

        stats = []
        for layer in self.hidden_layers:
            if layer == -1:
                layer_name = ""
            else:
                layer_name = f"_{layer}"
            stats += [
                f"sample_embeddings{layer_name}",
                f"sample_embeddings_last_token{layer_name}",
            ]
        super().__init__(
            stats,
            ["sample_embeddings_all"],
        )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the statistics of sampling texts.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'sample_texts' (List[List[str]]): `samples_n` texts for each input text in the batch,
                - 'sample_tokens' (List[List[List[float]]]): tokenized 'sample_texts',
                - 'sample_log_probs' (List[List[float]]): sum of the log probabilities at each token of the sampling generation.
                - 'sample_log_likelihoods' (List[List[List[float]]]): log probabilities at each token of the sampling generation.
        """
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)

        batch_size = len(batch["input_ids"])
        results = {}

        for layer in self.hidden_layers:
            if layer == -1:
                layer_name = ""
            else:
                layer_name = f"_{layer}"

            embeddings = [[] for _ in range(batch_size)]
            embeddings_last_token = [[] for _ in range(batch_size)]

            for sample_embeddings in dependencies["sample_embeddings_all"]:
                out = OutputWrapper()
                if model.model_type == "CausalLM":
                    out.hidden_states = sample_embeddings[
                        "sample_embeddings_all_decoder"
                    ]
                elif model.model_type == "Seq2SeqLM":
                    out.decoder_hidden_states = sample_embeddings[
                        "sample_embeddings_all_decoder"
                    ]
                    out.encoder_hidden_states = sample_embeddings[
                        "sample_embeddings_all_encoder"
                    ]

                _, cur_embeddings = get_embeddings_from_output(
                    out,
                    batch,
                    model.model_type,
                    level="sequence",
                    hidden_layer=layer,
                )
                _, cur_token_embeddings = get_embeddings_from_output(
                    out,
                    batch,
                    model.model_type,
                    level="token",
                    hidden_layer=layer,
                )

                for i in range(batch_size):
                    embeddings[i].append(cur_embeddings[i].cpu().detach().numpy())
                    if len(cur_token_embeddings.shape) > 2:
                        embeddings_last_token[i].append(
                            cur_token_embeddings[i, -1].cpu().detach().numpy()
                        )
                    else:
                        embeddings_last_token[i].append(
                            cur_token_embeddings[i].cpu().detach().numpy()
                        )
            results[f"sample_embeddings{layer_name}"] = embeddings
            results[f"sample_embeddings_last_token{layer_name}"] = embeddings_last_token

        return results
    
    
class SamplingGenerationAttentionCalculator(StatCalculator):

    def __init__(self, samples_n: int = 10):
        """
        Parameters:
            samples_n (int): number of samples to generate per input text. Default: 10
        """
        self.samples_n = samples_n

        super().__init__(
            ["sample_attention"],
            ["sample_attention_all"],
        )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the statistics of sampling texts.

        Parameters:
            dependencies (Dict[str, np.ndarray]): input statistics, can be empty (not used).
            texts (List[str]): Input texts batch used for model generation.
            model (Model): Model used for generation.
            max_new_tokens (int): Maximum number of new tokens at model generation. Default: 100.
        Returns:
            Dict[str, np.ndarray]: dictionary with the following items:
                - 'sample_texts' (List[List[str]]): `samples_n` texts for each input text in the batch,
                - 'sample_tokens' (List[List[List[float]]]): tokenized 'sample_texts',
                - 'sample_log_probs' (List[List[float]]): sum of the log probabilities at each token of the sampling generation.
                - 'sample_log_likelihoods' (List[List[List[float]]]): log probabilities at each token of the sampling generation.
        """
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)

        batch_size = len(batch["input_ids"])
        results = {}
    
        attention_all = [[] for _ in range(batch_size)]
        samples_n = getattr(model.generation_parameters, "samples_n", self.samples_n)

        for k, sample_attention in enumerate(dependencies["sample_attention_all"]):
            for i in range(samples_n):
                attentions = sample_attention[i]
                c = len(dependencies["sample_tokens"][k][i])
                attn_mask = np.zeros(
                    shape=(
                        model.model.config.num_attention_heads
                        * model.model.config.num_hidden_layers,
                        c,
                        c,
                    )
                )
                for j in range(1, c):
                    stacked_attention = torch.vstack(
                        [
                            attentions[j][layer][0][head][0][-j:]
                            for layer in range(len(attentions[j]))
                            for head in range(len(attentions[j][layer][0]))
                        ]
                    )
                    if stacked_attention.dtype == torch.bfloat16:
                        stacked_attention = stacked_attention.to(
                            torch.float16
                        )  # numpy does not support bfloat16

                    attn_mask[:, j, :j] = stacked_attention.cpu().numpy()
                attention_all[k].append(attn_mask)

            results[f"sample_attention"] = attention_all

        return results
