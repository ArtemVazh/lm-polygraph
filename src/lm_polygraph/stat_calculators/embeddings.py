import torch
import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel
from transformers import AutoTokenizer, AutoModel


NAMING_MAP = {"bert-base-uncased": "bert_base", 
              "bert-large-uncased": "bert_large", 
              "google/electra-small-discriminator": "electra_base", 
              "roberta-base": "roberta_base", 
              "roberta-large": "roberta_large",
              "meta-llama/Llama-3.2-1B": "llama1b", 
              "meta-llama/Llama-3.2-3B": "llama3b", 
              "meta-llama/Llama-3.1-8B": "llama8b"}

MODELS = {}

def get_embeddings_from_output(
    output,
    batch,
    model_type,
    hidden_state: List[str] = ["encoder", "decoder"],
    ignore_padding: bool = True,
    use_averaging: bool = True,
    all_layers: bool = False,
    aggregation_method: str = "mean",
    level: str = "sequence",
    hidden_layer: int = -1,
    return_source_hidden_states: bool = False,
):
    batch_embeddings = None
    batch_embeddings_decoder = None
    batch_size = len(batch["input_ids"])

    if model_type == "CausalLM":
        input_tokens_hs = output.hidden_states[0][hidden_layer].cpu().detach()
        if return_source_hidden_states:
            batch_embeddings_decoder = input_tokens_hs.mean(axis=1).cpu().detach()
        else:
            if not all_layers:
                if len(output.hidden_states) > 1:
                    generated_tokens_hs = torch.cat(
                        [
                            h[hidden_layer].cpu().detach()
                            for h in output.hidden_states[1:]
                        ],
                        dim=1,
                    )
            else:
                input_tokens_hs = output.hidden_states[0].mean(axis=0).cpu().detach()
                if len(output.hidden_states) > 1:
                    generated_tokens_hs = torch.cat(
                        [
                            h.mean(axis=0).cpu().detach()
                            for h in output.hidden_states[1:]
                        ],
                        dim=1,
                    )
            if len(output.hidden_states) > 1:
                if level == "sequence":
                    batch_embeddings_decoder = (
                        torch.cat([input_tokens_hs, generated_tokens_hs], dim=1)
                        .mean(axis=1)
                        .cpu()
                        .detach()
                    )
                elif level == "token":
                    batch_embeddings_decoder = (
                        torch.cat([input_tokens_hs[:, -1:], generated_tokens_hs], dim=1)
                        .cpu()
                        .detach()
                    )
            else:
                batch_embeddings_decoder = input_tokens_hs.mean(axis=1).cpu().detach()
        batch_embeddings = None
    elif model_type == "Seq2SeqLM":
        if use_averaging:
            if "decoder" in hidden_state:
                try:
                    decoder_hidden_states = torch.stack(
                        [torch.stack(hidden) for hidden in output.decoder_hidden_states]
                    )
                    if all_layers:
                        agg_decoder_hidden_states = decoder_hidden_states[
                            :, :, :, 0
                        ].mean(axis=1)
                    else:
                        agg_decoder_hidden_states = decoder_hidden_states[:, -1, :, 0]

                    batch_embeddings_decoder = aggregate(
                        agg_decoder_hidden_states, aggregation_method, axis=0
                    )
                    batch_embeddings_decoder = (
                        batch_embeddings_decoder.cpu()
                        .detach()
                        .reshape(batch_size, -1, agg_decoder_hidden_states.shape[-1])[
                            :, 0
                        ]
                    )
                except TypeError:
                    if all_layers:
                        agg_decoder_hidden_states = torch.stack(
                            output.decoder_hidden_states
                        ).mean(axis=0)
                    else:
                        agg_decoder_hidden_states = torch.stack(
                            output.decoder_hidden_states
                        )[-1]

                    batch_embeddings_decoder = aggregate(
                        agg_decoder_hidden_states, aggregation_method, axis=1
                    )
                    batch_embeddings_decoder = (
                        batch_embeddings_decoder.cpu()
                        .detach()
                        .reshape(-1, agg_decoder_hidden_states.shape[-1])
                    )

            if "encoder" in hidden_state:
                mask = batch["attention_mask"][:, :, None].cpu().detach()
                seq_lens = batch["attention_mask"].sum(-1)[:, None].cpu().detach()
                if all_layers:
                    encoder_embeddings = (
                        aggregate(
                            torch.stack(output.encoder_hidden_states), "mean", axis=0
                        )
                        .cpu()
                        .detach()
                        * mask
                    )
                else:
                    encoder_embeddings = (
                        output.encoder_hidden_states[-1].cpu().detach() * mask
                    )

                if ignore_padding:
                    if aggregation_method == "mean":
                        batch_embeddings = (encoder_embeddings).sum(
                            1
                        ).cpu().detach() / seq_lens
                    else:
                        batch_embeddings = (
                            aggregate(encoder_embeddings, aggregation_method, axis=1)
                            .cpu()
                            .detach()
                        )
                else:
                    batch_embeddings = (
                        aggregate(encoder_embeddings, aggregation_method, axis=1)
                        .cpu()
                        .detach()
                    )
            if not ("encoder" in hidden_state) and not ("decoder" in hidden_state):
                raise NotImplementedError
        else:
            if "decoder" in hidden_state:
                decoder_hidden_states = torch.stack(
                    [torch.stack(hidden) for hidden in output.decoder_hidden_states]
                )
                last_decoder_hidden_states = decoder_hidden_states[-1, -1, :, 0]
                batch_embeddings_decoder = (
                    last_decoder_hidden_states.reshape(
                        batch_size, -1, last_decoder_hidden_states.shape[-1]
                    )[:, 0]
                    .cpu()
                    .detach()
                )
            if "encoder" in hidden_state:
                batch_embeddings = output.encoder_hidden_states[-1][:, 0].cpu().detach()
            if not ("encoder" in hidden_state) and not ("decoder" in hidden_state):
                raise NotImplementedError
    else:
        raise NotImplementedError

    return batch_embeddings, batch_embeddings_decoder


def aggregate(x, aggregation_method, axis):
    if aggregation_method == "max":
        return x.max(axis=axis).values
    elif aggregation_method == "mean":
        return x.mean(axis=axis)
    elif aggregation_method == "sum":
        return x.sum(axis=axis)


class AllEmbeddingsCalculator(StatCalculator):
    def __init__(self):
        super().__init__(["train_embeddings_all", "train_greedy_texts", "background_train_greedy_texts"], [])

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
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
            
            sequences = out.sequences

        cut_sequences = []
        cut_texts = []
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

        if model.model_type == "CausalLM":
            return {
                "embeddings_all_decoder": out.hidden_states,
                "greedy_tokens": cut_sequences,
                "greedy_texts": cut_texts,
            }
        elif model.model_type == "Seq2SeqLM":
            return {
                "embeddings_all_encoder": out.encoder_hidden_states,
                "embeddings_all_decoder": out.decoder_hidden_states,
            }
        else:
            raise NotImplementedError


class OutputWrapper:
    hidden_states = None
    encoder_hidden_states = None
    decoder_hidden_states = None


class EmbeddingsCalculator(StatCalculator):
    def __init__(self, hidden_layers: List[int] = [-1], stage: str = "train"):
        self.hidden_layers = hidden_layers
        self.stage = stage
        if stage == "train":
            self.stage += "_"

        stats = []
        for layer in self.hidden_layers:
            if layer == -1:
                layer_name = ""
            else:
                layer_name = f"_{layer}"
            if stage == "train":
                stats += [
                    f"{self.stage}embeddings{layer_name}",
                    f"background_{self.stage}embeddings{layer_name}",
                    f"{self.stage}token_embeddings{layer_name}",
                    f"background_{self.stage}token_embeddings{layer_name}",
                ]
            else:
                stats += [
                    f"{self.stage}embeddings{layer_name}",
                    f"{self.stage}token_embeddings{layer_name}",
                ]
        super().__init__(
            stats,
            [f"{self.stage}embeddings_all"],
        )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        with torch.no_grad():
            out = OutputWrapper()
            if model.model_type == "CausalLM":
                out.hidden_states = dependencies["embeddings_all_decoder"]
            elif model.model_type == "Seq2SeqLM":
                out.decoder_hidden_states = dependencies["embeddings_all_decoder"]
                out.encoder_hidden_states = dependencies["embeddings_all_encoder"]

            results = {}
            for layer in self.hidden_layers:
                if layer == -1:
                    layer_name = ""
                else:
                    layer_name = f"_{layer}"
                embeddings_encoder, embeddings_decoder = get_embeddings_from_output(
                    out,
                    batch,
                    model.model_type,
                    level="sequence",
                    hidden_layer=layer,
                )
                token_embeddings_encoder, token_embeddings_decoder = (
                    get_embeddings_from_output(
                        out,
                        batch,
                        model.model_type,
                        level="token",
                        hidden_layer=layer,
                    )
                )
                if token_embeddings_decoder is None:
                    token_embeddings_decoder = torch.empty(
                        (0, embeddings_decoder.shape[-1]), dtype=torch.float32
                    )
                elif len(token_embeddings_decoder.shape) == 3:
                    token_embeddings_decoder = token_embeddings_decoder.reshape(
                        -1, token_embeddings_decoder.shape[-1]
                    )

                if model.model_type == "CausalLM":
                    results[f"embeddings_decoder{layer_name}"] = (
                        embeddings_decoder.cpu().detach().numpy()
                    )
                    results[f"token_embeddings_decoder{layer_name}"] = (
                        token_embeddings_decoder.cpu().detach().numpy()
                    )
                elif model.model_type == "Seq2SeqLM":
                    pass
                else:
                    raise NotImplementedError

        return results


class SourceEmbeddingsCalculator(StatCalculator):
    def __init__(self, hidden_layer=-1):
        self.hidden_layer = hidden_layer
        if self.hidden_layer == -1:
            super().__init__(["train_source_embeddings"], ["train_embeddings_all"])
        else:
            super().__init__(
                [f"train_source_embeddings_{self.hidden_layer}"],
                ["train_embeddings_all"],
            )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        with torch.no_grad():
            out = OutputWrapper()
            if model.model_type == "CausalLM":
                out.hidden_states = dependencies["embeddings_all_decoder"]
            elif model.model_type == "Seq2SeqLM":
                out.decoder_hidden_states = dependencies["embeddings_all_decoder"]
                out.encoder_hidden_states = dependencies["embeddings_all_encoder"]

            embeddings_encoder, embeddings_decoder = get_embeddings_from_output(
                out,
                batch,
                model.model_type,
                level="sequence",
                hidden_layer=self.hidden_layer,
                return_source_hidden_states=True,
            )

        if model.model_type == "CausalLM":
            if self.hidden_layer == -1:
                return {
                    "source_embeddings_decoder": embeddings_decoder.cpu()
                    .detach()
                    .numpy(),
                }
            else:
                return {
                    f"source_embeddings_decoder_{self.hidden_layer}": embeddings_decoder.cpu()
                    .detach()
                    .numpy(),
                }
        else:
            raise NotImplementedError
        
class ProxyEmbeddingsCalculator(StatCalculator):
    def __init__(self, proxy_model: str = "bert-base-uncased", hidden_layers: List[int] = [-1], stage: str = "train"):
        self.tokenizer = None
        self.model = None
        self.stage = stage
        self.proxy_model = proxy_model
        self.model_name = NAMING_MAP[proxy_model]
        self.hidden_layers = hidden_layers
        
        stats = []
        if stage == "train":
            self.stage += "_"
            stats.append(f"{self.stage}proxy_{self.model_name}_tokens")
            for layer in self.hidden_layers:
                if layer == -1:
                    layer_name = ""
                else:
                    layer_name = f"_{layer}"
                stats += [
                    f"{self.stage}proxy_{self.model_name}_token_embeddings{layer_name}",
                    f"background_{self.stage}proxy_{self.model_name}_token_embeddings{layer_name}",
                ]
            
            super().__init__(
                stats,
                [f"{self.stage}greedy_texts",
                 f"background_{self.stage}greedy_texts"
                ],
            )

        else:
            stats.append(f"{self.stage}proxy_{self.model_name}_tokens")
            for layer in self.hidden_layers:
                if layer == -1:
                    layer_name = ""
                else:
                    layer_name = f"_{layer}"
                stats += [
                    f"{self.stage}proxy_{self.model_name}_token_embeddings{layer_name}",
                ]
            
            super().__init__(
                stats,
                [f"{self.stage}greedy_texts"]
            )

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        full_texts = []
        for input_text, text in zip(dependencies["input_texts"], dependencies["greedy_texts"]):
            full_texts.append(input_text+text)
            
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.proxy_model, truncation_side='left')
            if self.model_name in ["bert_base", "bert_large", "electra_base", "roberta_base", "roberta_large"]:
                if self.model_name in MODELS:
                    self.model = MODELS[self.model_name]
                else:
                    self.model = AutoModel.from_pretrained(self.proxy_model).to("cuda")
                    MODELS[self.model_name] = self.model
            else:
                if self.model_name in MODELS:
                    self.model = MODELS[self.model_name]
                else:
                    self.model = AutoModel.from_pretrained(self.proxy_model, device_map="auto")
                    MODELS[self.model_name] = self.model
            
        with torch.no_grad():
            encoded_greedy = self.tokenizer(dependencies["greedy_texts"], return_tensors='pt', truncation=True)
            encoded_input = self.tokenizer(full_texts, return_tensors='pt', truncation=True)
            encoded_input = {k: v.to(self.model.device) for k, v in encoded_input.items()}
            output = self.model(**encoded_input, output_hidden_states=True)

        proxy_tokens = encoded_input["input_ids"].cpu().detach().numpy()[:, encoded_greedy["input_ids"].shape[1]:].tolist()
        results = {}
        for layer in self.hidden_layers:
            if layer == -1:
                layer_name = ""
            else:
                layer_name = f"_{layer}"
                          
            if self.model_name in ["bert_base", "bert_large", "electra_base", "roberta_base", "roberta_large"]:
                token_embeddings = output.hidden_states[layer]
                results[f"proxy_{self.model_name}_token_embeddings_decoder{layer_name}"] = (
                    token_embeddings.cpu().detach().numpy().reshape(-1, token_embeddings.shape[-1])[-len(proxy_tokens[0]):]
                )
            else:
                token_embeddings = torch.cat(output.hidden_states)[layer]
                results[f"proxy_{self.model_name}_token_embeddings_decoder{layer_name}"] = (
                    token_embeddings.cpu().detach().numpy().reshape(-1, token_embeddings.shape[-1])[-len(proxy_tokens[0])-1:-1]
                )

        results[f"proxy_{self.model_name}_tokens"] = proxy_tokens
        return results

class InternalStatesCalculator(StatCalculator):
    def __init__(self, topk: int = 10, stage: str = "train"):
        self.stage = stage
        if stage == "train":
            self.stage += "_"
        super().__init__(
            [
                f"{self.stage}final_output_ranks",
                f"{self.stage}topk_layer_distance",
                f"{self.stage}topk_prob",
            ],
            [f"{self.stage}embeddings_all"],
        )
        self.topk = topk

    def process_word_id_topk_rank_data(self, word_id_topk_rank, model_emb, device):
        layer_distance = torch.zeros((1, word_id_topk_rank.shape[-1])).to(device)
        for layer in range(word_id_topk_rank.shape[0] - 1):
            words0 = word_id_topk_rank[layer, :]
            words1 = word_id_topk_rank[layer + 1, :]
            if isinstance(words0, torch.Tensor):
                words0 = words0.clone().detach().unsqueeze(0).to(device)
            else:
                words0 = torch.tensor(words0).unsqueeze(0).to(device)

            if isinstance(words1, torch.Tensor):
                words1 = words1.clone().detach().unsqueeze(0).to(device)
            else:
                words1 = torch.tensor(words1).unsqueeze(0).to(device)

            emb0 = model_emb(words0)
            emb1 = model_emb(words1)

            distances = torch.cosine_similarity(emb0, emb1, dim=2).to(device)
            layer_distance = torch.cat((layer_distance, distances), dim=0)
        return layer_distance

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        with torch.no_grad():
            out = OutputWrapper()
            if model.model_type == "CausalLM":
                out.hidden_states = dependencies["embeddings_all_decoder"]
                first_tokens_hs = torch.cat(
                    [
                        out.hidden_states[0][layer][:, -1, :]
                        for layer in range(len(out.hidden_states[0]))
                    ]
                )
                layerwise_preds = model.model.lm_head(first_tokens_hs)

            elif model.model_type == "Seq2SeqLM":
                out.decoder_hidden_states = dependencies["embeddings_all_decoder"]
                out.encoder_hidden_states = dependencies["embeddings_all_encoder"]
                first_tokens_hs = torch.cat(
                    [
                        out.hidden_states[0][layer][:, -1, :]
                        for layer in range(len(out.decoder_hidden_states[0]))
                    ]
                )
                layerwise_preds = model.model.lm_head(first_tokens_hs)

            logits = torch.softmax(layerwise_preds, dim=-1)
            sorted, indices = torch.sort(logits)
            location = indices[:, -self.topk :]
            prob = sorted[:, -self.topk :]
            predicted_token = indices[-1][-1].item()
            layer_distance = (
                self.process_word_id_topk_rank_data(
                    location, model.model.model.embed_tokens, model.device()
                )
                .cpu()
                .detach()
                .numpy()
            )
            ranks = []
            if model.model_type == "CausalLM":
                for layer in range(len(out.hidden_states[0])):
                    ranks.append(
                        np.argwhere(
                            indices[layer].cpu().detach().numpy()[::-1]
                            == predicted_token
                        )[0, 0]
                        + 1
                    )
            elif model.model_type == "Seq2SeqLM":
                for layer in range(len(out.decoder_hidden_states[0])):
                    ranks.append(
                        np.argwhere(
                            indices[layer].cpu().detach().numpy()[::-1]
                            == predicted_token
                        )[0, 0]
                        + 1
                    )

        return {
            "final_output_ranks": ranks,
            "topk_layer_distance": layer_distance,
            "topk_prob": prob.cpu().detach().numpy(),
        }


class TokenInternalStatesCalculator(StatCalculator):
    def __init__(self, topk=10, hidden_layer=-1, stage: str = "train"):
        self.stage = stage
        if stage == "train":
            self.stage += "_"
        super().__init__(
            [
                f"{self.stage}final_output_ranks_all",
                f"{self.stage}topk_layer_distance_all",
                f"{self.stage}topk_prob_all",
            ],
            [f"{self.stage}embeddings_all"],
        )
        self.hidden_layer = hidden_layer
        self.topk = topk

    def process_word_id_topk_rank_data(self, word_id_topk_rank, model_emb, device):
        layer_distance = torch.zeros((1, word_id_topk_rank.shape[-1])).to(device)
        for layer in range(word_id_topk_rank.shape[0] - 1):
            words0 = word_id_topk_rank[layer, :]
            words1 = word_id_topk_rank[layer + 1, :]
            if isinstance(words0, torch.Tensor):
                words0 = words0.clone().detach().unsqueeze(0).to(device)
            else:
                words0 = torch.tensor(words0).unsqueeze(0).to(device)

            if isinstance(words1, torch.Tensor):
                words1 = words1.clone().detach().unsqueeze(0).to(device)
            else:
                words1 = torch.tensor(words1).unsqueeze(0).to(device)

            emb0 = model_emb(words0)
            emb1 = model_emb(words1)

            distances = torch.cosine_similarity(emb0, emb1, dim=2).to(device)
            layer_distance = torch.cat((layer_distance, distances), dim=0)
        return layer_distance

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        batch = {k: v.to(model.device()) for k, v in batch.items()}
        ranks = []
        topk_layer_distance = []
        topk_prob = []
        with torch.no_grad():
            out = OutputWrapper()
            if model.model_type == "CausalLM":
                out.hidden_states = dependencies["embeddings_all_decoder"]

            elif model.model_type == "Seq2SeqLM":
                out.decoder_hidden_states = dependencies["embeddings_all_decoder"]
                out.encoder_hidden_states = dependencies["embeddings_all_encoder"]

            _, token_embeddings_decoder = get_embeddings_from_output(
                out,
                batch,
                model.model_type,
                level="token",
                hidden_layer=self.hidden_layer,
            )
            if len(token_embeddings_decoder.shape) == 3:
                token_embeddings_decoder = token_embeddings_decoder.reshape(
                    -1, token_embeddings_decoder.shape[-1]
                )

            for token_idx in range(token_embeddings_decoder.shape[0]):
                if model.model_type == "CausalLM":
                    tokens_hs = torch.cat(
                        [
                            out.hidden_states[token_idx][layer][:, -1, :]
                            for layer in range(len(out.hidden_states[0]))
                        ]
                    )

                elif model.model_type == "Seq2SeqLM":
                    tokens_hs = torch.cat(
                        [
                            out.hidden_states[token_idx][layer][:, -1, :]
                            for layer in range(len(out.decoder_hidden_states[0]))
                        ]
                    )

                layerwise_preds = model.model.lm_head(tokens_hs.to(model.device()))
                logits = torch.softmax(layerwise_preds, dim=-1)
                sorted, indices = torch.sort(logits)
                location = indices[:, -self.topk :]
                prob = sorted[:, -self.topk :]
                predicted_token = indices[-1][-1].item()
                layer_distance = (
                    self.process_word_id_topk_rank_data(
                        location, model.model.model.embed_tokens, model.device()
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
                ranks_t = []
                if model.model_type == "CausalLM":
                    for layer in range(len(out.hidden_states[0])):
                        ranks_t.append(
                            np.argwhere(
                                indices[layer].cpu().detach().numpy()[::-1]
                                == predicted_token
                            )[0, 0]
                            + 1
                        )
                elif model.model_type == "Seq2SeqLM":
                    for layer in range(len(out.decoder_hidden_states[0])):
                        ranks_t.append(
                            np.argwhere(
                                indices[layer].cpu().detach().numpy()[::-1]
                                == predicted_token
                            )[0, 0]
                            + 1
                        )
                ranks.append(ranks_t)
                topk_prob.append(prob.cpu().detach().numpy())
                topk_layer_distance.append(layer_distance)

        ranks = np.array(ranks)
        topk_prob = np.array(topk_prob)
        topk_layer_distance = np.array(topk_layer_distance)

        return {
            "final_output_ranks_all": ranks,
            "topk_layer_distance_all": topk_layer_distance,
            "topk_prob_all": topk_prob,
        }
