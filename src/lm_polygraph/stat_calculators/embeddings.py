import torch
import numpy as np

from typing import Dict, List

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


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
                        [h[hidden_layer].cpu().detach() for h in output.hidden_states[1:]],
                        dim=1,
                    )
            else:
                input_tokens_hs = output.hidden_states[0].mean(axis=0).cpu().detach()
                if len(output.hidden_states) > 1:
                    generated_tokens_hs = torch.cat(
                        [h.mean(axis=0).cpu().detach() for h in output.hidden_states[1:]],
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
        super().__init__(["embeddings_all"], [])

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
             
        if model.model_type == "CausalLM":
            return {
                "embeddings_all_decoder": out.hidden_states,
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
    def __init__(self, hidden_layer = -1):
        self.hidden_layer = hidden_layer
        if self.hidden_layer == -1:
            super().__init__(["train_embeddings", "background_train_embeddings", "train_token_embeddings", "background_train_token_embeddings"], ["embeddings_all"])
        else:
            super().__init__([f"embeddings_{self.hidden_layer}", f"train_embeddings_{self.hidden_layer}", f"background_train_embeddings_{self.hidden_layer}", f"background_token_embeddings_{self.hidden_layer}",
                              f"token_embeddings_{self.hidden_layer}", f"train_token_embeddings_{self.hidden_layer}", f"background_train_token_embeddings_{self.hidden_layer}"], ["embeddings_all"])

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
                out, batch, model.model_type, level="sequence", hidden_layer=self.hidden_layer
            )
            token_embeddings_encoder, token_embeddings_decoder = get_embeddings_from_output(
                out, batch, model.model_type, level="token", hidden_layer=self.hidden_layer
            )
            if token_embeddings_decoder is None:
                token_embeddings_decoder = torch.empty((0, embeddings_decoder.shape[-1]), dtype=torch.float32)
            elif len(token_embeddings_decoder.shape) == 3:
                token_embeddings_decoder = token_embeddings_decoder.reshape(-1, token_embeddings_decoder.shape[-1])

        if model.model_type == "CausalLM":
            if self.hidden_layer == -1:
                return {
                    "embeddings_decoder": embeddings_decoder.cpu().detach().numpy(),
                    "token_embeddings_decoder": token_embeddings_decoder.cpu().detach().numpy(),
                }
            else:
                return {
                    f"embeddings_decoder_{self.hidden_layer}": embeddings_decoder.cpu().detach().numpy(),
                    f"token_embeddings_decoder_{self.hidden_layer}": token_embeddings_decoder.cpu().detach().numpy(),
                }
        elif model.model_type == "Seq2SeqLM":
            return {
                "embeddings_encoder": embeddings_encoder.cpu().detach().numpy(),
                "embeddings_decoder": embeddings_decoder.cpu().detach().numpy(),
            }
        else:
            raise NotImplementedError
        
class SourceEmbeddingsCalculator(StatCalculator):
    def __init__(self, hidden_layer = -1):
        self.hidden_layer = hidden_layer
        if self.hidden_layer == -1:
            super().__init__(["train_source_embeddings"], ["embeddings_all"])
        else:
            super().__init__([f"train_source_embeddings_{self.hidden_layer}"], ["embeddings_all"])
        
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
                out, batch, model.model_type, level="sequence", hidden_layer=self.hidden_layer, return_source_hidden_states=True
            )
            
        if model.model_type == "CausalLM":
            if self.hidden_layer == -1:
                return {
                    "source_embeddings_decoder": embeddings_decoder.cpu().detach().numpy(),
                }
            else:
                return {
                    f"source_embeddings_decoder_{self.hidden_layer}": embeddings_decoder.cpu().detach().numpy(),
                }
        else:
            raise NotImplementedError
        
