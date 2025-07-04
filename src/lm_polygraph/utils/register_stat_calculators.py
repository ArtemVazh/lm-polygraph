import os
import logging

from lm_polygraph.stat_calculators import *
from lm_polygraph.utils.deberta import Deberta, MultilingualDeberta
from lm_polygraph.utils.openai_chat import OpenAIChat
from transformers import AutoConfig
from lm_polygraph.utils.model import Model, BlackboxModel

from typing import Dict, List, Optional, Tuple

log = logging.getLogger("lm_polygraph")


def register_stat_calculators(
    deberta_batch_size: int = 10,  # TODO: rename to NLI model
    deberta_device: Optional[str] = None,  # TODO: rename to NLI model
    language: str = "en",
    n_ccp_alternatives: int = 10,
    cache_path=os.path.expanduser("~") + "/.cache",
    model: Model = None,
) -> Tuple[Dict[str, "StatCalculator"], Dict[str, List[str]]]:
    """
    Registers all available statistic calculators to be seen by UEManager for properly organizing the calculations
    order.
    """
    stat_calculators: Dict[str, "StatCalculator"] = {}
    stat_dependencies: Dict[str, List[str]] = {}

    log.info("=" * 100)
    log.info("Loading NLI model...")

    if language == "en":
        nli_model = Deberta(batch_size=deberta_batch_size, device=deberta_device)
    elif language in ["zh", "ar", "ru"]:
        nli_model = MultilingualDeberta(
            batch_size=deberta_batch_size,
            device=deberta_device,
        )
    else:
        raise Exception(f"Unsupported language: {language}")

    log.info("=" * 100)
    log.info("Initializing stat calculators...")

    openai_chat = OpenAIChat(openai_model="gpt-4o-mini", cache_path=cache_path)

    def _register(calculator_class: StatCalculator):
        for stat in calculator_class.stats:
            if stat in stat_calculators.keys():
                print(stat, stat_calculators.keys())
                raise ValueError(
                    "A statistic is supposed to be processed by a single calculator only."
                )
            stat_calculators[stat] = calculator_class
            stat_dependencies[stat] = calculator_class.stat_dependencies

    _register(InitialStateCalculator())
    _register(SemanticMatrixCalculator(nli_model=nli_model))
    _register(SemanticClassesCalculator())

    if isinstance(model, BlackboxModel):
        _register(BlackboxGreedyTextsCalculator())
        _register(BlackboxSamplingGenerationCalculator())

        proxy_models = ["bert-base-uncased", "bert-large-uncased", "google/electra-small-discriminator", "roberta-base", "roberta-large",
                    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B"]

        for proxy_model in proxy_models:
            cfg = AutoConfig.from_pretrained(proxy_model)
            proxy_hidden_layers = list(range(cfg.num_hidden_layers-1)) + [-1]
            
            _register(ProxyEmbeddingsCalculator(proxy_model=proxy_model, hidden_layers=proxy_hidden_layers, stage="train"))
            _register(ProxyEmbeddingsCalculator(proxy_model=proxy_model, hidden_layers=proxy_hidden_layers, stage=""))
        
    else:
        _register(GreedyProbsCalculator(n_alternatives=n_ccp_alternatives))
        _register(EntropyCalculator())
        _register(GreedyLMProbsCalculator())
        _register(SamplingGenerationCalculator())
        _register(BartScoreCalculator())
        _register(ModelScoreCalculator())
        
        _register(GreedySemanticMatrixCalculator(nli_model=nli_model))
        _register(ConcatGreedySemanticMatrixCalculator(nli_model=nli_model))
        
        # _register(EmbeddingsCalculator())
        _register(EnsembleTokenLevelDataCalculator())
        _register(CrossEncoderSimilarityMatrixCalculator(nli_model=nli_model))
        _register(GreedyAlternativesNLICalculator(nli_model=nli_model))
        _register(GreedyAlternativesFactPrefNLICalculator(nli_model=nli_model))
        _register(TrainGreedyAlternativesNLICalculator(nli_model=nli_model))
        _register(TrainGreedyAlternativesFactPrefNLICalculator(nli_model=nli_model))
        _register(ClaimsExtractor(openai_chat=openai_chat, language=language))
        _register(
            PromptCalculator(
                "Question: {q}\n Possible answer:{a}\n "
                "Is the possible answer:\n (A) True\n (B) False\n The possible answer is:",
                "True",
                "p_true",
                sample_text_dependency=None,  # Not calculate T text samples for P(True)
            )
        )
        _register(
            PromptCalculator(
                "Question: {q}\n Here are some ideas that were brainstormed: {s}\n Possible answer:{a}\n "
                "Is the possible answer:\n (A) True\n (B) False\n The possible answer is:",
                "True",
                "p_true_sampling",
            )
        )
        _register(
            PromptCalculator(
                "Question: {q}\n Possible answer:{a}\n "
                "Is the possible answer True or False? The possible answer is: ",
                "True",
                "p_true_claim",
                input_text_dependency="claim_input_texts_concatenated",
                sample_text_dependency=None,
                generation_text_dependency="claim_texts_concatenated",
            )
        )

        _register(AttentionCalculator())
        _register(AttentionForwardPassCalculator())

        # _register(AllEmbeddingsCalculator())
        _register(SourceEmbeddingsCalculator())
        # _register(SamplingGenerationEmbeddingsCalculator())
    
        _register(InternalStatesCalculator(stage="train"))
        _register(TokenInternalStatesCalculator(stage="train"))
    
        if "gemma-3" in model.model_path:
            hidden_layers = list(range(model.model.config.text_config.num_hidden_layers - 1)) + [-1]
        else:
            hidden_layers = list(range(model.model.config.num_hidden_layers - 1)) + [-1]
    
        _register(EmbeddingsCalculator(hidden_layers=hidden_layers, stage="train"))
        _register(EmbeddingsCalculator(hidden_layers=hidden_layers, stage=""))

        _register(SamplingGenerationEmbeddingsCalculator(hidden_layers=hidden_layers))
        _register(SamplingGenerationAttentionCalculator())

        if "gemma-3" in model.model_path:
            for layer in range(model.model.config.text_config.num_hidden_layers - 1):
                _register(SourceEmbeddingsCalculator(hidden_layer=layer))
        else:
            for layer in range(model.model.config.num_hidden_layers - 1):
                _register(SourceEmbeddingsCalculator(hidden_layer=layer))
    
        _register(InternalStatesCalculator(stage=""))
        _register(TokenInternalStatesCalculator(stage=""))

    log.info("Done intitializing stat calculators...")

    return stat_calculators, stat_dependencies
