import os
import logging

from lm_polygraph.stat_calculators import *
from lm_polygraph.utils.deberta import Deberta, MultilingualDeberta
from lm_polygraph.utils.openai_chat import OpenAIChat

from typing import Dict, List, Optional, Tuple

log = logging.getLogger("lm_polygraph")


def register_stat_calculators(
    deberta_batch_size: int = 10,  # TODO: rename to NLI model
    deberta_device: Optional[str] = None,  # TODO: rename to NLI model
    language: str = "en",
    n_ccp_alternatives: int = 10,
    cache_path=os.path.expanduser("~") + "/.cache",
    model=None,
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
                continue
            stat_calculators[stat] = calculator_class
            stat_dependencies[stat] = calculator_class.stat_dependencies

    _register(GreedyProbsCalculator())
    _register(BlackboxGreedyTextsCalculator())
    _register(EntropyCalculator())
    _register(GreedyLMProbsCalculator())
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
    _register(SamplingGenerationCalculator())
    _register(BlackboxSamplingGenerationCalculator())
    _register(BartScoreCalculator())
    _register(ModelScoreCalculator())
    _register(EnsembleTokenLevelDataCalculator())
    _register(SemanticMatrixCalculator(nli_model=nli_model))
    _register(CrossEncoderSimilarityMatrixCalculator(nli_model=nli_model))
    _register(GreedyProbsCalculator(n_alternatives=n_ccp_alternatives))
    _register(GreedyAlternativesNLICalculator(nli_model=nli_model))
    _register(TrainGreedyAlternativesNLICalculator(nli_model=nli_model))
    _register(TrainGreedyAlternativesFactPrefNLICalculator(nli_model=nli_model))
    _register(GreedyAlternativesFactPrefNLICalculator(nli_model=nli_model))
    _register(ClaimsExtractor(openai_chat=openai_chat, language=language))

    _register(AllEmbeddingsCalculator())
    _register(SourceEmbeddingsCalculator())
    _register(SamplingGenerationEmbeddingsCalculator())

    _register(InternalStatesCalculator(stage="train"))
    _register(TokenInternalStatesCalculator(stage="train"))

    hidden_layers = list(range(model.model.config.num_hidden_layers - 1)) + [-1]

    _register(EmbeddingsCalculator(hidden_layers=hidden_layers, stage="train"))
    _register(EmbeddingsCalculator(hidden_layers=hidden_layers, stage=""))

    proxy_hidden_layers = list(range(12)) + [-1]
    
    _register(ProxyEmbeddingsCalculator(hidden_layers=proxy_hidden_layers, stage="train"))
    _register(ProxyEmbeddingsCalculator(hidden_layers=proxy_hidden_layers, stage=""))
    
    _register(ProxyEmbeddingsBaseCalculator(stage="train"))
    _register(ProxyEmbeddingsBaseCalculator(stage=""))

    _register(SamplingGenerationEmbeddingsCalculator(hidden_layers=hidden_layers))

    for layer in range(model.model.config.num_hidden_layers - 1):
        _register(SourceEmbeddingsCalculator(hidden_layer=layer))

    _register(InternalStatesCalculator(stage=""))
    _register(TokenInternalStatesCalculator(stage=""))

    log.info("Done intitializing stat calculators...")

    return stat_calculators, stat_dependencies
