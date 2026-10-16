"""The eval's layer list fits the model family that validation enforces: it ends at the last layer,
whose state the metrics turn into next-token log-probs, and names each layer once in order, as
vLLM's hidden-state extractor expects."""

from __future__ import annotations

from common.config import chain_spec
from eval_service import config
from model_validation import families


def test_layers_end_at_the_last_layer_and_are_sorted_and_unique():
    depth = families.load(chain_spec().family).arch_value("text_config.num_hidden_layers")
    assert config.LAYERS[-1] == depth
    assert list(config.LAYERS) == sorted(set(config.LAYERS))
