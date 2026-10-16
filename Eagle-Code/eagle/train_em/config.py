"""Configuration for EM-Style Multi-Drafter Training.

This module defines the configuration class for the EM-style load balancing
and routing approach to multi-drafter speculative decoding.
"""

from transformers.configuration_utils import PretrainedConfig


class EMConfig(PretrainedConfig):
    r"""Configuration class for EM-Style Multi-Drafter Speculative Decoding.

    This configuration extends the standard EAGLE configuration with parameters
    specific to the EM-style training approach with multiple independent drafters.

    Args:
        num_drafters (`int`, *optional*, defaults to 4):
            Number of independent drafter models (K) to train in parallel.
            Recommended range: 4-5 for balanced diversity and compute.

        draft_gamma (`int`, *optional*, defaults to 60):
            Block-length gamma for candidate draft sequence generation.
            This is the maximum number of tokens each drafter proposes.

        kl_forward_weight (`float`, *optional*, defaults to 1.0):
            Weight for Forward-KL divergence loss on ACCEPTED tokens.
            Forward-KL encourages mode-covering behavior (student covers teacher).

        kl_reverse_weight (`float`, *optional*, defaults to 0.5):
            Weight for Reverse-KL divergence loss on REJECTED tokens.
            Reverse-KL encourages mode-seeking behavior (student avoids wrong peaks).

        feature_loss_type (`str`, *optional*, defaults to "l1"):
            Type of feature regression loss to use. Options: "l1", "mse".
            This is kept separate from the KL divergences as in standard EAGLE.

        feature_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight for the feature regression loss term.

        routing_strategy (`str`, *optional*, defaults to "best-length"):
            Strategy for selecting drafter at inference time. Options:
            - "best-length": Use drafter with historically longest accepted drafts
            - "confidence": Use confidence-based scoring (like dual-head EAGLE)
            - "mixed": Weighted combination of length and confidence
            - "adaptive": Sliding window based on recent performance

        routing_update_interval (`int`, *optional*, defaults to 100):
            How often (in steps) to update routing statistics during training.

        winner_takes_all (`bool`, *optional*, defaults to True):
            If True, only the drafter with longest accepted length receives gradients
            (pure EM-style). If False, all drafters train with weighted gradients.

        top_m_for_training (`int`, *optional*, defaults to 1):
            If winner_takes_all is False, train top-M drafters by acceptance length.

        sliding_window_size (`int`, *optional*, defaults to 100):
            Size of sliding window for adaptive routing statistics.

        initialization_mode (`str`, *optional*, defaults to "random"):
            How to initialize the K drafters. Options:
            - "random": Standard random initialization (same as EAGLE)
            - "same_checkpoint": Load same checkpoint, different seeds for divergence
            - "specialized": Each drafter from different fine-tuned checkpoint

        verbose (`bool`, *optional*, defaults to False):
            Whether to print routing statistics during training.

    Example:
        ```python
        from eagle.em-alg.config import EMConfig

        config = EMConfig(
            num_drafters=4,
            draft_gamma=60,
            kl_forward_weight=1.0,
            kl_reverse_weight=0.5,
            routing_strategy="best-length"
        )
        ```
    """
    model_type = "em_multi_drafter"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        num_drafters=4,
        draft_gamma=60,
        kl_forward_weight=1.0,
        kl_reverse_weight=0.5,
        feature_loss_type="l1",
        feature_loss_weight=1.0,
        routing_strategy="best-length",
        routing_update_interval=100,
        winner_takes_all=True,
        top_m_for_training=1,
        sliding_window_size=100,
        initialization_mode="random",
        verbose=False,
        **kwargs,
    ):
        self.num_drafters = num_drafters
        self.draft_gamma = draft_gamma
        self.kl_forward_weight = kl_forward_weight
        self.kl_reverse_weight = kl_reverse_weight
        self.feature_loss_type = feature_loss_type
        self.feature_loss_weight = feature_loss_weight
        self.routing_strategy = routing_strategy
        self.routing_update_interval = routing_update_interval
        self.winner_takes_all = winner_takes_all
        self.top_m_for_training = top_m_for_training
        self.sliding_window_size = sliding_window_size
        self.initialization_mode = initialization_mode
        self.verbose = verbose

        super().__init__(**kwargs)
