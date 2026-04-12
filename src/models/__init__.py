from .base import BaseRecommender, EmbeddingLayer
from .backbone import DeepFM, DLRM, DCNv2, AutoInt, FinalMLP, DCNv3
from .experts import Expert, ExpertLayer
from .moe import MoELayer, TrieGuidedMoE
from .trie_router import TrieRouter, HybridRouter
from .adaptive_alpha import (
    AdaptiveAlphaRouter,
    AdaptiveAlphaMoE,
    AdaptiveAlphaRecommender,
    compute_token_frequencies,
    get_batch_token_freqs,
)
from .enhanced_adaptive import (
    EnhancedAdaptiveRouter,
    EnhancedAdaptiveMoE,
    EnhancedAdaptiveRecommender,
)
from .cold_start_baselines import (
    DropoutNet,
    MeLU,
    MetaEmb,
    MMoE,
    PLE,
    WarmUp,
)

__all__ = [
    "BaseRecommender",
    "EmbeddingLayer",
    # CTR Backbones
    "DeepFM",
    "DLRM",
    "DCNv2",
    "AutoInt",
    "FinalMLP",
    "DCNv3",
    # Experts
    "Expert",
    "ExpertLayer",
    "MoELayer",
    "TrieGuidedMoE",
    "TrieRouter",
    "HybridRouter",
    # Adaptive Alpha
    "AdaptiveAlphaRouter",
    "AdaptiveAlphaMoE",
    "AdaptiveAlphaRecommender",
    "compute_token_frequencies",
    "get_batch_token_freqs",
    # Enhanced Adaptive (with data structures)
    "EnhancedAdaptiveRouter",
    "EnhancedAdaptiveMoE",
    "EnhancedAdaptiveRecommender",
    # Cold-Start Baselines
    "DropoutNet",
    "MeLU",
    "MetaEmb",
    "MMoE",
    "PLE",
    "WarmUp",
]
