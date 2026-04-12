"""
Structured Prompt Builder for LLM integration.

Key principle: Prompts contain ONLY statistical structure, never raw tokens.
This enables LLM reasoning even when token semantics are unavailable.

The LLM receives:
- Trie node statistics (frequency, CTR lift, information content)
- Distribution summaries (head/mid/tail ratios)
- Expert routing hints
- Historical patterns

The LLM can then:
- Suggest routing strategies
- Explain prediction rationale
- Identify anomalous patterns
- Recommend personalization strategies
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json


@dataclass
class TrieNodeSummary:
    """Summary of a Trie node for prompt inclusion."""
    path: str  # e.g., "head > high_info"
    token_count: int
    avg_ctr: float
    avg_lift: float
    expert_id: int
    coverage: float  # Fraction of sample's tokens in this node


@dataclass
class SampleStatistics:
    """Statistical summary of a sample for LLM prompt."""
    # Distribution across Trie
    head_ratio: float
    mid_ratio: float
    tail_ratio: float

    # Information content
    high_info_ratio: float
    avg_lift: float
    max_lift: float

    # Diversity
    num_unique_paths: int
    path_entropy: float

    # Expert routing
    suggested_experts: List[int]
    expert_confidence: float

    # Anomaly indicators
    is_cold_start: bool  # Many unseen tokens
    is_high_variance: bool  # Conflicting signals


class TriePromptTemplate:
    """
    Templates for different LLM tasks.

    All templates are designed to work WITHOUT semantic token information.
    """

    ROUTING_DECISION = """
You are an expert routing advisor for a recommendation system.
Your task is to suggest the best expert network for processing a user's request.

## System Context
- We have {num_experts} specialized expert networks
- Experts are specialized for different patterns:
  - Experts 0-1: High-frequency, stable patterns (head)
  - Experts 2-4: Medium-frequency patterns (mid)
  - Experts 5-7: Long-tail, rare patterns (tail)

## Sample Statistics
The current sample has the following statistical profile:
- Head token ratio: {head_ratio:.2%}
- Mid token ratio: {mid_ratio:.2%}
- Tail token ratio: {tail_ratio:.2%}
- High-information token ratio: {high_info_ratio:.2%}
- Average CTR lift: {avg_lift:.2f}
- Maximum CTR lift: {max_lift:.2f}
- Number of unique Trie paths: {num_unique_paths}
- Path diversity (entropy): {path_entropy:.2f}

## Trie-suggested experts: {suggested_experts}
## System confidence in suggestion: {expert_confidence:.2%}

## Your Task
Based on these statistics, recommend:
1. Primary expert to use (0-{max_expert_id})
2. Confidence level (low/medium/high)
3. Brief reasoning (1-2 sentences)

Respond in JSON format:
{{"expert": <int>, "confidence": "<str>", "reasoning": "<str>"}}
"""

    PREDICTION_EXPLANATION = """
You are explaining a recommendation system's prediction to analysts.
The system uses encrypted/hashed features, so you cannot see actual content.
Instead, you must explain based on statistical patterns.

## Prediction Context
- Predicted CTR: {predicted_ctr:.4f}
- Global average CTR: {global_ctr:.4f}
- Prediction is {direction} average by {lift_factor:.2f}x

## Statistical Profile of this Sample
- Head tokens (high frequency): {head_ratio:.1%}
- Tail tokens (rare): {tail_ratio:.1%}
- High-information tokens: {high_info_ratio:.1%}
- Average CTR lift of tokens: {avg_lift:.2f}

## Trie Path Distribution
{trie_path_summary}

## Expert Routing
- Selected expert: {selected_expert} ({expert_type})
- Routing confidence: {routing_confidence:.1%}

## Your Task
Provide a statistical explanation for this prediction.
Focus on:
1. Which statistical patterns drove the prediction
2. Why this sample is {direction} average
3. Confidence level in the prediction

Keep explanation under 100 words. Do not reference any specific tokens.
"""

    ANOMALY_DETECTION = """
You are monitoring a recommendation system for anomalous patterns.

## Current Sample Statistics
{sample_stats}

## Historical Baseline (past 24 hours)
{baseline_stats}

## Flags
- Cold start detected: {is_cold_start}
- High variance signals: {is_high_variance}
- Expert routing unusual: {routing_unusual}

## Your Task
1. Is this sample anomalous? (yes/no)
2. If yes, what type of anomaly?
   - cold_start: Many unseen tokens
   - distribution_shift: Unusual head/tail ratio
   - signal_conflict: Contradictory CTR signals
   - routing_outlier: Unexpected expert selection
3. Recommended action:
   - proceed: Normal processing
   - flag: Log for review
   - fallback: Use conservative baseline model

Respond in JSON:
{{"is_anomaly": <bool>, "anomaly_type": "<str>|null", "action": "<str>", "notes": "<str>"}}
"""

    BATCH_ANALYSIS = """
You are analyzing a batch of recommendations for quality assurance.

## Batch Summary ({batch_size} samples)
- Average predicted CTR: {avg_predicted_ctr:.4f}
- CTR standard deviation: {std_predicted_ctr:.4f}
- Positive prediction rate: {positive_rate:.1%}

## Expert Utilization
{expert_utilization}

## Trie Coverage
- Samples with >50% head tokens: {head_heavy_pct:.1%}
- Samples with >50% tail tokens: {tail_heavy_pct:.1%}
- Cold start samples: {cold_start_pct:.1%}

## Potential Issues
{potential_issues}

## Your Task
Provide a quality assessment:
1. Overall batch quality (good/acceptable/concerning)
2. Key observations (2-3 bullet points)
3. Recommendations for improvement (if any)
"""


class StructuredPromptBuilder:
    """
    Builds prompts from Trie statistics for LLM consumption.

    This class is the bridge between the Trie-based statistical
    structure and LLM reasoning capabilities.
    """

    def __init__(
        self,
        num_experts: int = 8,
        global_ctr: float = 0.04,
        templates: Optional[Dict[str, str]] = None,
    ):
        self.num_experts = num_experts
        self.global_ctr = global_ctr
        self.templates = templates or {
            'routing': TriePromptTemplate.ROUTING_DECISION,
            'explanation': TriePromptTemplate.PREDICTION_EXPLANATION,
            'anomaly': TriePromptTemplate.ANOMALY_DETECTION,
            'batch': TriePromptTemplate.BATCH_ANALYSIS,
        }

    def build_routing_prompt(
        self,
        sample_stats: SampleStatistics,
    ) -> str:
        """Build a prompt for routing decision."""
        return self.templates['routing'].format(
            num_experts=self.num_experts,
            head_ratio=sample_stats.head_ratio,
            mid_ratio=sample_stats.mid_ratio,
            tail_ratio=sample_stats.tail_ratio,
            high_info_ratio=sample_stats.high_info_ratio,
            avg_lift=sample_stats.avg_lift,
            max_lift=sample_stats.max_lift,
            num_unique_paths=sample_stats.num_unique_paths,
            path_entropy=sample_stats.path_entropy,
            suggested_experts=sample_stats.suggested_experts,
            expert_confidence=sample_stats.expert_confidence,
            max_expert_id=self.num_experts - 1,
        )

    def build_explanation_prompt(
        self,
        sample_stats: SampleStatistics,
        predicted_ctr: float,
        selected_expert: int,
        expert_type: str,
        routing_confidence: float,
        trie_node_summaries: List[TrieNodeSummary],
    ) -> str:
        """Build a prompt for prediction explanation."""
        # Determine direction
        lift_factor = predicted_ctr / self.global_ctr
        direction = "above" if lift_factor > 1 else "below"

        # Format Trie path summary
        path_lines = []
        for node in sorted(trie_node_summaries, key=lambda x: -x.coverage)[:5]:
            path_lines.append(
                f"  - {node.path}: {node.coverage:.1%} coverage, "
                f"lift={node.avg_lift:.2f}, expert={node.expert_id}"
            )
        trie_path_summary = "\n".join(path_lines) if path_lines else "  (no significant paths)"

        return self.templates['explanation'].format(
            predicted_ctr=predicted_ctr,
            global_ctr=self.global_ctr,
            direction=direction,
            lift_factor=lift_factor,
            head_ratio=sample_stats.head_ratio,
            tail_ratio=sample_stats.tail_ratio,
            high_info_ratio=sample_stats.high_info_ratio,
            avg_lift=sample_stats.avg_lift,
            trie_path_summary=trie_path_summary,
            selected_expert=selected_expert,
            expert_type=expert_type,
            routing_confidence=routing_confidence,
        )

    def build_anomaly_prompt(
        self,
        sample_stats: SampleStatistics,
        baseline_stats: Dict[str, float],
        routing_unusual: bool = False,
    ) -> str:
        """Build a prompt for anomaly detection."""
        sample_stats_str = json.dumps({
            'head_ratio': f"{sample_stats.head_ratio:.2%}",
            'tail_ratio': f"{sample_stats.tail_ratio:.2%}",
            'avg_lift': f"{sample_stats.avg_lift:.2f}",
            'num_unique_paths': sample_stats.num_unique_paths,
        }, indent=2)

        baseline_stats_str = json.dumps({
            k: f"{v:.2%}" if 'ratio' in k else f"{v:.2f}"
            for k, v in baseline_stats.items()
        }, indent=2)

        return self.templates['anomaly'].format(
            sample_stats=sample_stats_str,
            baseline_stats=baseline_stats_str,
            is_cold_start=sample_stats.is_cold_start,
            is_high_variance=sample_stats.is_high_variance,
            routing_unusual=routing_unusual,
        )

    def build_batch_prompt(
        self,
        batch_predictions: List[float],
        expert_counts: Dict[int, int],
        trie_coverage_stats: Dict[str, float],
        potential_issues: List[str],
    ) -> str:
        """Build a prompt for batch analysis."""
        import numpy as np

        predictions = np.array(batch_predictions)

        expert_util_lines = [
            f"  - Expert {i}: {count} samples ({count/len(batch_predictions)*100:.1f}%)"
            for i, count in sorted(expert_counts.items())
        ]

        issues_str = "\n".join(f"  - {issue}" for issue in potential_issues) or "  (none detected)"

        return self.templates['batch'].format(
            batch_size=len(batch_predictions),
            avg_predicted_ctr=predictions.mean(),
            std_predicted_ctr=predictions.std(),
            positive_rate=(predictions > 0.5).mean(),
            expert_utilization="\n".join(expert_util_lines),
            head_heavy_pct=trie_coverage_stats.get('head_heavy', 0),
            tail_heavy_pct=trie_coverage_stats.get('tail_heavy', 0),
            cold_start_pct=trie_coverage_stats.get('cold_start', 0),
            potential_issues=issues_str,
        )

    @staticmethod
    def parse_routing_response(response: str) -> Dict[str, Any]:
        """Parse LLM response for routing decision."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Fallback: return default
        return {
            'expert': 0,
            'confidence': 'low',
            'reasoning': 'Failed to parse LLM response',
        }

    @staticmethod
    def parse_anomaly_response(response: str) -> Dict[str, Any]:
        """Parse LLM response for anomaly detection."""
        try:
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return {
            'is_anomaly': False,
            'anomaly_type': None,
            'action': 'proceed',
            'notes': 'Failed to parse LLM response',
        }
