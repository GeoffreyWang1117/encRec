"""
LLM-based reranking and explanation for recommendations.

While the main prediction uses fast neural models,
LLM can provide:
1. Post-hoc explanations
2. Reranking based on statistical patterns
3. Quality assurance checks
4. User-facing explanations
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import json
import asyncio


@dataclass
class RerankedItem:
    """A reranked recommendation item."""
    original_rank: int
    new_rank: int
    original_score: float
    adjustment_factor: float
    reasoning: str


@dataclass
class PredictionExplanation:
    """Explanation for a prediction."""
    prediction: float
    confidence: str
    main_factors: List[str]
    statistical_reasoning: str
    caveats: List[str]


class LLMReranker:
    """
    LLM-based reranking of recommendation results.

    Uses statistical patterns to adjust rankings,
    particularly useful for:
    - Diversity injection
    - Cold-start item boosting
    - Anomaly handling
    """

    def __init__(
        self,
        llm_backend: 'LLMBackend',
        prompt_template: Optional[str] = None,
    ):
        self.backend = llm_backend
        self.prompt_template = prompt_template or self._default_template()

    def _default_template(self) -> str:
        return """
You are optimizing a recommendation list based on statistical patterns.

## Current Rankings (top {num_items})
{rankings_summary}

## Statistical Context
- Global average CTR: {global_ctr:.4f}
- User's historical profile: {user_profile}
- Current context: {context}

## Patterns to Consider
{patterns}

## Your Task
Suggest reranking adjustments to improve:
1. Expected click-through rate
2. Diversity of recommendations
3. Long-tail item exposure

For each item, provide:
- adjustment: multiplier (0.5 to 2.0)
- reasoning: brief explanation

Respond in JSON format:
{{"adjustments": [{{"item_id": <int>, "adjustment": <float>, "reasoning": "<str>"}}]}}
"""

    async def rerank(
        self,
        items: List[Dict[str, Any]],
        user_stats: Dict[str, float],
        context: Dict[str, Any],
        top_k: int = 10,
    ) -> List[RerankedItem]:
        """
        Rerank items using LLM guidance.

        Args:
            items: List of items with scores and statistics
            user_stats: User profile statistics
            context: Current context (time, device, etc.)
            top_k: Number of items to rerank

        Returns:
            List of RerankedItem with new rankings
        """
        # Prepare rankings summary
        rankings_lines = []
        for i, item in enumerate(items[:top_k]):
            rankings_lines.append(
                f"  {i+1}. Score: {item['score']:.4f}, "
                f"Lift: {item.get('lift', 1.0):.2f}, "
                f"Freq: {item.get('frequency_bucket', 'unknown')}"
            )

        # Prepare patterns summary
        patterns = self._extract_patterns(items[:top_k])

        prompt = self.prompt_template.format(
            num_items=top_k,
            rankings_summary="\n".join(rankings_lines),
            global_ctr=context.get('global_ctr', 0.04),
            user_profile=json.dumps(user_stats),
            context=json.dumps(context),
            patterns=patterns,
        )

        # Get LLM response
        response = await self.backend.generate(prompt)

        # Parse response
        try:
            import re
            json_match = re.search(r'\{[^{}]*"adjustments"[^{}]*\[.*?\][^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                adjustments = {
                    adj['item_id']: (adj['adjustment'], adj['reasoning'])
                    for adj in parsed.get('adjustments', [])
                }
            else:
                adjustments = {}
        except (json.JSONDecodeError, KeyError):
            adjustments = {}

        # Apply adjustments and rerank
        reranked = []
        for i, item in enumerate(items[:top_k]):
            adj_factor, reasoning = adjustments.get(i, (1.0, "No adjustment"))
            new_score = item['score'] * adj_factor

            reranked.append(RerankedItem(
                original_rank=i + 1,
                new_rank=0,  # Will be set after sorting
                original_score=item['score'],
                adjustment_factor=adj_factor,
                reasoning=reasoning,
            ))

        # Sort by adjusted score
        scored_items = [(r, items[r.original_rank - 1]['score'] * r.adjustment_factor)
                       for r in reranked]
        scored_items.sort(key=lambda x: -x[1])

        # Assign new ranks
        for new_rank, (item, _) in enumerate(scored_items, 1):
            item.new_rank = new_rank

        return [item for item, _ in scored_items]

    def _extract_patterns(self, items: List[Dict]) -> str:
        """Extract statistical patterns from items."""
        patterns = []

        # Head/tail distribution
        head_count = sum(1 for i in items if i.get('frequency_bucket') == 'head')
        tail_count = sum(1 for i in items if i.get('frequency_bucket') == 'tail')
        patterns.append(f"- Head items: {head_count}, Tail items: {tail_count}")

        # Score distribution
        scores = [i['score'] for i in items]
        if scores:
            import numpy as np
            patterns.append(f"- Score range: [{min(scores):.4f}, {max(scores):.4f}]")
            patterns.append(f"- Score std: {np.std(scores):.4f}")

        # Diversity
        unique_paths = set(i.get('trie_path', '') for i in items)
        patterns.append(f"- Unique Trie paths: {len(unique_paths)}")

        return "\n".join(patterns)


class LLMExplainer:
    """
    Generate human-readable explanations for predictions.

    Critical for:
    - User trust and transparency
    - Debugging and analysis
    - Regulatory compliance
    """

    def __init__(
        self,
        llm_backend: 'LLMBackend',
        global_ctr: float = 0.04,
    ):
        self.backend = llm_backend
        self.global_ctr = global_ctr

    async def explain_prediction(
        self,
        prediction: float,
        sample_stats: 'SampleStatistics',
        routing_info: Dict[str, Any],
        trie_summaries: List['TrieNodeSummary'],
    ) -> PredictionExplanation:
        """
        Generate explanation for a single prediction.

        Explanation is based on statistical patterns,
        NOT on actual token content (which is encrypted).
        """
        from .prompt_builder import StructuredPromptBuilder, TrieNodeSummary

        prompt_builder = StructuredPromptBuilder(global_ctr=self.global_ctr)

        prompt = prompt_builder.build_explanation_prompt(
            sample_stats=sample_stats,
            predicted_ctr=prediction,
            selected_expert=routing_info.get('expert_id', 0),
            expert_type=routing_info.get('expert_type', 'general'),
            routing_confidence=routing_info.get('confidence', 0.5),
            trie_node_summaries=trie_summaries,
        )

        response = await self.backend.generate(prompt)

        # Parse into structured explanation
        explanation = self._parse_explanation(response, prediction)

        return explanation

    def _parse_explanation(
        self,
        response: str,
        prediction: float,
    ) -> PredictionExplanation:
        """Parse LLM response into structured explanation."""
        # Extract key components from response
        lines = response.strip().split('\n')

        main_factors = []
        caveats = []
        reasoning = response

        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('*'):
                if 'caveat' in line.lower() or 'note' in line.lower():
                    caveats.append(line[1:].strip())
                else:
                    main_factors.append(line[1:].strip())

        # Determine confidence
        lift = prediction / self.global_ctr
        if 0.8 <= lift <= 1.2:
            confidence = 'medium'
        elif lift > 1.5 or lift < 0.5:
            confidence = 'high'
        else:
            confidence = 'low'

        return PredictionExplanation(
            prediction=prediction,
            confidence=confidence,
            main_factors=main_factors[:5] if main_factors else ["Statistical patterns indicate this prediction"],
            statistical_reasoning=reasoning[:500],
            caveats=caveats[:3] if caveats else [],
        )

    async def explain_batch(
        self,
        predictions: List[Tuple[float, 'SampleStatistics', Dict]],
        concurrency: int = 5,
    ) -> List[PredictionExplanation]:
        """Explain a batch of predictions."""
        semaphore = asyncio.Semaphore(concurrency)

        async def explain_one(pred, stats, routing):
            async with semaphore:
                return await self.explain_prediction(pred, stats, routing, [])

        tasks = [
            explain_one(pred, stats, routing)
            for pred, stats, routing in predictions
        ]

        return await asyncio.gather(*tasks)

    def generate_summary_report(
        self,
        explanations: List[PredictionExplanation],
    ) -> str:
        """
        Generate a summary report for a batch of explanations.

        Useful for offline analysis and quality assurance.
        """
        if not explanations:
            return "No explanations to summarize."

        import numpy as np

        predictions = [e.prediction for e in explanations]
        confidences = {'low': 0, 'medium': 0, 'high': 0}
        for e in explanations:
            confidences[e.confidence] = confidences.get(e.confidence, 0) + 1

        # Collect common factors
        all_factors = []
        for e in explanations:
            all_factors.extend(e.main_factors)

        from collections import Counter
        common_factors = Counter(all_factors).most_common(5)

        report = f"""
## Prediction Explanation Summary

### Statistics
- Total predictions: {len(explanations)}
- Mean prediction: {np.mean(predictions):.4f}
- Std prediction: {np.std(predictions):.4f}
- Range: [{min(predictions):.4f}, {max(predictions):.4f}]

### Confidence Distribution
- High confidence: {confidences['high']} ({confidences['high']/len(explanations)*100:.1f}%)
- Medium confidence: {confidences['medium']} ({confidences['medium']/len(explanations)*100:.1f}%)
- Low confidence: {confidences['low']} ({confidences['low']/len(explanations)*100:.1f}%)

### Common Factors
{chr(10).join(f'- {factor}: {count} occurrences' for factor, count in common_factors)}

### Recommendations
- Review low-confidence predictions for potential issues
- Common patterns suggest {'stable model behavior' if confidences['high'] > confidences['low'] else 'potential instability'}
"""
        return report.strip()
