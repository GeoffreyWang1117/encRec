"""
LLM-based routing for recommendation.

Uses LLM to make or refine expert routing decisions based on
structured statistical prompts.

Use cases:
1. Cold-start scenarios where learned routing fails
2. Anomaly detection and handling
3. Explainable routing decisions
4. A/B testing different routing strategies
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
import json
import asyncio
from abc import ABC, abstractmethod


@dataclass
class LLMRoutingDecision:
    """Structured output from LLM routing."""
    expert_id: int
    confidence: str  # 'low', 'medium', 'high'
    reasoning: str
    fallback_expert: Optional[int] = None
    raw_response: Optional[str] = None


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate response from LLM."""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 256,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    async def generate(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert recommendation system advisor."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 256,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    async def generate(self, prompt: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class LocalLLMBackend(LLMBackend):
    """Local LLM backend using transformers."""

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: str = "cuda",
        max_new_tokens: int = 256,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
            )
        except ImportError:
            raise ImportError("transformers package required")

    async def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.3,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        return response[len(prompt):].strip()


class LLMRouter:
    """
    LLM-based router for expert selection.

    Can be used:
    1. As primary router (for interpretability)
    2. As fallback for edge cases
    3. For offline analysis
    """

    def __init__(
        self,
        backend: LLMBackend,
        prompt_builder: 'StructuredPromptBuilder',
        num_experts: int = 8,
        cache_size: int = 1000,
        async_batch_size: int = 10,
    ):
        from .prompt_builder import StructuredPromptBuilder
        self.backend = backend
        self.prompt_builder = prompt_builder
        self.num_experts = num_experts
        self.async_batch_size = async_batch_size

        # Simple LRU cache for repeated queries
        self._cache: Dict[str, LLMRoutingDecision] = {}
        self.cache_size = cache_size

    def _get_cache_key(self, sample_stats: 'SampleStatistics') -> str:
        """Generate cache key from sample statistics."""
        # Quantize statistics for caching
        return f"{int(sample_stats.head_ratio*10)}_{int(sample_stats.tail_ratio*10)}_{int(sample_stats.avg_lift*10)}"

    async def route_single(
        self,
        sample_stats: 'SampleStatistics',
        use_cache: bool = True,
    ) -> LLMRoutingDecision:
        """Route a single sample using LLM."""
        from .prompt_builder import SampleStatistics, StructuredPromptBuilder

        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(sample_stats)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Build prompt
        prompt = self.prompt_builder.build_routing_prompt(sample_stats)

        # Get LLM response
        response = await self.backend.generate(prompt)

        # Parse response
        parsed = StructuredPromptBuilder.parse_routing_response(response)

        decision = LLMRoutingDecision(
            expert_id=min(max(parsed.get('expert', 0), 0), self.num_experts - 1),
            confidence=parsed.get('confidence', 'low'),
            reasoning=parsed.get('reasoning', ''),
            raw_response=response,
        )

        # Update cache
        if use_cache:
            if len(self._cache) >= self.cache_size:
                # Remove oldest entry (simple approach)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = decision

        return decision

    async def route_batch(
        self,
        sample_stats_list: List['SampleStatistics'],
        use_cache: bool = True,
    ) -> List[LLMRoutingDecision]:
        """Route a batch of samples using LLM (with async parallelism)."""
        # Process in batches for efficiency
        results = []

        for i in range(0, len(sample_stats_list), self.async_batch_size):
            batch = sample_stats_list[i:i + self.async_batch_size]
            tasks = [
                self.route_single(stats, use_cache=use_cache)
                for stats in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        return results

    def route_sync(
        self,
        sample_stats: 'SampleStatistics',
        use_cache: bool = True,
    ) -> LLMRoutingDecision:
        """Synchronous routing (for compatibility)."""
        return asyncio.run(self.route_single(sample_stats, use_cache))


class HybridLLMRouter:
    """
    Hybrid router that combines learned routing with LLM fallback.

    Strategy:
    1. Use learned router by default (fast)
    2. Fall back to LLM for:
       - Low-confidence decisions
       - Anomalous samples
       - Cold-start scenarios
    """

    def __init__(
        self,
        learned_router: torch.nn.Module,
        llm_router: LLMRouter,
        confidence_threshold: float = 0.7,
        anomaly_detector: Optional[Callable] = None,
    ):
        self.learned_router = learned_router
        self.llm_router = llm_router
        self.confidence_threshold = confidence_threshold
        self.anomaly_detector = anomaly_detector

        # Tracking
        self.llm_fallback_count = 0
        self.total_count = 0

    def should_use_llm(
        self,
        router_confidence: float,
        sample_stats: 'SampleStatistics',
    ) -> bool:
        """Determine if LLM fallback should be used."""
        # Low confidence
        if router_confidence < self.confidence_threshold:
            return True

        # Cold start
        if sample_stats.is_cold_start:
            return True

        # Anomaly
        if self.anomaly_detector and self.anomaly_detector(sample_stats):
            return True

        return False

    def route(
        self,
        x: torch.Tensor,
        trie_features: torch.Tensor,
        sample_stats: 'SampleStatistics',
    ) -> Dict[str, Any]:
        """
        Hybrid routing decision.

        Returns dict with expert selection and metadata.
        """
        self.total_count += 1

        # Get learned router decision
        with torch.no_grad():
            router_logits = self.learned_router(trie_features)
            router_probs = torch.softmax(router_logits, dim=-1)
            confidence = router_probs.max(dim=-1).values.item()
            learned_expert = router_probs.argmax(dim=-1).item()

        result = {
            'expert_id': learned_expert,
            'confidence': confidence,
            'source': 'learned',
            'reasoning': None,
        }

        # Check if LLM fallback needed
        if self.should_use_llm(confidence, sample_stats):
            self.llm_fallback_count += 1

            # Use LLM (sync for simplicity)
            llm_decision = self.llm_router.route_sync(sample_stats)

            result.update({
                'expert_id': llm_decision.expert_id,
                'confidence': 0.8 if llm_decision.confidence == 'high' else 0.5,
                'source': 'llm',
                'reasoning': llm_decision.reasoning,
                'learned_expert': learned_expert,  # Keep for analysis
            })

        return result

    @property
    def llm_fallback_rate(self) -> float:
        """Fraction of decisions using LLM fallback."""
        if self.total_count == 0:
            return 0.0
        return self.llm_fallback_count / self.total_count
