#!/usr/bin/env python3
"""
LLM Routing Experiment.

Tests LLM-based routing for Trie-MoE:
1. LLM routing accuracy vs learned routing
2. LLM routing latency analysis
3. Hybrid routing (learned + LLM fallback)
4. Qualitative analysis of LLM explanations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import argparse
import asyncio
import time
import torch
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.data.amazon_loader import load_amazon_data
from src.trie.builder import TrieBuilder, StatisticalTrie
from src.trie.encoder import TrieEncoder
from src.llm.prompt_builder import StructuredPromptBuilder, SampleStatistics
from src.llm.llm_router import OpenAIBackend, LLMRouter, LLMRoutingDecision


def build_trie(dataset, sparse_fields, num_experts=8, strategy='frequency_aware'):
    """Build Trie with specified parameters."""
    builder = TrieBuilder(
        hierarchy_config=['frequency', 'info'],
        num_experts=num_experts,
        expert_strategy=strategy,
    )

    for field in sparse_fields:
        if field in dataset.freq_stats:
            freq_dict = dataset.freq_stats[field]
            for token, count in freq_dict.items():
                for _ in range(min(count, 100)):
                    builder.statistics.update(field, token, label=1)

    builder.statistics.compute_statistics()

    for field in sparse_fields:
        if field in builder.statistics.statistics:
            trie = StatisticalTrie(field_name=field, hierarchy_config=['frequency', 'info'])
            trie.build(builder.statistics.statistics[field])
            trie.assign_experts(num_experts, strategy)
            builder.tries[field] = trie

    return builder


def compute_sample_statistics(
    sample_idx: int,
    dataset,
    trie_builder: TrieBuilder,
    sparse_fields: list,
) -> SampleStatistics:
    """Compute statistics for a single sample."""
    sample = dataset[sample_idx]
    sparse_features = sample['sparse']

    # Collect statistics across all sparse fields
    head_count = 0
    mid_count = 0
    tail_count = 0
    high_info_count = 0
    total_count = 0
    lifts = []
    paths = set()
    expert_votes = defaultdict(int)

    for i, field in enumerate(sparse_fields):
        if field not in trie_builder.tries:
            continue

        trie = trie_builder.tries[field]
        token_idx = sparse_features[i].item()

        # Get token info from trie
        token_str = str(token_idx)
        if token_str in trie.token_to_node:
            node = trie.token_to_node[token_str]
            path = node.path if hasattr(node, 'path') else str(node.depth)
            paths.add(path)

            # Parse path for bucket based on node properties
            if hasattr(node, 'bucket'):
                if node.bucket == 'head':
                    head_count += 1
                elif node.bucket == 'tail':
                    tail_count += 1
                else:
                    mid_count += 1
            else:
                # Infer from path string or depth
                mid_count += 1

            # Check high info based on node stats
            if hasattr(node, 'stats') and node.stats.get('lift', 1.0) > 1.5:
                high_info_count += 1

            # Get expert assignment
            expert_id = trie.get_expert_for_token(token_str)
            if expert_id is not None:
                expert_votes[expert_id] += 1

            # Get lift if available
            if hasattr(node, 'stats') and 'lift' in node.stats:
                lifts.append(node.stats['lift'])
        else:
            # Unknown token -> tail
            tail_count += 1

        total_count += 1

    # Compute ratios
    if total_count == 0:
        total_count = 1  # Avoid division by zero

    head_ratio = head_count / total_count
    mid_ratio = mid_count / total_count
    tail_ratio = tail_count / total_count
    high_info_ratio = high_info_count / total_count

    avg_lift = np.mean(lifts) if lifts else 1.0
    max_lift = max(lifts) if lifts else 1.0

    # Path entropy
    path_counts = [1] * len(paths) if paths else [1]
    path_probs = np.array(path_counts) / sum(path_counts)
    path_entropy = -np.sum(path_probs * np.log(path_probs + 1e-10))

    # Suggested experts
    if expert_votes:
        sorted_experts = sorted(expert_votes.items(), key=lambda x: -x[1])
        suggested_experts = [e[0] for e in sorted_experts[:3]]
        top_votes = sorted_experts[0][1]
        expert_confidence = top_votes / total_count
    else:
        suggested_experts = [0]
        expert_confidence = 0.5

    return SampleStatistics(
        head_ratio=head_ratio,
        mid_ratio=mid_ratio,
        tail_ratio=tail_ratio,
        high_info_ratio=high_info_ratio,
        avg_lift=avg_lift,
        max_lift=max_lift,
        num_unique_paths=len(paths),
        path_entropy=path_entropy,
        suggested_experts=suggested_experts,
        expert_confidence=expert_confidence,
        is_cold_start=tail_ratio > 0.8,
        is_high_variance=len(paths) > 5,
    )


async def test_llm_routing(
    llm_router: LLMRouter,
    test_samples: list,
    dataset,
    trie_builder: TrieBuilder,
    sparse_fields: list,
    num_samples: int = 50,
):
    """Test LLM routing on a subset of samples."""
    results = []

    # Select diverse samples
    sample_indices = np.random.choice(len(test_samples), min(num_samples, len(test_samples)), replace=False)

    print(f"\nTesting LLM routing on {len(sample_indices)} samples...")

    for idx in tqdm(sample_indices):
        sample_stats = compute_sample_statistics(idx, dataset, trie_builder, sparse_fields)

        start_time = time.time()
        try:
            decision = await llm_router.route_single(sample_stats, use_cache=True)
            latency = (time.time() - start_time) * 1000  # ms

            results.append({
                'sample_idx': int(idx),
                'expert_id': decision.expert_id,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'latency_ms': latency,
                'trie_suggested': sample_stats.suggested_experts[0] if sample_stats.suggested_experts else 0,
                'head_ratio': sample_stats.head_ratio,
                'tail_ratio': sample_stats.tail_ratio,
                'is_cold_start': sample_stats.is_cold_start,
                'success': True,
            })
        except Exception as e:
            results.append({
                'sample_idx': int(idx),
                'error': str(e),
                'success': False,
            })

    return results


def analyze_results(results: list) -> dict:
    """Analyze LLM routing results."""
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    if not successful:
        return {'error': 'No successful routing decisions'}

    # Latency analysis
    latencies = [r['latency_ms'] for r in successful]

    # Agreement with trie suggestion
    agreements = [r['expert_id'] == r['trie_suggested'] for r in successful]

    # Confidence distribution
    confidence_counts = defaultdict(int)
    for r in successful:
        confidence_counts[r['confidence']] += 1

    # Cold start handling
    cold_start_samples = [r for r in successful if r.get('is_cold_start', False)]

    # Expert distribution
    expert_counts = defaultdict(int)
    for r in successful:
        expert_counts[r['expert_id']] += 1

    analysis = {
        'total_samples': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'latency': {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
        },
        'trie_agreement_rate': np.mean(agreements),
        'confidence_distribution': dict(confidence_counts),
        'expert_distribution': dict(expert_counts),
        'cold_start_samples': len(cold_start_samples),
    }

    return analysis


def print_sample_explanations(results: list, num_examples: int = 5):
    """Print example LLM explanations."""
    successful = [r for r in results if r.get('success', False) and r.get('reasoning')]

    print(f"\n{'='*60}")
    print("EXAMPLE LLM ROUTING EXPLANATIONS")
    print('='*60)

    for i, r in enumerate(successful[:num_examples]):
        print(f"\nSample {i+1}:")
        print(f"  Head ratio: {r['head_ratio']:.2%}, Tail ratio: {r['tail_ratio']:.2%}")
        print(f"  Trie suggested: Expert {r['trie_suggested']}")
        print(f"  LLM selected: Expert {r['expert_id']} ({r['confidence']} confidence)")
        print(f"  Reasoning: {r['reasoning']}")
        print(f"  Latency: {r['latency_ms']:.1f}ms")


async def main():
    parser = argparse.ArgumentParser(description='LLM Routing Experiment')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to test')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model to use')
    parser.add_argument('--num_experts', type=int, default=8, help='Number of experts')
    args = parser.parse_args()

    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        print("Please set it in .env file or environment variable")
        return

    print(f"Using OpenAI model: {args.model}")
    print(f"API key: {api_key[:10]}...{api_key[-4:]}")

    # Load data
    print("\nLoading data...")
    train_dataset, val_dataset, test_dataset = load_amazon_data(
        reviews_path='data/amazon/electronics_encrypted.parquet',
        mode='encrypted',
    )
    print(f"Test dataset size: {len(test_dataset)}")

    # Get sparse fields
    sparse_fields = test_dataset.actual_sparse_cols
    print(f"Sparse fields: {sparse_fields}")

    # Build Trie
    print("\nBuilding Trie structure...")
    trie_builder = build_trie(train_dataset, sparse_fields, num_experts=args.num_experts)

    # Initialize LLM components
    print("\nInitializing LLM router...")
    backend = OpenAIBackend(
        model=args.model,
        api_key=api_key,
        temperature=0.3,
        max_tokens=256,
    )

    prompt_builder = StructuredPromptBuilder(num_experts=args.num_experts)

    llm_router = LLMRouter(
        backend=backend,
        prompt_builder=prompt_builder,
        num_experts=args.num_experts,
        cache_size=1000,
        async_batch_size=5,
    )

    # Run experiment
    print(f"\nRunning LLM routing experiment with {args.num_samples} samples...")
    results = await test_llm_routing(
        llm_router=llm_router,
        test_samples=list(range(len(test_dataset))),
        dataset=test_dataset,
        trie_builder=trie_builder,
        sparse_fields=sparse_fields,
        num_samples=args.num_samples,
    )

    # Analyze results
    analysis = analyze_results(results)

    # Print summary
    print(f"\n{'='*60}")
    print("LLM ROUTING EXPERIMENT RESULTS")
    print('='*60)
    print(f"Total samples: {analysis['total_samples']}")
    print(f"Successful: {analysis['successful']}")
    print(f"Failed: {analysis['failed']}")
    print(f"\nLatency Statistics:")
    print(f"  Mean: {analysis['latency']['mean_ms']:.1f}ms")
    print(f"  Std: {analysis['latency']['std_ms']:.1f}ms")
    print(f"  P50: {analysis['latency']['p50_ms']:.1f}ms")
    print(f"  P95: {analysis['latency']['p95_ms']:.1f}ms")
    print(f"\nTrie Agreement Rate: {analysis['trie_agreement_rate']:.2%}")
    print(f"Confidence Distribution: {analysis['confidence_distribution']}")
    print(f"Expert Distribution: {analysis['expert_distribution']}")
    print(f"Cold Start Samples: {analysis['cold_start_samples']}")

    # Print example explanations
    print_sample_explanations(results)

    # Save results
    output = {
        'config': {
            'model': args.model,
            'num_samples': args.num_samples,
            'num_experts': args.num_experts,
        },
        'analysis': analysis,
        'sample_results': results[:20],  # Save first 20 detailed results
    }

    output_path = Path('logs') / f'llm_routing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    asyncio.run(main())
