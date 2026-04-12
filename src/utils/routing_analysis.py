"""
Routing Entropy Analysis for MoE Models.

This module provides tools to measure and visualize:
1. Routing entropy (diversity of expert usage)
2. Expert utilization distribution
3. Routing collapse detection
4. Comparison between Trie-MoE and Standard MoE

Key insight: MoE routing collapses with small data (low entropy).
Trie-MoE maintains diverse routing through statistical priors.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path


class RoutingAnalyzer:
    """
    Analyzes MoE routing behavior during training/inference.

    Tracks:
    - Per-sample routing entropy
    - Expert utilization over time
    - Alpha (prior weight) distribution
    - Routing collapse indicators
    """

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.reset()

    def reset(self):
        """Reset all collected statistics."""
        self.entropy_history = []
        self.expert_counts = np.zeros(self.num_experts)
        self.alpha_values = []
        self.per_epoch_stats = []
        self.batch_entropies = []

    def record_routing(
        self,
        router_logits: torch.Tensor,
        expert_indices: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
    ):
        """
        Record routing decisions from a batch.

        Args:
            router_logits: (batch, num_experts) routing logits
            expert_indices: (batch, top_k) selected expert indices
            alpha: (batch,) adaptive alpha values (Trie-MoE only)
        """
        with torch.no_grad():
            # Compute routing probabilities
            probs = F.softmax(router_logits, dim=-1)

            # Entropy: H = -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            self.batch_entropies.extend(entropy.cpu().numpy().tolist())
            self.entropy_history.append(entropy.mean().item())

            # Expert utilization
            if expert_indices is not None:
                for idx in expert_indices[:, 0].cpu().numpy():
                    self.expert_counts[idx] += 1

            # Alpha values (for Trie-MoE)
            if alpha is not None:
                self.alpha_values.extend(alpha.cpu().numpy().tolist())

    def get_epoch_stats(self) -> Dict[str, float]:
        """
        Compute statistics for the current epoch.

        Returns:
            Dict with entropy stats, utilization stats, etc.
        """
        stats = {}

        if self.entropy_history:
            stats['mean_entropy'] = np.mean(self.entropy_history)
            stats['std_entropy'] = np.std(self.entropy_history)
            stats['min_entropy'] = np.min(self.entropy_history)
            stats['max_entropy'] = np.max(self.entropy_history)

        if self.expert_counts.sum() > 0:
            # Normalized utilization
            utilization = self.expert_counts / self.expert_counts.sum()
            stats['expert_utilization'] = utilization.tolist()

            # Utilization entropy (how evenly experts are used)
            util_entropy = -np.sum(utilization * np.log(utilization + 1e-8))
            stats['utilization_entropy'] = util_entropy

            # Max utilization (collapse indicator)
            stats['max_expert_utilization'] = np.max(utilization)

            # Number of effectively used experts
            stats['effective_num_experts'] = np.exp(util_entropy)

        if self.alpha_values:
            stats['mean_alpha'] = np.mean(self.alpha_values)
            stats['std_alpha'] = np.std(self.alpha_values)

        self.per_epoch_stats.append(stats)
        return stats

    def is_collapsed(self, threshold: float = 0.5) -> bool:
        """
        Check if routing has collapsed.

        Collapse = most samples go to same expert
        Metric: max utilization > threshold, or entropy < log(2)
        """
        if self.expert_counts.sum() == 0:
            return False

        utilization = self.expert_counts / self.expert_counts.sum()
        max_util = np.max(utilization)

        # Check utilization
        if max_util > threshold:
            return True

        # Check entropy (log(2) ≈ 0.693 is minimum for 2 experts equally used)
        if self.entropy_history:
            avg_entropy = np.mean(self.entropy_history)
            if avg_entropy < 0.5:  # Very low entropy
                return True

        return False

    def collapse_severity(self) -> float:
        """
        Measure collapse severity (0 = diverse, 1 = fully collapsed).

        Based on effective number of experts.
        """
        if self.expert_counts.sum() == 0:
            return 0.0

        utilization = self.expert_counts / self.expert_counts.sum()
        util_entropy = -np.sum(utilization * np.log(utilization + 1e-8))
        effective_experts = np.exp(util_entropy)

        # Severity: 1 - (effective / total)
        return 1.0 - (effective_experts / self.num_experts)


def compute_routing_entropy(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute routing entropy for a batch.

    Args:
        router_logits: (batch, num_experts)

    Returns:
        (batch,) entropy values
    """
    probs = F.softmax(router_logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return entropy


def compute_expert_utilization(
    expert_indices: torch.Tensor,
    num_experts: int,
) -> np.ndarray:
    """
    Compute expert utilization from selection indices.

    Args:
        expert_indices: (batch, top_k) selected expert indices
        num_experts: Total number of experts

    Returns:
        (num_experts,) normalized utilization
    """
    counts = np.zeros(num_experts)
    for idx in expert_indices[:, 0].cpu().numpy():
        counts[idx] += 1
    return counts / counts.sum() if counts.sum() > 0 else counts


def compare_routing_entropy(
    trie_moe_model,
    standard_moe_model,
    dataloader,
    device: str = 'cuda',
) -> Dict[str, Dict[str, float]]:
    """
    Compare routing entropy between Trie-MoE and Standard MoE.

    Returns:
        Dict with 'trie_moe' and 'standard_moe' statistics
    """
    results = {
        'trie_moe': {'entropies': [], 'expert_counts': None},
        'standard_moe': {'entropies': [], 'expert_counts': None},
    }

    num_experts = trie_moe_model.num_experts

    trie_moe_model.eval()
    standard_moe_model.eval()

    trie_counts = np.zeros(num_experts)
    std_counts = np.zeros(num_experts)

    with torch.no_grad():
        for batch in dataloader:
            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)

            # Get Trie-MoE routing
            # Note: need trie_features and token_freqs for full Trie-MoE
            # For simplicity, use default values here
            trie_features = torch.zeros(dense.shape[0], 7 * sparse.shape[1] + 5).to(device)
            token_freqs = torch.ones(dense.shape[0]).to(device)

            trie_out = trie_moe_model(
                dense, sparse, trie_features, token_freqs, return_routing=True
            )

            if trie_out.get('routing_info') and trie_out['routing_info'].get('routing_entropy') is not None:
                results['trie_moe']['entropies'].extend(
                    trie_out['routing_info']['routing_entropy'].cpu().numpy().tolist()
                )
                for idx in trie_out['routing_info']['expert_indices'][:, 0].cpu().numpy():
                    trie_counts[idx] += 1

            # Get Standard MoE routing
            std_out = standard_moe_model(dense, sparse, return_routing=True)

            if std_out.get('routing_info') and std_out['routing_info'].get('routing_entropy') is not None:
                results['standard_moe']['entropies'].extend(
                    std_out['routing_info']['routing_entropy'].cpu().numpy().tolist()
                )
                for idx in std_out['routing_info']['expert_indices'][:, 0].cpu().numpy():
                    std_counts[idx] += 1

    # Compute statistics
    for name, counts in [('trie_moe', trie_counts), ('standard_moe', std_counts)]:
        results[name]['expert_counts'] = counts.tolist()
        if counts.sum() > 0:
            util = counts / counts.sum()
            results[name]['utilization_entropy'] = -np.sum(util * np.log(util + 1e-8))
            results[name]['effective_experts'] = np.exp(results[name]['utilization_entropy'])
            results[name]['max_utilization'] = np.max(util)

        if results[name]['entropies']:
            results[name]['mean_entropy'] = np.mean(results[name]['entropies'])
            results[name]['std_entropy'] = np.std(results[name]['entropies'])

    return results


def plot_routing_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "Routing Entropy Comparison",
):
    """
    Plot comparison of routing entropy between models.

    Args:
        results: Output from compare_routing_entropy
        save_path: Optional path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Entropy distribution
    ax1 = axes[0]
    for name, color in [('trie_moe', 'blue'), ('standard_moe', 'red')]:
        if results[name]['entropies']:
            ax1.hist(
                results[name]['entropies'],
                bins=50,
                alpha=0.5,
                label=name,
                color=color,
            )
    ax1.set_xlabel('Routing Entropy')
    ax1.set_ylabel('Count')
    ax1.set_title('Entropy Distribution')
    ax1.legend()

    # 2. Expert utilization
    ax2 = axes[1]
    x = np.arange(len(results['trie_moe']['expert_counts']))
    width = 0.35

    trie_util = np.array(results['trie_moe']['expert_counts'])
    std_util = np.array(results['standard_moe']['expert_counts'])

    if trie_util.sum() > 0:
        trie_util = trie_util / trie_util.sum()
    if std_util.sum() > 0:
        std_util = std_util / std_util.sum()

    ax2.bar(x - width/2, trie_util, width, label='Trie-MoE', color='blue', alpha=0.7)
    ax2.bar(x + width/2, std_util, width, label='Standard MoE', color='red', alpha=0.7)
    ax2.set_xlabel('Expert Index')
    ax2.set_ylabel('Utilization')
    ax2.set_title('Expert Utilization')
    ax2.legend()

    # 3. Summary statistics
    ax3 = axes[2]
    metrics = ['mean_entropy', 'utilization_entropy', 'effective_experts']
    metric_names = ['Mean Entropy', 'Util. Entropy', 'Effective Experts']

    trie_vals = [results['trie_moe'].get(m, 0) for m in metrics]
    std_vals = [results['standard_moe'].get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    ax3.bar(x - width/2, trie_vals, width, label='Trie-MoE', color='blue', alpha=0.7)
    ax3.bar(x + width/2, std_vals, width, label='Standard MoE', color='red', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_names)
    ax3.set_title('Routing Quality Metrics')
    ax3.legend()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def analyze_routing_vs_data_size(
    model_class,
    model_kwargs: Dict,
    data_sizes: List[int],
    dataloader_factory,
    num_epochs: int = 5,
    device: str = 'cuda',
) -> Dict[str, List[float]]:
    """
    Analyze how routing entropy changes with training data size.

    This is key evidence for the routing collapse phenomenon.

    Args:
        model_class: Model class to instantiate
        model_kwargs: Arguments for model construction
        data_sizes: List of data sizes to test (e.g., [1000, 5000, 10000, 50000])
        dataloader_factory: Function(size) -> (train_loader, val_loader)
        num_epochs: Training epochs per data size
        device: Device to use

    Returns:
        Dict with 'data_sizes', 'mean_entropy', 'collapse_severity'
    """
    results = {
        'data_sizes': data_sizes,
        'mean_entropy': [],
        'utilization_entropy': [],
        'collapse_severity': [],
    }

    for size in data_sizes:
        print(f"Testing with {size} samples...")

        # Create model
        model = model_class(**model_kwargs).to(device)

        # Create dataloaders
        train_loader, val_loader = dataloader_factory(size)

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                dense = batch['dense'].to(device)
                sparse = batch['sparse'].to(device)
                labels = batch['label'].to(device)

                # Forward (with default Trie features if needed)
                if hasattr(model, 'trie_encoder'):
                    trie_features = torch.zeros(
                        dense.shape[0], 7 * sparse.shape[1] + 5
                    ).to(device)
                    token_freqs = torch.ones(dense.shape[0]).to(device)
                    outputs = model(dense, sparse, trie_features, token_freqs)
                else:
                    outputs = model(dense, sparse)

                loss = model.compute_loss(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate routing
        analyzer = RoutingAnalyzer(model.num_experts)
        model.eval()

        with torch.no_grad():
            for batch in val_loader:
                dense = batch['dense'].to(device)
                sparse = batch['sparse'].to(device)

                if hasattr(model, 'trie_encoder'):
                    trie_features = torch.zeros(
                        dense.shape[0], 7 * sparse.shape[1] + 5
                    ).to(device)
                    token_freqs = torch.ones(dense.shape[0]).to(device)
                    outputs = model(dense, sparse, trie_features, token_freqs, return_routing=True)
                else:
                    outputs = model(dense, sparse, return_routing=True)

                if 'routing_info' in outputs:
                    ri = outputs['routing_info']
                    # Reconstruct logits from indices and weights for entropy
                    # Use entropy directly if available
                    if ri.get('routing_entropy') is not None:
                        for e in ri['routing_entropy'].cpu().numpy():
                            analyzer.entropy_history.append(e)
                    if ri.get('expert_indices') is not None:
                        for idx in ri['expert_indices'][:, 0].cpu().numpy():
                            analyzer.expert_counts[idx] += 1

        stats = analyzer.get_epoch_stats()
        results['mean_entropy'].append(stats.get('mean_entropy', 0))
        results['utilization_entropy'].append(stats.get('utilization_entropy', 0))
        results['collapse_severity'].append(analyzer.collapse_severity())

        print(f"  Mean entropy: {stats.get('mean_entropy', 0):.3f}, "
              f"Collapse severity: {analyzer.collapse_severity():.3f}")

    return results


def plot_entropy_vs_data_size(
    trie_results: Dict,
    standard_results: Dict,
    save_path: Optional[str] = None,
):
    """
    Plot routing entropy vs training data size for both models.

    This is a key figure for the paper showing routing collapse.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 1. Mean entropy
    ax1 = axes[0]
    ax1.plot(
        trie_results['data_sizes'],
        trie_results['mean_entropy'],
        'bo-',
        label='Trie-MoE',
        linewidth=2,
        markersize=8,
    )
    ax1.plot(
        standard_results['data_sizes'],
        standard_results['mean_entropy'],
        'rs--',
        label='Standard MoE',
        linewidth=2,
        markersize=8,
    )
    ax1.set_xlabel('Training Data Size')
    ax1.set_ylabel('Mean Routing Entropy')
    ax1.set_title('Routing Entropy vs Data Size')
    ax1.legend()
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # Add collapse threshold
    ax1.axhline(y=0.5, color='gray', linestyle=':', label='Collapse threshold')

    # 2. Collapse severity
    ax2 = axes[1]
    ax2.plot(
        trie_results['data_sizes'],
        trie_results['collapse_severity'],
        'bo-',
        label='Trie-MoE',
        linewidth=2,
        markersize=8,
    )
    ax2.plot(
        standard_results['data_sizes'],
        standard_results['collapse_severity'],
        'rs--',
        label='Standard MoE',
        linewidth=2,
        markersize=8,
    )
    ax2.set_xlabel('Training Data Size')
    ax2.set_ylabel('Collapse Severity')
    ax2.set_title('Routing Collapse vs Data Size')
    ax2.legend()
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
