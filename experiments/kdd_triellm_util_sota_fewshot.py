#!/usr/bin/env python3
"""
SOTA Baseline Comparison in Few-shot Scenarios.

Compare Trie-MoE with state-of-the-art CTR models in low-data settings.
This is a critical experiment for paper acceptance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime
from scipy import stats

from src.data.criteo_loader import load_criteo_data, SPARSE_COLS
from src.models.backbone import DeepFM, DLRM, DCNv2, AutoInt, FinalMLP, DCNv3
from src.models.moe import TrieMoERecommender
from src.trie.builder import TrieBuilder, StatisticalTrie
from src.trie.fast_encoder import FastTrieEncoder
from src.utils.metrics import compute_metrics


# PLE Implementation (simplified)
class PLE(nn.Module):
    """Progressive Layered Extraction for multi-task learning."""

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: dict,
        embedding_dim: int = 16,
        num_tasks: int = 1,
        num_experts_shared: int = 2,
        num_experts_task: int = 2,
        expert_hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.num_experts_shared = num_experts_shared
        self.num_experts_task = num_experts_task

        # Embeddings
        self.embeddings = nn.ModuleDict({
            field: nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            for field, vocab_size in sparse_dims.items()
        })
        self.field_names = list(sparse_dims.keys())
        self.dense_bn = nn.BatchNorm1d(dense_dim)

        input_dim = dense_dim + len(sparse_dims) * embedding_dim

        # Expert networks for each layer
        self.shared_experts = nn.ModuleList()
        self.task_experts = nn.ModuleList()
        self.gates = nn.ModuleList()

        for layer in range(num_layers):
            layer_input_dim = input_dim if layer == 0 else expert_hidden_dim

            # Shared experts
            shared = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(layer_input_dim, expert_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts_shared)
            ])
            self.shared_experts.append(shared)

            # Task-specific experts
            task = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(layer_input_dim, expert_hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    )
                    for _ in range(num_experts_task)
                ])
                for _ in range(num_tasks)
            ])
            self.task_experts.append(task)

            # Gates
            total_experts = num_experts_shared + num_experts_task
            gate = nn.ModuleList([
                nn.Linear(layer_input_dim, total_experts)
                for _ in range(num_tasks)
            ])
            self.gates.append(gate)

        # Tower networks
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, dense, sparse, **kwargs):
        batch_size = dense.shape[0]

        # Embeddings
        dense = self.dense_bn(dense)
        emb_list = [self.embeddings[f](sparse[:, i]) for i, f in enumerate(self.field_names)]
        sparse_emb = torch.cat(emb_list, dim=1)

        x = torch.cat([dense, sparse_emb], dim=1)

        # Task inputs
        task_inputs = [x for _ in range(self.num_tasks)]

        for layer in range(self.num_layers):
            task_outputs = []

            for task_id in range(self.num_tasks):
                # Get expert outputs
                shared_outputs = [expert(task_inputs[task_id]) for expert in self.shared_experts[layer]]
                task_expert_outputs = [expert(task_inputs[task_id]) for expert in self.task_experts[layer][task_id]]

                expert_outputs = torch.stack(shared_outputs + task_expert_outputs, dim=1)

                # Gate
                gate_input = task_inputs[task_id]
                gate_weights = torch.softmax(self.gates[layer][task_id](gate_input), dim=-1)

                # Weighted sum
                output = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
                task_outputs.append(output)

            task_inputs = task_outputs

        # Tower outputs
        logits = self.towers[0](task_inputs[0]).squeeze(-1)

        return {'logits': logits}


def build_trie(dataset, sparse_fields, num_experts=8):
    """Build Trie with specified parameters."""
    builder = TrieBuilder(
        hierarchy_config=['frequency', 'info'],
        num_experts=num_experts,
        expert_strategy='frequency_aware',
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
            trie.assign_experts(num_experts, 'frequency_aware')
            builder.tries[field] = trie
    return builder


def train_and_evaluate(model, train_loader, val_loader, test_loader, device,
                       epochs=10, lr=0.001, use_trie=False, trie_encoder=None,
                       sparse_fields=None, vocab_maps=None):
    """Train and evaluate a model."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_auc = 0
    best_model_state = None
    train_time = 0

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()

        for batch in train_loader:
            optimizer.zero_grad()

            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)
            labels = batch['label'].to(device)

            if use_trie and trie_encoder is not None:
                trie_features = trie_encoder(sparse, sparse_fields, vocab_maps)
                output = model(dense, sparse, trie_routing_vec=trie_features)
            else:
                output = model(dense, sparse)

            logits = output['logits'] if isinstance(output, dict) else output
            loss = criterion(logits.squeeze(), labels.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_time += time.time() - epoch_start

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                dense = batch['dense'].to(device)
                sparse = batch['sparse'].to(device)

                if use_trie and trie_encoder is not None:
                    trie_features = trie_encoder(sparse, sparse_fields, vocab_maps)
                    output = model(dense, sparse, trie_routing_vec=trie_features)
                else:
                    output = model(dense, sparse)

                logits = output['logits'] if isinstance(output, dict) else output
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(batch['label'].numpy())

        val_metrics = compute_metrics(np.array(val_preds), np.array(val_labels))
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Test evaluation
    model.eval()
    test_preds, test_labels = [], []
    inference_start = time.time()

    with torch.no_grad():
        for batch in test_loader:
            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)

            if use_trie and trie_encoder is not None:
                trie_features = trie_encoder(sparse, sparse_fields, vocab_maps)
                output = model(dense, sparse, trie_routing_vec=trie_features)
            else:
                output = model(dense, sparse)

            logits = output['logits'] if isinstance(output, dict) else output
            test_preds.extend(torch.sigmoid(logits).cpu().numpy())
            test_labels.extend(batch['label'].numpy())

    inference_time = time.time() - inference_start

    test_metrics = compute_metrics(np.array(test_preds), np.array(test_labels))

    return {
        'test_auc': test_metrics['auc'],
        'test_logloss': test_metrics['logloss'],
        'best_val_auc': best_val_auc,
        'train_time': train_time,
        'inference_time': inference_time,
        'num_params': sum(p.numel() for p in model.parameters()),
    }


def run_experiment(data_size, device, num_runs=5, epochs=10):
    """Run SOTA comparison experiment."""
    print(f"\n{'='*70}")
    print(f"SOTA COMPARISON: {data_size:,} samples")
    print(f"{'='*70}")

    # Load data
    print(f"Loading Criteo data ({data_size:,} samples)...")
    train_dataset, val_dataset, test_dataset = load_criteo_data(
        'data/criteo/criteo_5m.parquet',
        sample_size=data_size,
    )

    sparse_fields = SPARSE_COLS
    feature_dims = train_dataset.get_feature_dims()
    vocab_maps = {f: {i: t for t, i in v.items()} for f, v in train_dataset.vocab.items()}
    vocab_sizes = {f: len(v) for f, v in train_dataset.vocab.items()}

    print(f"  Train/Val/Test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    # Data loaders
    batch_size = min(512, data_size // 10)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4
    )

    # Build Trie
    print("Building Trie...")
    trie_builder = build_trie(train_dataset, sparse_fields)
    trie_encoder = FastTrieEncoder(
        tries=trie_builder.tries,
        vocab_sizes=vocab_sizes,
    ).to(device)

    # Model configurations
    models_config = {
        'DeepFM': {
            'class': DeepFM,
            'kwargs': {
                'dense_dim': feature_dims['dense_dim'],
                'sparse_dims': feature_dims['sparse_dims'],
                'embedding_dim': 16,
                'dropout': 0.2,
            },
            'use_trie': False,
        },
        'DLRM': {
            'class': DLRM,
            'kwargs': {
                'dense_dim': feature_dims['dense_dim'],
                'sparse_dims': feature_dims['sparse_dims'],
                'embedding_dim': 16,
                'dropout': 0.2,
            },
            'use_trie': False,
        },
        'DCNv2': {
            'class': DCNv2,
            'kwargs': {
                'dense_dim': feature_dims['dense_dim'],
                'sparse_dims': feature_dims['sparse_dims'],
                'embedding_dim': 16,
                'num_cross_layers': 2,
                'dropout': 0.2,
            },
            'use_trie': False,
        },
        'AutoInt': {
            'class': AutoInt,
            'kwargs': {
                'dense_dim': feature_dims['dense_dim'],
                'sparse_dims': feature_dims['sparse_dims'],
                'embedding_dim': 16,
                'num_heads': 2,
                'num_layers': 2,
                'dropout': 0.2,
            },
            'use_trie': False,
        },
        'FinalMLP': {
            'class': FinalMLP,
            'kwargs': {
                'dense_dim': feature_dims['dense_dim'],
                'sparse_dims': feature_dims['sparse_dims'],
                'embedding_dim': 16,
                'dropout': 0.2,
            },
            'use_trie': False,
        },
        'PLE': {
            'class': PLE,
            'kwargs': {
                'dense_dim': feature_dims['dense_dim'],
                'sparse_dims': feature_dims['sparse_dims'],
                'embedding_dim': 16,
                'dropout': 0.2,
            },
            'use_trie': False,
        },
        'Trie-MoE': {
            'class': TrieMoERecommender,
            'kwargs': {
                'dense_dim': feature_dims['dense_dim'],
                'sparse_dims': feature_dims['sparse_dims'],
                'embedding_dim': 16,
                'num_experts': 4,
                'top_k': 1,
                'routing_mode': 'learned_only',
                'dropout': 0.2,
            },
            'use_trie': True,
        },
    }

    results = {name: {'aucs': [], 'times': [], 'params': None} for name in models_config}

    for model_name, config in models_config.items():
        print(f"\n--- {model_name} ---")

        for run in range(num_runs):
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)

            try:
                model = config['class'](**config['kwargs']).to(device)

                result = train_and_evaluate(
                    model, train_loader, val_loader, test_loader, device,
                    epochs=epochs, lr=0.001,
                    use_trie=config['use_trie'],
                    trie_encoder=trie_encoder if config['use_trie'] else None,
                    sparse_fields=sparse_fields if config['use_trie'] else None,
                    vocab_maps=vocab_maps if config['use_trie'] else None,
                )

                results[model_name]['aucs'].append(result['test_auc'])
                results[model_name]['times'].append(result['train_time'])
                results[model_name]['params'] = result['num_params']

                print(f"  Run {run+1}: AUC={result['test_auc']:.4f}")

            except Exception as e:
                print(f"  Run {run+1}: Error - {e}")
                continue

    # Compute statistics
    summary = {}
    for model_name, r in results.items():
        if r['aucs']:
            summary[model_name] = {
                'auc_mean': np.mean(r['aucs']),
                'auc_std': np.std(r['aucs']),
                'time_mean': np.mean(r['times']),
                'params': r['params'],
                'runs': r['aucs'],
            }

    return summary


def main():
    print("=" * 70)
    print("SOTA BASELINE COMPARISON IN FEW-SHOT SCENARIOS")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test on few-shot scenarios
    data_sizes = [50000, 100000]  # Focus on few-shot

    all_results = {}

    for data_size in data_sizes:
        result = run_experiment(data_size, device, num_runs=5, epochs=10)
        all_results[data_size] = result

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for data_size in data_sizes:
        print(f"\n--- {data_size:,} samples ---")
        print(f"{'Model':<15} {'AUC (mean±std)':<20} {'Train Time':<12} {'Params':<12}")
        print("-" * 60)

        sorted_models = sorted(all_results[data_size].items(),
                              key=lambda x: -x[1]['auc_mean'])

        best_auc = sorted_models[0][1]['auc_mean']

        for rank, (model_name, r) in enumerate(sorted_models, 1):
            auc_str = f"{r['auc_mean']:.4f}±{r['auc_std']:.4f}"
            diff = r['auc_mean'] - best_auc
            diff_str = f"({diff:+.4f})" if diff < 0 else "(best)"

            print(f"{model_name:<15} {auc_str:<20} {r['time_mean']:.1f}s        {r['params']:,}")

    # Statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE (Trie-MoE vs Others)")
    print("=" * 70)

    for data_size in data_sizes:
        print(f"\n--- {data_size:,} samples ---")
        trie_moe_aucs = all_results[data_size]['Trie-MoE']['runs']

        for model_name, r in all_results[data_size].items():
            if model_name == 'Trie-MoE':
                continue

            other_aucs = r['runs']
            if len(other_aucs) >= 3 and len(trie_moe_aucs) >= 3:
                t_stat, p_value = stats.ttest_rel(trie_moe_aucs[:min(len(trie_moe_aucs), len(other_aucs))],
                                                   other_aucs[:min(len(trie_moe_aucs), len(other_aucs))])
                diff = np.mean(trie_moe_aucs) - np.mean(other_aucs)
                sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))
                print(f"  vs {model_name:<12}: diff={diff:+.4f}, p={p_value:.4f} {sig}")

    # Save results
    results_path = Path('results/sota_comparison')
    results_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_results = {}
    for size, size_results in all_results.items():
        json_results[str(size)] = {
            model: {
                'auc_mean': float(r['auc_mean']),
                'auc_std': float(r['auc_std']),
                'time_mean': float(r['time_mean']),
                'params': r['params'],
                'runs': [float(x) for x in r['runs']],
            }
            for model, r in size_results.items()
        }

    with open(results_path / f'results_{timestamp}.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {results_path / f'results_{timestamp}.json'}")


if __name__ == '__main__':
    main()
