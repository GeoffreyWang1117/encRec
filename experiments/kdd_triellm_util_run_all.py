#!/usr/bin/env python3
"""
KDD 2026 综合实验脚本

运行所有KDD论文需要的实验：
1. 跨数据集验证 (Criteo, Amazon, MovieLens)
2. 数据规模消融 (10K-500K)
3. 自适应Alpha分析
4. 效率测试

用法: python experiments/run_all_kdd_experiments.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from scipy import stats
from collections import defaultdict

# 项目模块
from src.data.criteo_loader import load_criteo_data, CriteoDataset, SPARSE_COLS
from src.data.movielens_loader import load_movielens_data
from src.models.backbone import DeepFM
from src.models.moe import TrieMoERecommender
from src.trie.builder import TrieBuilder, StatisticalTrie
from src.trie.fast_encoder import FastTrieEncoder
from src.utils.metrics import compute_metrics


class ExperimentRunner:
    """综合实验运行器"""

    def __init__(self, output_dir='results/kdd_experiments'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results = {}

        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")

    def build_trie(self, dataset, sparse_fields, num_experts=8):
        """构建Trie结构"""
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

    def train_model(self, model, train_loader, val_loader, epochs=10, lr=0.001,
                   use_trie=False, trie_encoder=None, sparse_fields=None, vocab_maps=None):
        """训练模型"""
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        # 移动trie_encoder到device
        if trie_encoder is not None:
            trie_encoder = trie_encoder.to(self.device)

        best_val_auc = 0
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            num_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()

                dense = batch['dense'].to(self.device)
                sparse = batch['sparse'].to(self.device)
                labels = batch['label'].to(self.device)

                if use_trie and trie_encoder is not None:
                    # FastTrieEncoder直接接收sparse indices
                    trie_features = trie_encoder(sparse)
                    output = model(dense, sparse, trie_routing_vec=trie_features)
                else:
                    output = model(dense, sparse)

                logits = output['logits'] if isinstance(output, dict) else output
                loss = criterion(logits.squeeze(), labels.float())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # 简化：不做验证early stopping，直接训练完

        return model

    def evaluate_model(self, model, data_loader, use_trie=False, trie_encoder=None,
                      sparse_fields=None, vocab_maps=None):
        """评估模型"""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                dense = batch['dense'].to(self.device)
                sparse = batch['sparse'].to(self.device)

                if use_trie and trie_encoder is not None:
                    # FastTrieEncoder直接接收sparse indices
                    trie_features = trie_encoder(sparse)
                    output = model(dense, sparse, trie_routing_vec=trie_features)
                else:
                    output = model(dense, sparse)

                logits = output['logits'] if isinstance(output, dict) else output
                preds = torch.sigmoid(logits).cpu().numpy()

                all_preds.extend(preds.flatten())
                all_labels.extend(batch['label'].numpy().flatten())

        # 注意：compute_metrics的参数顺序是 (predictions, labels)
        return compute_metrics(np.array(all_preds), np.array(all_labels))

    def run_single_experiment(self, dataset_name, data_size, model_name, seed=42):
        """运行单次实验"""
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 加载数据
        if dataset_name == 'criteo':
            train_ds, val_ds, test_ds = load_criteo_data(
                'data/criteo/criteo_5m.parquet',
                sample_size=data_size
            )
            sparse_fields = SPARSE_COLS
            feature_dims = train_ds.get_feature_dims()
            dense_dim = feature_dims['dense_dim']
            sparse_dims = feature_dims['sparse_dims']  # Dict[str, int]

        elif dataset_name == 'movielens':
            train_ds, val_ds, test_ds = load_movielens_data(
                'data/movielens',
                sample_size=data_size
            )
            feature_dims = train_ds.get_feature_dims()
            sparse_fields = train_ds.actual_sparse_cols
            dense_dim = feature_dims['dense_dim']
            sparse_dims = feature_dims['sparse_dims']  # Dict[str, int]

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # 创建DataLoader
        batch_size = min(512, max(32, data_size // 20))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        print(f"    Data splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

        # 训练计时
        start_time = time.time()

        if model_name == 'DeepFM':
            model = DeepFM(
                dense_dim=dense_dim,
                sparse_dims=sparse_dims,  # Dict[str, int]
                embedding_dim=32,
                hidden_dims=[256, 128, 64],
            ).to(self.device)
            model = self.train_model(model, train_loader, val_loader, epochs=10, lr=0.001)
            test_metrics = self.evaluate_model(model, test_loader)

        elif model_name == 'Trie-MoE':
            # 构建Trie
            trie_builder = self.build_trie(train_ds, sparse_fields, num_experts=8)

            # 创建vocab_sizes和tries字典
            vocab_sizes = sparse_dims  # 直接使用sparse_dims

            # 创建FastTrieEncoder
            trie_encoder = FastTrieEncoder(
                tries=trie_builder.tries,
                vocab_sizes=vocab_sizes,
                routing_dim=32,
            )

            model = TrieMoERecommender(
                dense_dim=dense_dim,
                sparse_dims=sparse_dims,  # Dict[str, int]
                embedding_dim=32,
                trie_routing_dim=32,
                num_experts=8,
                top_k=2,
                hidden_dims=[128, 64],
            ).to(self.device)

            model = self.train_model(
                model, train_loader, val_loader, epochs=10, lr=0.001,
                use_trie=True, trie_encoder=trie_encoder,
                sparse_fields=sparse_fields, vocab_maps=None
            )
            test_metrics = self.evaluate_model(
                model, test_loader,
                use_trie=True, trie_encoder=trie_encoder,
                sparse_fields=sparse_fields, vocab_maps=None
            )

        else:
            raise ValueError(f"Unknown model: {model_name}")

        train_time = time.time() - start_time

        return {
            'auc': test_metrics['auc'],
            'logloss': test_metrics['logloss'],
            'train_time': train_time,
        }

    def run_data_scale_experiment(self, dataset_name='criteo', num_runs=5):
        """数据规模消融实验"""
        print("\n" + "="*60)
        print(f"DATA SCALE ABLATION: {dataset_name}")
        print("="*60)

        data_sizes = [10000, 20000, 50000, 100000, 200000]
        models = ['DeepFM', 'Trie-MoE']

        results = {}

        for size in data_sizes:
            print(f"\n--- Data size: {size} ---")
            results[size] = {}

            for model_name in models:
                runs = []
                for run in range(num_runs):
                    seed = 42 + run
                    print(f"  {model_name} run {run+1}/{num_runs}...", end=' ', flush=True)

                    try:
                        metrics = self.run_single_experiment(dataset_name, size, model_name, seed)
                        runs.append(metrics)
                        print(f"AUC={metrics['auc']:.4f}")
                    except Exception as e:
                        print(f"FAILED: {e}")
                        continue

                if runs:
                    auc_values = [r['auc'] for r in runs]
                    logloss_values = [r['logloss'] for r in runs]

                    results[size][model_name] = {
                        'auc_mean': np.mean(auc_values),
                        'auc_std': np.std(auc_values),
                        'logloss_mean': np.mean(logloss_values),
                        'logloss_std': np.std(logloss_values),
                        'runs': runs,
                    }

        # 计算统计显著性
        self._compute_significance(results)

        return results

    def run_cross_dataset_experiment(self, num_runs=3):
        """跨数据集验证"""
        print("\n" + "="*60)
        print("CROSS-DATASET VALIDATION")
        print("="*60)

        datasets = ['criteo', 'movielens']
        data_size = 50000
        models = ['DeepFM', 'Trie-MoE']

        results = {}

        for dataset_name in datasets:
            print(f"\n--- Dataset: {dataset_name} ---")
            results[dataset_name] = {}

            for model_name in models:
                runs = []
                for run in range(num_runs):
                    seed = 42 + run
                    print(f"  {model_name} run {run+1}/{num_runs}...", end=' ', flush=True)

                    try:
                        metrics = self.run_single_experiment(dataset_name, data_size, model_name, seed)
                        runs.append(metrics)
                        print(f"AUC={metrics['auc']:.4f}")
                    except Exception as e:
                        print(f"FAILED: {e}")
                        continue

                if runs:
                    auc_values = [r['auc'] for r in runs]
                    results[dataset_name][model_name] = {
                        'auc_mean': np.mean(auc_values),
                        'auc_std': np.std(auc_values),
                        'runs': runs,
                    }

        return results

    def _compute_significance(self, results):
        """计算统计显著性"""
        for size in results:
            if 'DeepFM' in results[size] and 'Trie-MoE' in results[size]:
                baseline_runs = results[size]['DeepFM']['runs']
                ours_runs = results[size]['Trie-MoE']['runs']

                if len(baseline_runs) >= 2 and len(ours_runs) >= 2:
                    baseline_auc = [r['auc'] for r in baseline_runs]
                    ours_auc = [r['auc'] for r in ours_runs]

                    t_stat, p_value = stats.ttest_ind(ours_auc, baseline_auc)
                    improvement = (np.mean(ours_auc) - np.mean(baseline_auc)) / np.mean(baseline_auc) * 100

                    results[size]['significance'] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'improvement_pct': improvement,
                        'significant': p_value < 0.05,
                    }

    def save_results(self, results, experiment_name):
        """保存结果"""
        output_path = self.output_dir / f'{experiment_name}_{self.timestamp}.json'

        # 转换numpy类型
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        clean_results = json.loads(json.dumps(results, default=convert))

        with open(output_path, 'w') as f:
            json.dump(clean_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    def print_summary(self, results, experiment_name):
        """打印汇总"""
        print("\n" + "="*70)
        print(f"SUMMARY: {experiment_name}")
        print("="*70)

        if isinstance(list(results.keys())[0], int):
            # 数据规模实验
            print(f"{'Size':<10} {'DeepFM AUC':<18} {'Trie-MoE AUC':<18} {'Improvement':<12} {'p-value'}")
            print("-"*70)

            for size in sorted(results.keys()):
                if isinstance(size, int):
                    deepfm = results[size].get('DeepFM', {})
                    triemoe = results[size].get('Trie-MoE', {})
                    sig = results[size].get('significance', {})

                    deepfm_str = f"{deepfm.get('auc_mean', 0):.4f}±{deepfm.get('auc_std', 0):.4f}"
                    triemoe_str = f"{triemoe.get('auc_mean', 0):.4f}±{triemoe.get('auc_std', 0):.4f}"
                    imp_str = f"{sig.get('improvement_pct', 0):+.2f}%"
                    p_str = f"{sig.get('p_value', 1):.4f}"

                    print(f"{size:<10} {deepfm_str:<18} {triemoe_str:<18} {imp_str:<12} {p_str}")
        else:
            # 跨数据集实验
            for dataset in results:
                print(f"\n{dataset}:")
                for model, data in results[dataset].items():
                    print(f"  {model}: AUC={data.get('auc_mean', 0):.4f}±{data.get('auc_std', 0):.4f}")


def main():
    """主函数"""
    print("="*70)
    print("KDD 2026 COMPREHENSIVE EXPERIMENTS")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    runner = ExperimentRunner()

    # 实验1: 数据规模消融 (Criteo)
    print("\n[1/3] Running Data Scale Ablation on Criteo...")
    scale_results = runner.run_data_scale_experiment('criteo', num_runs=3)
    runner.save_results(scale_results, 'data_scale_criteo')
    runner.print_summary(scale_results, 'Data Scale Ablation (Criteo)')

    # 实验2: 跨数据集验证
    print("\n[2/3] Running Cross-Dataset Validation...")
    cross_results = runner.run_cross_dataset_experiment(num_runs=3)
    runner.save_results(cross_results, 'cross_dataset')
    runner.print_summary(cross_results, 'Cross-Dataset Validation')

    # 实验3: MovieLens数据规模
    print("\n[3/3] Running Data Scale Ablation on MovieLens...")
    ml_results = runner.run_data_scale_experiment('movielens', num_runs=3)
    runner.save_results(ml_results, 'data_scale_movielens')
    runner.print_summary(ml_results, 'Data Scale Ablation (MovieLens)')

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()
