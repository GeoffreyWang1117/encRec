#!/bin/bash
# Run SOTA Comparison Experiments for RecSys 2026 Paper
# Usage: ./run_sota_experiments.sh [dataset] [gpu_id]

set -e

DATASET=${1:-movielens}
GPU=${2:-0}

export CUDA_VISIBLE_DEVICES=$GPU

echo "============================================"
echo "SOTA Comparison Experiments"
echo "Dataset: $DATASET"
echo "GPU: $GPU"
echo "============================================"

cd "$(dirname "$0")/.."

# Experiment 1: Few-shot comparison (1K, 2K, 5K samples)
echo ""
echo "=== Experiment 1: Few-shot Comparison ==="
python experiments/sota_comparison.py \
    --dataset $DATASET \
    --models DeepFM DCNv2 AutoInt FinalMLP DropoutNet MMoE PLE AdaptiveAlpha EnhancedAdaptive \
    --train-sizes 1000 2000 5000 \
    --num-runs 5

# Experiment 2: Cold-start threshold sensitivity
echo ""
echo "=== Experiment 2: Cold-start Threshold Sensitivity ==="
python experiments/cold_start_threshold_analysis.py \
    --dataset $DATASET \
    --thresholds 1 3 5 10 20

# Experiment 3: Full scale comparison
echo ""
echo "=== Experiment 3: Full Scale Comparison ==="
python experiments/sota_comparison.py \
    --dataset $DATASET \
    --models DeepFM DCNv2 AutoInt FinalMLP AdaptiveAlpha EnhancedAdaptive \
    --train-sizes 10000 50000 \
    --num-runs 3

echo ""
echo "============================================"
echo "All experiments completed!"
echo "Results saved to: results/sota_comparison/"
echo "============================================"
