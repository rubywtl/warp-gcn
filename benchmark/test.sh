#!/bin/bash
# test.sh — load-imbalance reorder benchmark
# Tests none / degree_asc / degree_desc across datasets and backends
# to verify the load-imbalance hypothesis.
#
# Usage:
#   chmod +x test.sh
#   ./test.sh
#
# Override dataset or backend via env vars:
#   DATASETS="cora citeseer" BACKENDS="scatter" ./test.sh

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export CUDA_VISIBLE_DEVICES

DATASETS=${DATASETS:-"reddit"}
BACKENDS=${BACKENDS:-"cugraph"}
REORDERS="none"
NUM_RUNS=${NUM_RUNS:-200}
HIDDEN=${HIDDEN:-64}

DIVIDER="────────────────────────────────────────────────────"

echo "$DIVIDER"
echo "  Load-Imbalance Reorder Benchmark"
echo "  datasets : $DATASETS"
echo "  backends : $BACKENDS"
echo "  reorders : $REORDERS"
echo "  gpu      : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  NOTE: cugraph backend requires the cugraph-bench conda env (see README)"
echo "  num-runs : $NUM_RUNS  |  hidden : $HIDDEN"
echo "$DIVIDER"
echo ""

for dataset in $DATASETS; do
    for backend in $BACKENDS; do
        echo "▶ dataset=$dataset  backend=$backend"
        echo "$DIVIDER"
        for reorder in $REORDERS; do
            echo "  -- reorder: $reorder"
            python benchmark.py \
                --dataset   "$dataset"  \
                --backend   "$backend"  \
                --reorder   "$reorder"  \
                --num-runs  "$NUM_RUNS" \
                --hidden    "$HIDDEN"
            echo ""
        done
        echo ""
    done
done

echo "$DIVIDER"
echo "  Done. Compare p50/p95/p99 across reorders per dataset+backend."
echo "  If degree_desc tightens p99, load imbalance is the bottleneck."
echo "$DIVIDER"