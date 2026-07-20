#!/usr/bin/env bash
set -ex

# This is the master script for the capsule. When you click "Reproducible Run", the code in this file will execute.
cd /code
export PYTHONPATH="/code:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=-1
export PYTHONHASHSEED=44
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u -m exp.run_all_sparse_sin \
    --datasets email-Enron \
    --seeds 41 \
    --sampling-seed 44 \
    --output-dir /results/results_mlshol_email_enron_seed41 \
    --cache-dir /tmp/cache_mlshol_email_enron_sampling44 \
    --rw-source dual_mapping_full \
    --rw-reference-cache /code/rw_embedding_full_email_enron_seed0.bin \
    --rw-fusion rw_only \
    --rw-seed 0 \
    --rw-dim 64 \
    --rw-scale 2.0 \
    --rw-input-normalization none \
    --rw-full-normalization train_standardize \
    --rw-walks-per-node 30 \
    --rw-epochs 10 \
    --temporal-activity-alpha 0.84 \
    --temporal-activity-recent-fraction 0.15 \
    --temporal-activity-aggregation product \
    --temporal-activity-entity node \
    --maxsimplex-prior-weight 0 \
    --sampling-mode stratified \
    --test-sampling-mode uniform \
    --train-positive-fraction 0.2 \
    --max-train-complexes 200 \
    --max-test-complexes 1000 \
    --epochs 20 \
    --eval-every 5 \
    --loss-mode weighted_cross_entropy \
    --strict \
    "$@"

python -u -m exp.run_all_sparse_sin \
    --datasets contact-high-school \
    --seeds 44 \
    --sampling-seed 44 \
    --output-dir /results/results_mlshol_contact_high_school_seed44 \
    --cache-dir /tmp/cache_mlshol_contact_high_school_sampling44 \
    --rw-source dual_mapping_full \
    --rw-reference-cache /code/rw_embedding_fast_positive_w40_e10_d64_seed0.bin \
    --rw-fusion rw_only \
    --rw-seed 0 \
    --rw-dim 64 \
    --rw-scale 2.0 \
    --rw-input-normalization none \
    --rw-full-normalization train_standardize \
    --rw-walks-per-node 40 \
    --rw-epochs 10 \
    --temporal-activity-alpha 0.55 \
    --temporal-activity-recent-fraction 1.0 \
    --temporal-activity-aggregation product \
    --temporal-activity-entity edge \
    --maxsimplex-prior-weight 1.25 \
    --maxsimplex-prior-column tri_all_edges_shared_count \
    --maxsimplex-feature-dir /code/.cache_features \
    --sampling-mode stratified \
    --test-sampling-mode stratified \
    --train-positive-fraction 0.2 \
    --test-positive-fraction 0.008022030651340996 \
    --max-train-complexes 200 \
    --max-test-complexes 500 \
    --epochs 20 \
    --eval-every 5 \
    --loss-mode weighted_cross_entropy \
    --strict \
    "$@"
