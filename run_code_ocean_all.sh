#!/usr/bin/env bash
set -euo pipefail
set -x

# Code Ocean master run:
#   1. Dual-Mapping + temporal activity prior / email-Enron (one trained seed)
#   2. Dual-Mapping / contact-high-school (one frozen seed)
#   3. MLSHOL / email-Enron (one seed)
#   4. MLSHOL / contact-high-school (one seed)

cd /code
export PYTHONPATH="/code:${PYTHONPATH:-}"
export PYTHONHASHSEED=44

SOURCE="/code/node_classfication_embedding_original_rwsearch.py"
MLSHOL_EMAIL_RW="/code/rw_embedding_full_email_enron_seed0.bin"
CONTACT_RW="/code/rw_embedding_fast_positive_w40_e10_d64_seed0.bin"
CONTACT_CHECKPOINT="/code/checkpoint_seed_54.bin"

EMAIL_TRAIN="/code/gpickle/email-Enron/classified_network_email-Enron_train_1.gpickle"
EMAIL_TEST="/code/gpickle/email-Enron/classified_network_email-Enron_all_1.gpickle"
CONTACT_TRAIN="/code/gpickle/contact-high-school/classified_network_contact-high-school_train_1.gpickle"
CONTACT_TEST="/code/gpickle/contact-high-school/classified_network_contact-high-school_all_1.gpickle"
CONTACT_TRAIN_FEATURES="/code/contact-high-school_train_edge_maxsimplex_features.bin"
CONTACT_TEST_FEATURES="/code/contact-high-school_test_edge_maxsimplex_features.bin"

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: required file not found: ${path}" >&2
        exit 1
    fi
}

verify_sha256() {
    local path="$1"
    local expected="$2"
    local actual
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "ERROR: SHA256 mismatch: ${path}" >&2
        echo "expected=${expected}" >&2
        echo "actual=${actual}" >&2
        exit 1
    fi
}

required_files=(
    "${SOURCE}"
    "${MLSHOL_EMAIL_RW}"
    "${CONTACT_RW}"
    "${CONTACT_CHECKPOINT}"
    "${EMAIL_TRAIN}"
    "${EMAIL_TEST}"
    "${CONTACT_TRAIN}"
    "${CONTACT_TEST}"
    "${CONTACT_TRAIN_FEATURES}"
    "${CONTACT_TEST_FEATURES}"
    "/code/dataset/train/email-Enron_openTriangles_train.txt"
    "/code/dataset/train/email-Enron_new_closures_train.txt"
    "/code/dataset/test/email-Enron_openTriangles_test.txt"
    "/code/dataset/train/contact-high-school_openTriangles_train.txt"
    "/code/dataset/train/contact-high-school_new_closures_train.txt"
    "/code/dataset/test/contact-high-school_openTriangles_test.txt"
    "/code/exp/run_all_sparse_sin.py"
)

for file in "${required_files[@]}"; do
    require_file "${file}"
done

required_directories=(
    "/code/dataset/raw_data/email-Enron"
    "/code/dataset/raw_data/contact-high-school"
    "/code/mp"
    "/code/simplicial_complex"
)

for directory in "${required_directories[@]}"; do
    if [[ ! -d "${directory}" ]]; then
        echo "ERROR: required directory not found: ${directory}" >&2
        exit 1
    fi
done

# Verify frozen binary/model artifacts and graph alignment inputs.
verify_sha256 "${MLSHOL_EMAIL_RW}" \
    "578222e3fe7d0169cbb3b20efebcd9de7aa806593d237c3588ff3248d0094821"
verify_sha256 "${CONTACT_RW}" \
    "922d9ee306c2ddc72b263557db31d65975606a895d594d992e01d8ecdaf5c6ac"
verify_sha256 "${CONTACT_CHECKPOINT}" \
    "71050f7051e692a28c3af9a67fb8f9cad349b482ecaa7101e934b9251963860a"
verify_sha256 "${EMAIL_TRAIN}" \
    "109f59970011f5ccf8da29ea4f15dc11e742e36da5f76aca44efde8f6dd005e7"
verify_sha256 "${EMAIL_TEST}" \
    "c221f6bc21bb06b761c7e5a8c34ed8a63bdd55e2c8ff266ea905e14c5fe1ac6c"
verify_sha256 "${CONTACT_TRAIN}" \
    "127912f259f3ca4b4d7468fdbee4313cdef8738397a65e040a5b34ec075e3ac4"
verify_sha256 "${CONTACT_TEST}" \
    "b9c129a7a412bbc7b769554d20c5a9bfac22fa96d5645d5a9eb4b34e85bab8ee"
verify_sha256 "${CONTACT_TRAIN_FEATURES}" \
    "cd08e8a06ff1f2fa7d687f0a9635615d06bf16f97042a03d48a4af33b8682fb8"
verify_sha256 "${CONTACT_TEST_FEATURES}" \
    "1dfcabfe24b8bb2c532ceae066847002845ffb3de2167aeddfa829598352f559"

mkdir -p \
    /results/results_dual_mapping_email_enron_seed41 \
    /results/results_dual_mapping_contact_high_seed54 \
    /results/results_mlshol_email_enron_seed41 \
    /results/results_mlshol_contact_high_school_seed44 \
    /tmp/cache_dual_mapping_email_enron_seed41 \
    /tmp/cache_dual_mapping_contact_high_seed54 \
    /tmp/cache_mlshol_email_enron_sampling44 \
    /tmp/cache_mlshol_contact_high_school_sampling44

# ---------------------------------------------------------------------------
# 1. Dual-Mapping + temporal activity prior: email-Enron, model seed 41.
# Verified deterministic result: AP=0.388658, lift=11.316563.
# The raw Dual-Mapping GNN component has lift=2.604330; the final reported
# score includes a context-only activity prior with alpha=0.90.
# ---------------------------------------------------------------------------
python -u "${SOURCE}" \
    --input_mode gpickle \
    --dataset email-Enron \
    --train "${EMAIL_TRAIN}" \
    --test "${EMAIL_TEST}" \
    --target /code/dataset/test/email-Enron_openTriangles_test.txt \
    --priority_nodes /code/dataset/train/email-Enron_new_closures_train.txt \
    --results /results/results_dual_mapping_email_enron_seed41 \
    --rw_cache_dir /tmp/cache_dual_mapping_email_enron_seed41 \
    --rw_reference_cache "${MLSHOL_EMAIL_RW}" \
    --run_mode exact_fast \
    --preset original_f2 \
    --reuse_splits false \
    --rw_seeds 0 \
    --model_seeds 41 \
    --val_seed 2026 \
    --rw_impl fast_positive \
    --rw_cache_mode per_rw_seed \
    --rw_walks_per_node 30 \
    --rw_epochs 10 \
    --rw_embedding_dim 64 \
    --rw_walk_length 10 \
    --rw_window_size 2 \
    --rw_graph_scope full \
    --rw_prune_fanout 100 \
    --model_type dual_adj_gat \
    --supervision_scope open_candidates \
    --model_num_layers 4 \
    --model_hidden_channels 128 \
    --model_heads 2 \
    --model_dropout 0.2 \
    --dual_self_hidden_dim 128 \
    --dual_gate_hidden_dim 64 \
    --dual_dropout 0.2 \
    --lr 0.002 \
    --weight_decay 0.001 \
    --model_selection_by val_ap \
    --temporal_activity_alpha 0.90 \
    --temporal_activity_recent_fraction 0.15 \
    --temporal_activity_aggregation product \
    --temporal_activity_entity node \
    --epochs 60 \
    --batch_size 2048 \
    --test_eval_interval 60 \
    --deterministic true \
    --device cuda

# ---------------------------------------------------------------------------
# 2. Dual-Mapping: contact-high-school, frozen model seed 54.
# Verified with average_precision_score: AP=0.202650, lift=25.261727.
# ---------------------------------------------------------------------------
python -u "${SOURCE}" \
    --input_mode gpickle \
    --dataset contact-high-school \
    --train "${CONTACT_TRAIN}" \
    --test "${CONTACT_TEST}" \
    --target /code/dataset/test/contact-high-school_openTriangles_test.txt \
    --priority_nodes /code/dataset/train/contact-high-school_new_closures_train.txt \
    --results /results/results_dual_mapping_contact_high_seed54 \
    --rw_cache_dir /tmp/cache_dual_mapping_contact_high_seed54 \
    --rw_reference_cache "${CONTACT_RW}" \
    --rw_seeds 0 \
    --model_seeds 54 \
    --pretrained_checkpoint "${CONTACT_CHECKPOINT}" \
    --pretrained_threshold 0.5 \
    --val_seed 2026 \
    --reuse_splits false \
    --rw_impl fast_positive \
    --rw_walks_per_node 40 \
    --rw_epochs 10 \
    --rw_embedding_dim 64 \
    --rw_graph_scope full \
    --rw_prune_fanout 100 \
    --model_type dual_adj_gat \
    --local_num_neighbors 50 \
    --use_edge_maxsimplex_features true \
    --train_edge_maxsimplex_txt "${CONTACT_TRAIN_FEATURES}" \
    --test_edge_maxsimplex_txt "${CONTACT_TEST_FEATURES}" \
    --edge_maxsimplex_fusion prior_logit \
    --prior_alpha 2.0 \
    --selector_val priority_holdout \
    --test_eval_interval 60 \
    --epochs 60 \
    --device auto

# MLSHOL runs are intentionally CPU-only and single-threaded for stable
# Code Ocean reproduction.
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---------------------------------------------------------------------------
# 3. MLSHOL: email-Enron, model seed 41, sampling seed 44.
# Verified: AP=0.398916, baseline=0.038076, lift=10.476795.
# ---------------------------------------------------------------------------
python -u -m exp.run_all_sparse_sin \
    --datasets email-Enron \
    --seeds 41 \
    --sampling-seed 44 \
    --output-dir /results/results_mlshol_email_enron_seed41 \
    --cache-dir /tmp/cache_mlshol_email_enron_sampling44 \
    --rw-source dual_mapping_full \
    --rw-reference-cache "${MLSHOL_EMAIL_RW}" \
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
    --strict

# ---------------------------------------------------------------------------
# 4. MLSHOL: contact-high-school, model seed 44, sampling seed 44.
# Verified: AP=0.085847, baseline=0.008000, lift=10.730906.
# ---------------------------------------------------------------------------
python -u -m exp.run_all_sparse_sin \
    --datasets contact-high-school \
    --seeds 44 \
    --sampling-seed 44 \
    --output-dir /results/results_mlshol_contact_high_school_seed44 \
    --cache-dir /tmp/cache_mlshol_contact_high_school_sampling44 \
    --rw-source dual_mapping_full \
    --rw-reference-cache "${CONTACT_RW}" \
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
    --maxsimplex-feature-dir /code \
    --sampling-mode stratified \
    --test-sampling-mode stratified \
    --train-positive-fraction 0.2 \
    --test-positive-fraction 0.008022030651340996 \
    --max-train-complexes 200 \
    --max-test-complexes 500 \
    --epochs 20 \
    --eval-every 5 \
    --loss-mode weighted_cross_entropy \
    --strict

FINAL_OUTPUT="/results/output.txt"
{
    echo "All requested Code Ocean runs completed."
    echo
    echo "=== Dual-Mapping + temporal activity prior: email-Enron ==="
    cat /results/results_dual_mapping_email_enron_seed41/seed_results.csv
    echo
    echo "=== Dual-Mapping: contact-high-school ==="
    cat /results/results_dual_mapping_contact_high_seed54/seed_results.csv
    echo
    echo "=== MLSHOL: email-Enron ==="
    cat /results/results_mlshol_email_enron_seed41/summary.csv
    echo
    echo "=== MLSHOL: contact-high-school ==="
    cat /results/results_mlshol_contact_high_school_seed44/summary.csv
} | tee "${FINAL_OUTPUT}"

echo "Combined final summary saved to: ${FINAL_OUTPUT}"
