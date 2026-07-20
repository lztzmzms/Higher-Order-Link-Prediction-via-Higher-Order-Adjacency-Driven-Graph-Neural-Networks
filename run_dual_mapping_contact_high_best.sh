#!/usr/bin/env bash
set -euo pipefail
set -x

cd /code
export PYTHONPATH="/code:${PYTHONPATH:-}"

SOURCE="/code/node_classfication_embedding_original_rwsearch.py"
REFERENCE_RW_PREFERRED="/code/rw_embedding_fast_positive_w40_e10_d64_seed0.bin"
CHECKPOINT_PREFERRED="/code/checkpoint_seed_54.bin"
TRAIN_EDGE_FEATURES_PREFERRED="/code/contact-high-school_train_edge_maxsimplex_features.bin"
TEST_EDGE_FEATURES_PREFERRED="/code/contact-high-school_test_edge_maxsimplex_features.bin"
RESULT_DIR="/results/results_contact_high_seed54_repro"
RW_CACHE_DIR="/tmp/cache_rw_contact_high_seed54_repro"
GENERATED_GPICKLE_DIR="/tmp/generated_gpickle_contact_high_seed54_repro"

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

resolve_asset() {
    local preferred="$1"
    local filename
    local candidate
    filename="$(basename "${preferred}")"
    if [[ -f "${preferred}" ]]; then
        printf '%s\n' "${preferred}"
        return 0
    fi
    candidate="$(find /code -type f -name "${filename}" -print -quit)"
    if [[ -n "${candidate}" ]]; then
        echo "INFO: resolved ${filename} at ${candidate}" >&2
        printf '%s\n' "${candidate}"
        return 0
    fi
    echo "ERROR: required runtime asset was not uploaded: ${filename}" >&2
    echo "Expected path: ${preferred}" >&2
    echo "Files with related names currently found under /code:" >&2
    find /code -maxdepth 5 -type f -name "*${filename##*_}" -print >&2 2>/dev/null || true
    return 1
}

REFERENCE_RW="$(resolve_asset "${REFERENCE_RW_PREFERRED}")"
CHECKPOINT="$(resolve_asset "${CHECKPOINT_PREFERRED}")"
TRAIN_EDGE_FEATURES="$(resolve_asset "${TRAIN_EDGE_FEATURES_PREFERRED}")"
TEST_EDGE_FEATURES="$(resolve_asset "${TEST_EDGE_FEATURES_PREFERRED}")"

verify_sha256 "${SOURCE}" "26bd1fd0fa2640d8b3b3d082089fe70d1db662d10761a24669cd541490eba6eb"
verify_sha256 "${REFERENCE_RW}" "922d9ee306c2ddc72b263557db31d65975606a895d594d992e01d8ecdaf5c6ac"
verify_sha256 "${TRAIN_EDGE_FEATURES}" "cd08e8a06ff1f2fa7d687f0a9635615d06bf16f97042a03d48a4af33b8682fb8"
verify_sha256 "${TEST_EDGE_FEATURES}" "1dfcabfe24b8bb2c532ceae066847002845ffb3de2167aeddfa829598352f559"
verify_sha256 "/code/dataset/train/contact-high-school_closeTriangles_train.txt" "a9c7a26b86d7e31e03cee41e72eed01d217270e04f21b964d1e95206d476949b"
verify_sha256 "/code/dataset/train/contact-high-school_openTriangles_train.txt" "933949562eb3f9547084ef5d6a25eb8e24d96b53420bbbf217c7d2d4840a1ed0"
verify_sha256 "/code/dataset/test/contact-high-school_closeTriangles_test.txt" "db328e28e27ddac12b618b2466165405ddc8e2a86e0626da50a75f7e54918047"
verify_sha256 "/code/dataset/test/contact-high-school_openTriangles_test.txt" "37af97fa3b36ce824046575d96ec30d74193521e70c9a888b2bf756155fe6775"
verify_sha256 "/code/dataset/train/contact-high-school_new_closures_train.txt" "ed445ebb04a9c80cf14564d0116086059a12bdc575e88b3d47dadc2906dbc21b"
verify_sha256 "/code/dataset/test/contact-high-school_new_closures_test.txt" "6f2dbbf95e3dd99e13f52f32dc68563e7356301d0f5ddfb57e0a726802b1ce79"
verify_sha256 "${CHECKPOINT}" "71050f7051e692a28c3af9a67fb8f9cad349b482ecaa7101e934b9251963860a"

mkdir -p "${RESULT_DIR}" "${RW_CACHE_DIR}" "${GENERATED_GPICKLE_DIR}"

exec python -u "${SOURCE}" \
    --input_mode protocol \
    --dataset contact-high-school \
    --generated_gpickle_dir "${GENERATED_GPICKLE_DIR}" \
    --target /code/dataset/test/contact-high-school_openTriangles_test.txt \
    --priority_nodes /code/dataset/train/contact-high-school_new_closures_train.txt \
    --results "${RESULT_DIR}" \
    --rw_cache_dir "${RW_CACHE_DIR}" \
    --rw_reference_cache "${REFERENCE_RW}" \
    --rw_seeds 0 \
    --model_seeds 54 \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --pretrained_threshold 0.5 \
    --val_seed 2026 \
    --reuse_splits false \
    --rw_impl fast_positive \
    --rw_walks_per_node 40 \
    --rw_epochs 10 \
    --rw_embedding_dim 64 \
    --rw_prune_fanout 100 \
    --local_num_neighbors 50 \
    --model_type dual_adj_gat \
    --use_edge_maxsimplex_features true \
    --train_edge_maxsimplex_txt "${TRAIN_EDGE_FEATURES}" \
    --test_edge_maxsimplex_txt "${TEST_EDGE_FEATURES}" \
    --edge_maxsimplex_fusion prior_logit \
    --prior_alpha 2.0 \
    --selector_val priority_holdout \
    --test_eval_interval 60 \
    --epochs 60 \
    "$@"
