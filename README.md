# K=2 reproduction release

新增的独立复现入口在 [`k2_reproduction_20261020/`](k2_reproduction_20261020/README.md)，覆盖8数据集 × Dual-Mapping/MLSHoL × 3种子，保留完整历史分支与融合；附录包含启发式及K2–K4结构实验源码。K3模型实验不在本次入口范围内。

- [下载完整包与数据资源](https://github.com/lztzmzms/Higher-Order-Link-Prediction-via-Higher-Order-Adjacency-Driven-Graph-Neural-Networks/releases/tag/k2-reproduction-20261020) · [源码与资源合并说明](k2_reproduction_20261020/GITHUB_DOWNLOAD.md)
- [最新独立训练实测](validation/k2_release_20261022/README.md)：42/48次核验、14组完整结果；部分成绩不一致，两个MLSHoL组合未完成，不能宣称全部汇报性能已复现。

```bash
cd k2_reproduction_20261020
# 首先按下载说明补齐资源，并安装匹配设备的PyTorch
python -m pip install -r requirements.txt
python run.py --dataset NDC-substances --model dual
python run.py --dataset tags-ask-ubuntu --model dual
```

NDC Dual保留2层及其22.623230历史结果，其余Dual固定4层。训练已按作者要求暂停；本次上传没有重启实验。

以下保留原有 Code Ocean 单种子运行说明，其协议与新增K=2三种子入口分别说明。

---

# Clean Dual-Mapping / MLSHOL

Minimal runnable package for:

- Dual-Mapping GNN runs.
- MLSHOL / SparseSIN runs.

Run commands from this directory.

## Code Ocean Reproducibility Scope

The Code Ocean capsule runs only the `email-Enron` and
`contact-high-school` datasets. To keep the reproducible run within the
available execution time, each dataset is evaluated once using one fixed
random seed. Consequently, the reported one-run mean is equal to that single
run and is not a multi-seed average.

Other supported datasets are not executed in Code Ocean. The complete code
can be obtained from the
[GitHub repository](https://github.com/lztzmzms/Higher-Order-Link-Prediction-via-Higher-Order-Adjacency-Driven-Graph-Neural-Networks),
which was already linked in the initial manuscript draft, and those datasets
can be run locally.

## Install

```bash
pip install -r requirements.txt
```

For GPU runs, install `torch`, `torch-geometric`, and compiled `torch-scatter`
versions that match your CUDA environment.

## Data

Included files cover the direct commands below:

```text
dataset/
  raw_data/email-Enron/
  raw_data/contact-high-school/
  train/
  test/
gpickle/
  email-Enron/
  contact-high-school/
rw_embedding_full_email_enron_seed0.bin
rw_embedding_fast_positive_w40_e10_d64_seed0.bin
checkpoint_seed_54.bin
contact-high-school_train_edge_maxsimplex_features.bin
contact-high-school_test_edge_maxsimplex_features.bin
```

## Dual-Mapping Reproduction Commands

The results below are single-seed runs. Therefore, each reported
one-run mean is the result of that run rather than a multi-seed average.

### email-Enron

This run uses random-walk seed `0`, model seed `41`, 60 training epochs, and
a recent node-activity prior.

```bash
cd /code
export PYTHONPATH="/code:${PYTHONPATH:-}"

python -u /code/node_classfication_embedding_original_rwsearch.py \
  --input_mode gpickle \
  --dataset email-Enron \
  --train /code/gpickle/email-Enron/classified_network_email-Enron_train_1.gpickle \
  --test /code/gpickle/email-Enron/classified_network_email-Enron_all_1.gpickle \
  --target /code/dataset/test/email-Enron_openTriangles_test.txt \
  --priority_nodes /code/dataset/train/email-Enron_new_closures_train.txt \
  --results /results/results_dual_mapping_email_enron_seed41 \
  --rw_cache_dir /tmp/cache_dual_mapping_email_enron_seed41 \
  --rw_reference_cache /code/rw_embedding_full_email_enron_seed0.bin \
  --rw_seeds 0 \
  --model_seeds 41 \
  --val_seed 2026 \
  --reuse_splits false \
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
  --test_eval_interval 60 \
  --deterministic true \
  --device cuda
```

Observed output:

```text
dataset=email-Enron
mean_AUC_PR=0.343619
randomline=0.034344
liftx=10.005159
```

### contact-high-school

This run uses a fixed random-walk embedding, model seed `54`, a frozen
checkpoint, and edge maximal-simplex features.

```bash
cd /code
export PYTHONPATH="/code:${PYTHONPATH:-}"

python -u /code/node_classfication_embedding_original_rwsearch.py \
  --input_mode gpickle \
  --dataset contact-high-school \
  --train /code/gpickle/contact-high-school/classified_network_contact-high-school_train_1.gpickle \
  --test /code/gpickle/contact-high-school/classified_network_contact-high-school_all_1.gpickle \
  --target /code/dataset/test/contact-high-school_openTriangles_test.txt \
  --priority_nodes /code/dataset/train/contact-high-school_new_closures_train.txt \
  --results /results/results_dual_mapping_contact_high_seed54 \
  --rw_cache_dir /tmp/cache_dual_mapping_contact_high_seed54 \
  --rw_reference_cache /code/rw_embedding_fast_positive_w40_e10_d64_seed0.bin \
  --rw_seeds 0 \
  --model_seeds 54 \
  --pretrained_checkpoint /code/checkpoint_seed_54.bin \
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
  --train_edge_maxsimplex_txt /code/contact-high-school_train_edge_maxsimplex_features.bin \
  --test_edge_maxsimplex_txt /code/contact-high-school_test_edge_maxsimplex_features.bin \
  --edge_maxsimplex_fusion prior_logit \
  --prior_alpha 2.0 \
  --selector_val priority_holdout \
  --test_eval_interval 60 \
  --epochs 60 \
  --device auto
```

Observed output:

```text
dataset=contact-high-school
mean_AUC_PR=0.202650
randomline=0.008022
liftx=25.261727
```

## MLSHOL / SparseSIN

The Code Ocean results for `email-Enron` and `contact-high-school` are
single-seed runs. The current script trains each dataset once with a fixed
model seed, sampling seed `44`, and 20 epochs. It writes `results.csv` and a
one-run `summary.csv`.

Code Ocean entry:

```bash
bash /code/run_mlshol.sh
```

Observed output:

```text
dataset=email-Enron
model_seed=41
sampling_seed=44
mean_auc_pr=0.398924
mean_baseline=0.038076
mean_lift=10.477001

dataset=contact-high-school
model_seed=44
sampling_seed=44
mean_auc_pr=0.085847
mean_baseline=0.008000
mean_lift=10.730906
```

Local entry for `email-Enron`:

```powershell
python -u -m exp.run_all_sparse_sin `
  --datasets email-Enron `
  --seeds 41 `
  --sampling-seed 44 `
  --output-dir results/results_mlshol_email_enron_seed41 `
  --cache-dir tmp/cache_mlshol_email_enron_sampling44 `
  --rw-source dual_mapping_full `
  --rw-reference-cache rw_embedding_full_email_enron_seed0.bin `
  --rw-fusion rw_only `
  --rw-seed 0 `
  --rw-dim 64 `
  --rw-scale 2.0 `
  --rw-input-normalization none `
  --rw-full-normalization train_standardize `
  --rw-walks-per-node 30 `
  --rw-epochs 10 `
  --temporal-activity-alpha 0.84 `
  --temporal-activity-recent-fraction 0.15 `
  --temporal-activity-aggregation product `
  --temporal-activity-entity node `
  --maxsimplex-prior-weight 0 `
  --sampling-mode stratified `
  --test-sampling-mode uniform `
  --train-positive-fraction 0.2 `
  --max-train-complexes 200 `
  --max-test-complexes 1000 `
  --epochs 20 `
  --eval-every 5 `
  --loss-mode weighted_cross_entropy `
  --strict
```

Local entry for `contact-high-school`:

```powershell
python -u -m exp.run_all_sparse_sin `
  --datasets contact-high-school `
  --seeds 44 `
  --sampling-seed 44 `
  --output-dir results/results_mlshol_contact_high_school_seed44 `
  --cache-dir tmp/cache_mlshol_contact_high_school_sampling44 `
  --rw-source dual_mapping_full `
  --rw-reference-cache rw_embedding_fast_positive_w40_e10_d64_seed0.bin `
  --rw-fusion rw_only `
  --rw-seed 0 `
  --rw-dim 64 `
  --rw-scale 2.0 `
  --rw-input-normalization none `
  --rw-full-normalization train_standardize `
  --rw-walks-per-node 40 `
  --rw-epochs 10 `
  --temporal-activity-alpha 0.55 `
  --temporal-activity-recent-fraction 1.0 `
  --temporal-activity-aggregation product `
  --temporal-activity-entity edge `
  --maxsimplex-prior-weight 1.25 `
  --maxsimplex-prior-column tri_all_edges_shared_count `
  --maxsimplex-feature-dir . `
  --sampling-mode stratified `
  --test-sampling-mode stratified `
  --train-positive-fraction 0.2 `
  --test-positive-fraction 0.008022030651340996 `
  --max-train-complexes 200 `
  --max-test-complexes 500 `
  --epochs 20 `
  --eval-every 5 `
  --loss-mode weighted_cross_entropy `
  --strict
```

## Raw Dual-Mapping

```powershell
python dual_mapping_from_raw.py `
  --nverts dataset/raw_data/email-Enron/email-Enron-nverts.txt `
  --simplices dataset/raw_data/email-Enron/email-Enron-simplices.txt `
  --times dataset/raw_data/email-Enron/email-Enron-times.txt `
  --results results_dual_mapping_email_enron
```
