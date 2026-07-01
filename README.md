# Clean Dual-Mapping / MLSHOL

Minimal runnable package for:

- Dual-Mapping GNN runs.
- MLSHOL / SparseSIN runs.

Run commands from this directory.

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
  train/
  test/
gpickle/
  email-Enron/
  contact-high-school/
.cache_features/
  contact-high-school_*_edge_maxsimplex_features.tsv
```

The Enron `maildir/` tree is not included; raw Dual-Mapping only needs
`email-Enron-nverts.txt`, `email-Enron-simplices.txt`, and
`email-Enron-times.txt`.

## Dual-Mapping Best Commands

### email-Enron

Reference: `mean_AUC_PR=0.340349`, `liftx=9.909956`.

```powershell
python node_classfication_embedding_original_rwsearch.py `
  --input_mode gpickle `
  --dataset email-Enron `
  --train gpickle/email-Enron/classified_network_email-Enron_train_1.gpickle `
  --test gpickle/email-Enron/classified_network_email-Enron_all_1.gpickle `
  --target dataset/test/email-Enron_openTriangles_test.txt `
  --priority_nodes dataset/train/email-Enron_new_closures_train.txt `
  --results results_email_enron_best_direct `
  --rw_cache_dir .cache_rw_email_enron_best `
  --rw_seeds 0 `
  --model_seeds 48,49,50,51,52,53,54,55,56,57 `
  --val_seed 2026 `
  --rw_impl fast_positive `
  --rw_walks_per_node 30 `
  --rw_epochs 10 `
  --rw_prune_fanout 100 `
  --local_num_neighbors 100 `
  --model_type dual_adj_gat `
  --epochs 150
```

### contact-high-school

Reference: `mean_AUC_PR=0.138553`, `liftx=17.271601`.

```powershell
python node_classfication_embedding_original_rwsearch.py `
  --input_mode gpickle `
  --dataset contact-high-school `
  --train gpickle/contact-high-school/classified_network_contact-high-school_train_1.gpickle `
  --test gpickle/contact-high-school/classified_network_contact-high-school_all_1.gpickle `
  --target dataset/test/contact-high-school_openTriangles_test.txt `
  --priority_nodes dataset/train/contact-high-school_new_closures_train.txt `
  --results results_contact_high_best_direct `
  --rw_cache_dir .cache_rw_contact_high_best `
  --rw_seeds 0 `
  --model_seeds 48,49,50,51,52,53,54,55,56,57 `
  --val_seed 2026 `
  --rw_impl fast_positive `
  --rw_walks_per_node 40 `
  --rw_epochs 10 `
  --rw_prune_fanout 100 `
  --local_num_neighbors 50 `
  --model_type dual_adj_gat `
  --use_edge_maxsimplex_features true `
  --train_edge_maxsimplex_txt .cache_features/contact-high-school_train_edge_maxsimplex_features.tsv `
  --test_edge_maxsimplex_txt .cache_features/contact-high-school_test_edge_maxsimplex_features.tsv `
  --edge_maxsimplex_fusion prior_logit `
  --prior_alpha 2.0 `
  --selector_val priority_holdout `
  --epochs 60
```

## MLSHOL / SparseSIN

Run with default parameters. The defaults are set to the verified `email-Enron`
configuration: seed `44`, stratified sampling, 20 epochs, and output directory
`exp/results/email_enron_seed44_default`.

Verification result:

- `lift=15.211955227435102`
- `best_eval_lift=15.212981744421906`
- result file: `exp/results/email_enron_seed44_default/results.csv`

Entry: `python -m exp.run_all_sparse_sin`

## Raw Dual-Mapping

```powershell
python dual_mapping_from_raw.py `
  --nverts dataset/raw_data/email-Enron/email-Enron-nverts.txt `
  --simplices dataset/raw_data/email-Enron/email-Enron-simplices.txt `
  --times dataset/raw_data/email-Enron/email-Enron-times.txt `
  --results results_dual_mapping_email_enron
```
