$ErrorActionPreference = "Stop"

$env:PYTHONPATH = (Get-Location).Path
$CondaPython = Join-Path $env:USERPROFILE "anaconda3\python.exe"
$PythonExe = if (Test-Path -LiteralPath $CondaPython) { $CondaPython } else { "python" }

& $PythonExe node_classfication_embedding_original_rwsearch.py `
  --input_mode gpickle `
  --dataset email-Enron `
  --train gpickle/email-Enron/classified_network_email-Enron_train_1.gpickle `
  --test gpickle/email-Enron/classified_network_email-Enron_all_1.gpickle `
  --target dataset/test/email-Enron_openTriangles_test.txt `
  --priority_nodes dataset/train/email-Enron_new_closures_train.txt `
  --results results/dual_mapping_email_enron_10lift_seed41_deterministic `
  --rw_cache_dir .cache_rw `
  --rw_reference_cache rw_embedding_full_email_enron_seed0.bin `
  --rw_seeds 0 `
  --model_seeds 41 `
  --val_seed 2026 `
  --reuse_splits false `
  --rw_impl fast_positive `
  --rw_walks_per_node 30 `
  --rw_epochs 10 `
  --rw_embedding_dim 64 `
  --rw_prune_fanout 100 `
  --model_type dual_adj_gat `
  --supervision_scope open_candidates `
  --model_num_layers 4 `
  --model_hidden_channels 128 `
  --model_heads 2 `
  --model_dropout 0.2 `
  --dual_dropout 0.2 `
  --lr 0.002 `
  --weight_decay 0.001 `
  --model_selection_by val_ap `
  --temporal_activity_alpha 0.90 `
  --temporal_activity_recent_fraction 0.15 `
  --temporal_activity_aggregation product `
  --temporal_activity_entity node `
  --epochs 60 `
  --test_eval_interval 60 `
  --device cuda `
  --deterministic true
