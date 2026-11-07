# K4 benchmark expansion

This bundle publishes source code and configurations. Its external raw and frozen protocol inputs are outside the distribution; a complete acquisition/regeneration chain for all frozen feature, baseline and checkpoint identities is not supplied. The commands below assume the required resources have been provided.

Datasets: NDC-classes, congress-bills, DAWN. Install `requirements.txt` in the experiment environment. The nine required raw event files are not included in this source release. Obtain them from the sources listed in `manifests/K4_LOCAL_EXPANSION_DATA_MANIFEST.json`, and set `K4_DATA_ROOT` to that directory. Additional frozen protocol resources may also be needed as specified by the stage configurations and source consumers.

From this directory:

```bash
python scripts/k4_server_pipeline.py s0 --dry-run
python scripts/k4_server_pipeline.py selftest
bash RUN_ALL_SERVER.sh
```

The stages are `s0` validation, `s1` data verification, `s2` exact candidate census, `s3` support gate, `s4` features and MLP, and `s5` summary. Each stage can be called directly; `--dataset` selects one dataset, and `--dry-run` checks its setup. Results and resumable checkpoints are written under `server_results/k4_server_benchmark_expansion_v1/`.

The primary endpoint is LogLoss, using strict timestamp-block T2, five seeds 21001-21005, standard-residual Full, cumulative C0*-C3* features and 2,000 paired candidate bootstraps with margin 0.02. AP uses `sklearn.metrics.average_precision_score`.

The congress-bills exact disk-index procedure checks for at least 60 GiB free disk before its feature stage. The structural definitions are retained; input availability and complete execution remain unverified.
