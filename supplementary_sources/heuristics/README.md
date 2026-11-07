# Hodge-Laplacian heuristics

From the package root:

```bash
python supplementary_sources/heuristics/run.py --dataset email-Enron --check
python supplementary_sources/heuristics/run.py --dataset email-Enron --method Simplicial_CN
python supplementary_sources/heuristics/run.py --dataset email-Enron --method RA --method AA
```

The default runs the eight original methods: Simplicial_CN, RA, Local Path, katz, AA, PA, Global AA and Global RA. `Degree` is also available. `--method` can be repeated. Dependencies are NumPy, SciPy and scikit-learn from the package requirements.

The input events remain in their supplied row order. The first 80% form the observed simplicial complex; the remaining 20% supply future triangle closures. Evaluation enumerates all open triangles in the observed complex.

Input events now come from the package's `data/<dataset>/events.npz` or `data/<dataset>/inputs.zip`. The original contact-high-school input for these heuristics is preserved separately in `supplementary_sources/heuristics/data/contact-high-school/events.npz`; it takes precedence over the K=2 model input. All eight original heuristic event arrays and their row order are preserved. Existing `backend/` input layouts are also accepted when present. To run a separate complete raw dataset, point `--raw-dir` to a directory containing `<dataset>-nverts.txt`, `<dataset>-times.txt`, and `<dataset>-simplices.txt`.

AP uses `sklearn.metrics.average_precision_score` on continuous scores. Lift is AP divided by the evaluated candidates' positive prevalence. New predictions and metrics are written under `results/heuristics/`, and preprocessing under `results/heuristics_inputs/`. Both directories are separated by the input content hash. Katz and global methods can require substantial RAM because sparse inverse matrices may become dense.

The former two-step entry points also accept the same dataset and input options:

```bash
python supplementary_sources/heuristics/Heuristics_using_HodgeLaplacian/saving.py --dataset email-Enron
python supplementary_sources/heuristics/Heuristics_using_HodgeLaplacian/main.py --dataset email-Enron --method RA
```
