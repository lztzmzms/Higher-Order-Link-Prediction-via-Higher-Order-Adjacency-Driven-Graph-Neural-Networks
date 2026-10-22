from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cross_order_saturation_completion_v1"
PREDICTIONS = ROOT / "results" / "k4_strict_feature_nested_saturation_v2" / "K4_V2_PREDICTIONS.parquet"
TAGS_ASK_RESULTS = ROOT / "results" / "k4_ensemble_first_formal_layer3" / "K4_ENSEMBLE_FIRST_FORMAL_LAYER3_RESULTS.csv"
TAGS_ASK_BOOTSTRAP = ROOT / "results" / "k4_ensemble_first_formal_layer3" / "K4_ENSEMBLE_FIRST_FORMAL_LAYER3_BOOTSTRAP.csv"

DATASETS = ["email-Eu", "tags-math-sx", "tags-ask-ubuntu"]
MODELS = ["C0*", "C1*", "C2*", "C3*", "Full"]
SEEDS = [21001, 21002, 21003, 21004, 21005]
MARGIN = 0.02
BOOTSTRAPS = 2000
BOOT_SEEDS = {"email-Eu": [72101, 72102, 72103, 72104], "tags-math-sx": [72201, 72202, 72203, 72204]}
DIMENSIONS = {"C0*": 17, "C1*": 56, "C2*": 95, "C3*": 134, "Full": 56}


def losses(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def paired_bootstrap(y, left, right, seed):
    diff = losses(y, left) - losses(y, right)
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAPS, dtype=np.float64)
    for i in range(BOOTSTRAPS):
        values[i] = float(np.mean(diff[rng.integers(0, len(diff), size=len(diff))]))
    low = float(np.quantile(values, 0.025))
    high = float(np.quantile(values, 0.975))
    return {
        "D_LL": float(np.mean(diff)), "bootstrap_SD": float(np.std(values)),
        "CI_lower": low, "CI_upper": high, "NI": "YES" if high <= MARGIN else "NO",
        "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": seed,
    }


def v2_rows():
    frame = pq.read_table(PREDICTIONS).to_pandas()
    frame = frame[frame["dataset"].isin(["email-Eu", "tags-math-sx"])].copy()
    rows = []
    audits = []
    for dataset in ["email-Eu", "tags-math-sx"]:
        d = frame[frame["dataset"] == dataset]
        ref = d[(d["model"] == "C0*") & (d["seed"] == SEEDS[0])].sort_values("candidate_id")
        ids = ref["candidate_id"].to_numpy()
        vertices = ref["candidate_vertices"].to_numpy()
        y = ref["y_true"].to_numpy(dtype=np.int8)
        ens = {}
        for model in MODELS:
            sub = d[d["model"] == model]
            pivot = sub.pivot(index="candidate_id", columns="seed", values="p_pred").sort_index()
            if list(pivot.columns) != SEEDS:
                raise RuntimeError(f"seed coverage mismatch {dataset}/{model}")
            ens[model] = pivot[SEEDS].to_numpy(dtype=np.float64).mean(axis=1)
            for seed in SEEDS:
                one = sub[sub["seed"] == seed].sort_values("candidate_id")
                ok = np.array_equal(one["candidate_id"].to_numpy(), ids) and np.array_equal(one["candidate_vertices"].to_numpy(), vertices) and np.array_equal(one["y_true"].to_numpy(dtype=np.int8), y)
                audits.append({"dataset": dataset, "model": model, "seed": seed, "n_eval": len(one), "candidate_alignment": "PASS" if ok else "FAIL"})
                if not ok:
                    raise RuntimeError(f"candidate alignment failed {dataset}/{model}/{seed}")
        n_eval, n_pos = len(y), int(y.sum())
        ll = {model: float(np.mean(losses(y, ens[model]))) for model in MODELS}
        for index, model in enumerate(MODELS):
            if model == "Full":
                stats = {"D_LL": 0.0, "bootstrap_SD": "", "CI_lower": "", "CI_upper": "", "NI": "", "bootstrap_replicates": "", "bootstrap_seed": ""}
            else:
                stats = paired_bootstrap(y, ens[model], ens["Full"], BOOT_SEEDS[dataset][index])
            rows.append({
                "K": 4, "Dataset": dataset, "Stage_index": index, "Stage_label": model,
                "Dimension": DIMENSIONS[model], "LL_rep": ll[model], "LL_Full": ll["Full"], **stats,
                "N_eval": n_eval, "N_positive": n_pos, "prevalence": float(n_pos / n_eval),
                "protocol": "ENSEMBLE_FIRST_T2_FULL_UNIVERSE_PAIRED_CANDIDATE_BOOTSTRAP",
                "status": "RECONSTRUCTED_FROM_FROZEN_PREDICTIONS",
                "source": str(PREDICTIONS),
            })
    return rows, audits


def tags_ask_rows():
    metrics = pd.read_csv(TAGS_ASK_RESULTS)
    boots = pd.read_csv(TAGS_ASK_BOOTSTRAP)
    d = metrics[metrics["dataset"] == "tags-ask-ubuntu"]
    full = float(d[d["model"] == "Full"].iloc[0]["ensemble_LogLoss"])
    n_eval = int(d.iloc[0]["n_eval"])
    n_pos = int(d.iloc[0]["eval_positives"])
    rows = []
    for index, model in enumerate(MODELS):
        m = d[d["model"] == model].iloc[0]
        if model == "Full":
            stats = {"D_LL": 0.0, "bootstrap_SD": "", "CI_lower": "", "CI_upper": "", "NI": "", "bootstrap_replicates": "", "bootstrap_seed": ""}
        else:
            b = boots[(boots["dataset"] == "tags-ask-ubuntu") & (boots["model"] == model)].iloc[0]
            stats = {"D_LL": float(b["D_LL"]), "bootstrap_SD": float(b["bootstrap_sd"]), "CI_lower": float(b["CI_lower"]), "CI_upper": float(b["CI_upper"]), "NI": "YES" if float(b["CI_upper"]) <= MARGIN else "NO", "bootstrap_replicates": int(b["bootstrap_replicates"]), "bootstrap_seed": int(b["bootstrap_seed"])}
        rows.append({
            "K": 4, "Dataset": "tags-ask-ubuntu", "Stage_index": index, "Stage_label": model,
            "Dimension": DIMENSIONS[model], "LL_rep": float(m["ensemble_LogLoss"]), "LL_Full": full, **stats,
            "N_eval": n_eval, "N_positive": n_pos, "prevalence": float(n_pos / n_eval),
            "protocol": "ENSEMBLE_FIRST_T2_FULL_UNIVERSE_PAIRED_CANDIDATE_BOOTSTRAP",
            "status": "EXISTING_FORMAL_ENSEMBLE_FIRST",
            "source": f"{TAGS_ASK_RESULTS};{TAGS_ASK_BOOTSTRAP}",
        })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, audits = v2_rows()
    rows += tags_ask_rows()
    pd.DataFrame(rows).sort_values(["Dataset", "Stage_index"]).to_csv(OUT / "K4_SHARED5_SATURATION.csv", index=False)
    pd.DataFrame(audits).to_csv(OUT / "K4_CANDIDATE_ALIGNMENT_AUDIT.csv", index=False)
    manifest = {
        "status": "COMPLETE_FOR_FEASIBLE_SHARED_DATASETS",
        "datasets_complete": ["email-Eu", "tags-math-sx", "tags-ask-ubuntu"],
        "datasets_not_run": ["NDC-substances", "threads-math-sx"],
        "seeds": SEEDS, "bootstrap_replicates": BOOTSTRAPS, "margin": MARGIN,
        "ensemble_first": True, "strict_T2": True, "full_candidate_universe": True,
        "candidate_sampling": "NO", "model_retraining": "NO",
        "sources": [str(PREDICTIONS), str(TAGS_ASK_RESULTS), str(TAGS_ASK_BOOTSTRAP)],
    }
    (OUT / "K4_SHARED5_RECONSTRUCTION_AUDIT.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
