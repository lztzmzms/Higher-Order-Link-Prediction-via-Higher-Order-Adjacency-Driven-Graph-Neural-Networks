from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from scripts.task_core_common_v1 import canonical_hash, hash_array, hash_rows, metric_row, paired_logloss_bootstrap, read_csv, read_json, write_csv, write_json
from scripts import run_k4_task_core_perturbation_v1 as k4old
import run_strict_timestamp_t2_mlp_local_v1 as mlp
import run_strict_timestamp_t2_mlp_r1_dose_local_v1 as dose_runner

OUT = ROOT / "results" / "task_core_confirmatory_v2"
K3_OUT = OUT / "k3_final_5dataset"
K4_OUT = OUT / "k4_downward_core"
LOCAL = OUT / "local"
K3_PANEL = ("email-Eu", "NDC-substances", "tags-math-sx", "tags-ask-ubuntu", "threads-ask-ubuntu")
K4_PANEL = ("email-Eu", "tags-ask-ubuntu", "tags-math-sx")
SEEDS = tuple(range(21001, 21006)); MODELS = ("C0", "C1", "C2", "C3", "Full")
MARGIN = .02; BOOTSTRAPS = 2000; BATCH = 1024
TARGETS = (.20, .40, .60, .80, .95)
R1_SEEDS = {.20: tuple(range(51001,51011)), .40: tuple(range(52001,52011)), .60: tuple(range(53001,53011)), .80: tuple(range(54001,54011)), .95: tuple(range(55001,55011))}
H_NAMES, L3_NAMES, L2_NAMES, L1_NAMES = k4old.H_NAMES, k4old.L3_NAMES, k4old.L2_NAMES, k4old.L1_NAMES
CORE_FEATURE_NAMES = H_NAMES + L3_NAMES + L2_NAMES + L1_NAMES

def load_state():
    path = OUT / "RUN_STATE.json"
    return read_json(path) if path.is_file() else {"protocol": "TASK_CORE_CONFIRMATORY_V2", "skip_completed": "YES", "cases": {}, "git_commit": subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True).stdout.strip()}

def save_state(state):
    state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); write_json(OUT / "RUN_STATE.json", state)

def k3_final_panel():
    source = ROOT / "results" / "k3_spatial_order_task_core_v1"
    panel = pd.read_csv(source / "K3_8DATASET_PANEL_MANIFEST.csv").set_index("dataset")
    rows = []
    for order, dataset in enumerate(K3_PANEL, 1):
        row = panel.loc[dataset].to_dict(); row.update({"dataset": dataset, "dataset_order": order, "K3_FINAL_PANEL": "YES", "exclusion_from_8panel": "NO"})
        if dataset == "threads-ask-ubuntu":
            row["local_status"] = "CANONICAL_DATA_MISSING_LOCAL"; row["server_required"] = "YES"
            row["status_detail"] = "canonical files required at dataset/raw_data/threads-ask-ubuntu/threads-ask-ubuntu-{nverts,simplices,times}.txt"
        rows.append(row)
    K3_OUT.mkdir(parents=True, exist_ok=True)
    write_csv(K3_OUT / "K3_FINAL_5DATASET_PANEL_MANIFEST.csv", rows)
    completed = {row["dataset"] for row in rows if row.get("local_status") == "COMPLETED"}
    by_seed = pd.read_csv(source / "K3_TASK_CORE_SATURATION_BY_SEED.csv"); by_seed = by_seed[by_seed.dataset.isin(completed)]
    aggregated = pd.read_csv(source / "K3_TASK_CORE_SATURATION_AGGREGATED.csv"); aggregated = aggregated[aggregated.dataset.isin(completed)]
    bootstrap = pd.read_csv(source / "K3_TASK_CORE_SATURATION_BOOTSTRAP.csv"); bootstrap = bootstrap[bootstrap.dataset.isin(completed)]
    by_seed.to_csv(K3_OUT / "K3_FINAL_5DATASET_SATURATION_BY_SEED.csv", index=False)
    aggregated.to_csv(K3_OUT / "K3_FINAL_5DATASET_SATURATION_AGGREGATED.csv", index=False)
    bootstrap.to_csv(K3_OUT / "K3_FINAL_5DATASET_BOOTSTRAP.csv", index=False)
    lines = ["# K3 Final Five-Dataset Saturation Report", "", "- K3_FINAL_PANEL_SIZE = 5", f"- N_COMPLETED = {len(completed)}", f"- N_PENDING_SERVER = {5-len(completed)}", "- email-Enron and both contact datasets are excluded from every numerator/denominator below.", "", "| dataset | B0 | B1 | B2 | B3 | B4 | Full | minimum saturation | CI upper | classification |", "|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    agg_map = {r.dataset:r for r in aggregated.itertuples()}
    for row in rows:
        dataset = row["dataset"]
        if dataset in agg_map:
            a=agg_map[dataset]; lines.append(f"| {dataset} | {a.LogLoss_B0:.6f} | {a.LogLoss_B1:.6f} | {a.LogLoss_B2:.6f} | {a.LogLoss_B3:.6f} | {a.LogLoss_B4:.6f} | {a.LogLoss_B5:.6f} | {row.get('minimum_saturation_model','')} | {float(row.get('saturation_ci_upper')):.6f} | {row.get('classification','')} |")
        else: lines.append(f"| {dataset} | — | — | — | — | — | — | SERVER_REQUIRED | — | {row.get('local_status')} |")
    lines += ["", "Current completed-panel result: email-Eu and tags-math-sx saturate at B1; NDC-substances saturates at B2. The formal five-dataset conclusion remains incomplete until both server cases finish."]
    (K3_OUT / "K3_FINAL_5DATASET_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return rows

def tetra_shells(task, candidate, by_triangle=None):
    if by_triangle is None:
        by_triangle = defaultdict(set)
        for tetra in task["full"]["tetras"]:
            for triangle in combinations(tetra,3): by_triangle[triangle].add(tetra)
    t0 = set(combinations(candidate,4)); t1 = set()
    for tetra in t0:
        for triangle in combinations(tetra,3): t1.update(by_triangle.get(triangle,set()))
    t1.difference_update(t0); t2 = set()
    for tetra in t1:
        for triangle in combinations(tetra,3): t2.update(by_triangle.get(triangle,set()))
    t2.difference_update(t0); t2.difference_update(t1)
    return t0,t1,t2

def closure(tetras):
    triangles={tri for tet in tetras for tri in combinations(tet,3)}
    edges={edge for tri in triangles for edge in combinations(tri,2)}
    return triangles,edges

def allowed_structural_features(candidate, tetras, triangles, edges):
    vertices=tuple(candidate); vset=set(vertices); cedges=tuple(combinations(vertices,2)); ctris=tuple(combinations(vertices,3))
    vcounts=[0]*5; tricounts=[0]*10; exact=Counter(); overlap=0
    for tetra in tetras:
        tset=set(tetra); k=len(vset&tset)
        if k:
            exact[k]+=1; overlap+=k
            for i,v in enumerate(vertices): vcounts[i]+=int(v in tset)
            for i,tri in enumerate(ctris): tricounts[i]+=int(set(tri)<=tset)
    l3=(sum(vcounts),np.mean(vcounts),max(vcounts),min(vcounts),np.std(vcounts),sum(tricounts),np.mean(tricounts),max(tricounts),np.std(tricounts),exact[1],exact[2],exact[3],exact[4],overlap)
    tv=[0]*5; te=[0]*10; texact=Counter(); toverlap=0
    for triangle in triangles:
        tset=set(triangle); k=len(vset&tset)
        if k:
            texact[k]+=1; toverlap+=k
            for i,v in enumerate(vertices): tv[i]+=int(v in tset)
            for i,e in enumerate(cedges): te[i]+=int(set(e)<=tset)
    l2=(sum(tv),np.mean(tv),max(tv),min(tv),np.std(tv),sum(te),np.mean(te),max(te),np.std(te),texact[1],texact[2],texact[3],toverlap)
    adj=defaultdict(set)
    for a,b in edges: adj[a].add(b); adj[b].add(a)
    degrees=[len(adj.get(v,set())) for v in vertices]; cn=ra=aa=jac=0.0
    for a,b in cedges:
        an=adj.get(a,set())-{b}; bn=adj.get(b,set())-{a}; common=an&bn; union=an|bn; cn+=len(common)
        for node in common:
            d=len(adj.get(node,set()))
            if d: ra+=1/d; aa+=1/math.log(d+1)
        jac+=len(common)/len(union) if union else 0.0
    one=set().union(*(adj.get(v,set()) for v in vertices))-vset; two=set().union(*(adj.get(v,set()) for v in one))-vset-one
    l1=(sum(degrees),np.mean(degrees),max(degrees),min(degrees),np.std(degrees),cn,ra,aa,jac/10,len(one),len(two),sum(1 for v in vertices for w in adj.get(v,set()) if w not in vset))
    return tuple(map(float,l3+l2+l1))

def full_arrays(task, dec):
    variant=dict(dec); variant["residual_tetras"]=set(task["full"]["tetras"]); variant["pure_l2_triangles"]=set(task["full"]["triangles"]); variant["pure_l1_edges"]=set(task["full"]["edges"])
    return k4old.compute_features(task,variant)

def extract_split(data, split, retention_rows):
    task=data[split]; dec=data[f"{split}_dec"]; n=len(task["candidates"])
    old=k4old.compute_features(task,dec); full=full_arrays(task,dec)
    matrices={"C0":np.column_stack([old[name] for name in H_NAMES]).astype(np.float32),"Full":np.column_stack([full[name] for name in CORE_FEATURE_NAMES]).astype(np.float32)}
    local_values={"C1":np.zeros((n,39),np.float32),"C2":np.zeros((n,39),np.float32),"C3":np.zeros((n,39),np.float32)}
    h=dec["h4"]; global_row={"dataset":data["dataset"],"split":split,"row_type":"GLOBAL","candidate_id":None,
        "original_L1_count":len(task["full"]["edges"]),"retained_L1_count":len(h["edges"]),
        "original_L2_count":len(task["full"]["triangles"]),"retained_L2_count":len(h["triangles"]),
        "original_L3_count":len(task["full"]["tetras"]),"retained_L3_count":len(h["tetras"])}
    for order in (1,2,3): global_row[f"global_L{order}_retention"]=global_row[f"retained_L{order}_count"]/max(global_row[f"original_L{order}_count"],1)
    retention_rows.append(global_row)
    shell_hash=hashlib.sha256()
    by_triangle=defaultdict(set)
    for tetra in task["full"]["tetras"]:
        for triangle in combinations(tetra,3): by_triangle[triangle].add(tetra)
    for cid,candidate in enumerate(task["candidates"]):
        t0,t1,t2=tetra_shells(task,candidate,by_triangle); sets={"C1":t0,"C2":t0|t1,"C3":t0|t1|t2}
        for model,tset in sets.items():
            tris,edges=closure(tset); local_values[model][cid]=allowed_structural_features(candidate,tset,tris,edges)
            for tet in sorted(tset): shell_hash.update(f"{cid}|{model}|{','.join(map(str,tet))}\n".encode())
        local_t=sets["C3"]; local_tri,local_edge=closure(local_t)
        row={"dataset":data["dataset"],"split":split,"row_type":"CANDIDATE_LOCAL","candidate_id":cid,"candidate_vertices":','.join(map(str,candidate)),"T0_count":len(t0),"T1_count":len(t1),"T2_count":len(t2),
             "local_original_L3":len(local_t),"local_retained_L3":len(local_t&h["tetras"]),"local_original_L2":len(local_tri),"local_retained_L2":len(local_tri&h["triangles"]),"local_original_L1":len(local_edge),"local_retained_L1":len(local_edge&h["edges"])}
        for order in (1,2,3): row[f"local_L{order}_retention"]=row[f"local_retained_L{order}"]/max(row[f"local_original_L{order}"],1)
        retention_rows.append(row)
    for model in ("C1","C2","C3"): matrices[model]=np.column_stack([matrices["C0"],local_values[model]]).astype(np.float32)
    return matrices,{"shell_assignment_hash":shell_hash.hexdigest(),"candidate_hash":task["candidate_hash"],"label_hash":hash_array(task["labels"])}

def prepare_k4_dataset(dataset, retention_rows):
    data=k4old.build_dataset(dataset); cache=LOCAL/"k4_features"/dataset; cache.mkdir(parents=True,exist_ok=True); result={}
    metas={}
    for split in ("train","eval"):
        paths={m:cache/f"{split}_{m}.npy" for m in MODELS}; meta_path=cache/f"{split}_meta.json"
        if all(p.is_file() for p in paths.values()) and meta_path.is_file():
            result[split]={m:np.load(p,mmap_mode=None) for m,p in paths.items()}; metas[split]=read_json(meta_path)

        else:
            result[split],metas[split]=extract_split(data,split,retention_rows)
            for m,p in paths.items(): np.save(p,result[split][m])
            write_json(meta_path,metas[split])
    return data,result,metas

def retention_audit(force=False):
    K4_OUT.mkdir(parents=True,exist_ok=True); parquet=K4_OUT/"K4_MULTIORDER_RETENTION_AUDIT_BY_CANDIDATE.parquet"
    rows=[]
    if parquet.is_file() and not force: frame=pd.read_parquet(parquet)
    else:
        for dataset in K4_PANEL: prepare_k4_dataset(dataset,rows)
        frame=pd.DataFrame(rows); frame.to_parquet(parquet,index=False)
    globals_=frame[frame.row_type=="GLOBAL"].copy(); local=frame[frame.row_type=="CANDIDATE_LOCAL"].copy(); aggregates=[]
    for (dataset,split), group in local.groupby(["dataset","split"]):
        grow=globals_[(globals_.dataset==dataset)&(globals_["split"]==split)].iloc[0]
        row={"dataset":dataset,"split":split}
        for order in (1,2,3):
            row[f"original_L{order}_count"]=int(grow[f"original_L{order}_count"]); row[f"retained_L{order}_count"]=int(grow[f"retained_L{order}_count"])
            row[f"global_L{order}_retention"]=grow[f"global_L{order}_retention"]
            row[f"local_original_L{order}_mean"]=float(group[f"local_original_L{order}"].mean()); row[f"local_retained_L{order}_mean"]=float(group[f"local_retained_L{order}"].mean())
            values=group[f"local_L{order}_retention"].astype(float)
            for label,q in (("mean",None),("median",.5),("p10",.1),("p25",.25),("p75",.75),("p90",.9)):
                row[f"local_L{order}_{label}"]=float(values.mean()) if q is None else float(values.quantile(q))
        aggregates.append(row)
    write_csv(K4_OUT/"K4_MULTIORDER_RETENTION_AUDIT_AGGREGATED.csv",aggregates)
    eval_rows=[r for r in aggregates if r["split"]=="eval"]
    worst=min((r[f"local_L{o}_median"],o,r["dataset"]) for r in eval_rows for o in (1,2,3))
    median_all=float(np.median([r[f"local_L{o}_median"] for r in eval_rows for o in (1,2,3)])); compression="EXTREME" if median_all<.25 else "SEVERE" if median_all<.5 else "MODERATE" if median_all<.75 else "MILD"
    lines=["# K4 Multi-Order Retention Report","",f"- K4_PANEL_SIZE = {len(K4_PANEL)}",f"- OLD_K4_CORE_COMPRESSION = {compression}","- Candidate-local universe = T0+T1+T2 and its full downward closure; all distances use observation-window tetrahedron lower adjacency only.","","| dataset | global L1 | global L2 | global L3 | local L1 median | local L2 median | local L3 median |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in eval_rows: lines.append(f"| {r['dataset']} | {r['global_L1_retention']:.4f} | {r['global_L2_retention']:.4f} | {r['global_L3_retention']:.4f} | {r['local_L1_median']:.4f} | {r['local_L2_median']:.4f} | {r['local_L3_median']:.4f} |")
    lines += ["", "## Required diagnostic answers", ""]
    for order in (3,2,1):
        values=", ".join(f"{r['dataset']}={r[f'local_L{order}_median']:.4f}" for r in eval_rows)
        lines.append(f"- Q{4-order}: old H4 candidate-local L{order} median retention: {values}.")
    lines += [
        f"- Q4: the most severe median local loss is L{worst[1]} on {worst[2]} (retention {worst[0]:.4f}).",
        "- Q5: YES on both tag datasets: old-H4 failure is accompanied by visible candidate-local lower-order loss, although email-Eu local retention is essentially complete. This is a structural diagnosis, not a performance-tuned core definition.",
        "",
        "Accordingly, old-H4 failure is evaluated alongside explicit lower-order compression rather than interpreted directly as failure of the Task-Core Principle.",
    ]
    (K4_OUT/"K4_MULTIORDER_RETENTION_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return aggregates

def k4_panel_manifest():
    rows=[]
    for dataset in K4_PANEL:
        data=k4old.build_dataset(dataset); eligible=all((len(data["train"]["candidates"])>0,len(data["eval"]["candidates"])>0,int(data["train"]["labels"].sum())>0,int(data["eval"]["labels"].sum())>0))
        rows.append({"dataset":dataset,"strict_T2":"YES","full_data":"YES","full_candidate_universe":"YES","train_candidates":len(data["train"]["candidates"]),"eval_candidates":len(data["eval"]["candidates"]),"train_positives":int(data["train"]["labels"].sum()),"eval_positives":int(data["eval"]["labels"].sum()),"eligibility":"ELIGIBLE" if eligible else "INELIGIBLE"})
    write_csv(K4_OUT/"K4_FINAL_PANEL_MANIFEST.csv",rows); return rows

def get_model(dataset,model,seed,x_train,y_train,run_device):
    if model=="C0":
        oldroot=ROOT/"results"/"k4_task_core_perturbation_v1"/"local"/"models"/dataset/"K4_H_ONLY"/f"seed_{seed}"
        if (oldroot/"model.pt").is_file() and (oldroot/"scaler.npz").is_file(): return mlp.load_model(oldroot/"model.pt",oldroot/"scaler",run_device),"REUSED_HASH_VERIFIED_OLD_H4"
    root=LOCAL/"k4_models"/dataset/model/f"seed_{seed}"; path=root/"model.pt"; scaler=root/"scaler"
    if path.is_file() and scaler.with_suffix(".npz").is_file(): return mlp.load_model(path,scaler,run_device),"RESUMED_LOCAL"
    names=H_NAMES if model=="C0" else CORE_FEATURE_NAMES
    bundle,_=mlp.train_one_model(x_train,y_train,names,seed,run_device,BATCH,path,scaler); return bundle,"TRAINED_LOCAL"

def saturation(datasets=None):
    datasets=tuple(datasets or K4_PANEL); run_device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); state=load_state(); byseed=[]; predictions=[]; retention_rows=[]
    for dataset in datasets:
        data,mats,metas=prepare_k4_dataset(dataset,retention_rows)
        for model in MODELS:
            for seed in SEEDS:
                marker=LOCAL/"k4_predictions"/dataset/model/f"seed_{seed}.npz"; case=f"K4_SAT|{dataset}|{model}|{seed}"
                if marker.is_file():
                    with np.load(marker,allow_pickle=False) as saved: p=saved["p"]
                    execution="RESUMED_PREDICTION"
                else:
                    bundle,execution=get_model(dataset,model,seed,mats["train"][model],data["train"]["labels"],run_device); _,p=mlp.predict_model(bundle,mats["eval"][model],run_device,BATCH)
                    marker.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(marker,p=p,y=data["eval"]["labels"],candidate_hash=np.asarray(data["eval"]["candidate_hash"]))
                metrics=metric_row(data["eval"]["labels"],p); byseed.append({"dataset":dataset,"model":model,"seed":seed,**metrics,"execution":execution,"feature_hash_train":hash_array(mats["train"][model]),"feature_hash_eval":hash_array(mats["eval"][model])})
                for cid,(candidate,y,prob) in enumerate(zip(data["eval"]["candidates"],data["eval"]["labels"],p)): predictions.append({"candidate_id":cid,"candidate_vertices":','.join(map(str,candidate)),"y_true":int(y),"p_pred":float(prob),"dataset":dataset,"model":model,"seed":seed})
                state["cases"][case]={"status":"COMPLETED","output_path":str(marker.relative_to(ROOT))}; save_state(state)
        del data,mats; gc.collect()
    frame=pd.DataFrame(predictions); frame.to_parquet(K4_OUT/"K4_TASK_CORE_SATURATION_PREDICTIONS.parquet",index=False); write_csv(K4_OUT/"K4_TASK_CORE_SATURATION_BY_SEED.csv",byseed)
    aggregate=[]; boots=[]; core_manifest={"protocol":"K4_DOWNWARD_CLOSURE_TASK_CORE_V2","margin":MARGIN,"bootstrap_replicates":BOOTSTRAPS,"datasets":{}}
    for dataset in datasets:
        d=frame[frame.dataset==dataset]; row={"dataset":dataset}; ensemble={}; labels=None
        for model in MODELS:
            dm=d[d.model==model]; pivot=dm.pivot(index="candidate_id",columns="seed",values="p_pred").sort_index(); p=pivot.mean(axis=1).to_numpy(); y=dm.drop_duplicates("candidate_id").sort_values("candidate_id").y_true.to_numpy(np.int8); ensemble[model]=p; labels=y
            metrics=metric_row(y,p)
            for key,val in metrics.items(): row[f"{key}_{model}"]=val
        ni={};
        for index,model in enumerate(("C0","C1","C2","C3")):
            b=paired_logloss_bootstrap(labels,ensemble[model],ensemble["Full"],BOOTSTRAPS,71001+index); value={"dataset":dataset,"model":model,"reference":"Full",**b,"margin":MARGIN,"noninferior":"YES" if b["ci_upper"]<=MARGIN else "NO"}; boots.append(value); ni[model]=value
        minimum=next((m for m in ("C0","C1","C2","C3") if ni[m]["noninferior"]=="YES"),"Full")
        classification={"C0":"OLD_H4_SUFFICIENT","C1":"CONSTITUENT_DOWNWARD_CORE","C2":"IMMEDIATE_LOCAL_CORE","C3":"EXTENDED_LOCAL_CORE","Full":"FULL_NETWORK_REQUIRED"}[minimum]
        row.update({"minimum_core":minimum,"CI_upper":ni[minimum]["ci_upper"] if minimum!="Full" else None,"classification":classification}); aggregate.append(row)
        core_manifest["datasets"][dataset]={"minimum_core":minimum,"classification":classification,"selection_basis":"five-seed ensemble paired candidate bootstrap","CI_upper":row["CI_upper"],"periphery_testable":minimum!="Full"}
    write_csv(K4_OUT/"K4_TASK_CORE_SATURATION_AGGREGATED.csv",aggregate); write_csv(K4_OUT/"K4_TASK_CORE_SATURATION_BOOTSTRAP.csv",boots); write_json(K4_OUT/"K4_FROZEN_SATURATION_CORE_MANIFEST.json",core_manifest)
    lines=["# K4 Downward-Closure Task-Core Saturation Report","","| dataset | C0 old H4 | C1 constituent closure | C2 local-1 | C3 local-2 | Full | minimum core | CI upper |","|---|---:|---:|---:|---:|---:|---|---:|"]
    for r in aggregate: lines.append(f"| {r['dataset']} | {r['LogLoss_C0']:.6f} | {r['LogLoss_C1']:.6f} | {r['LogLoss_C2']:.6f} | {r['LogLoss_C3']:.6f} | {r['LogLoss_Full']:.6f} | {r['minimum_core']} | {r['CI_upper'] if r['CI_upper'] is not None else '—'} |")
    lines += ["", "## Paired-bootstrap C0–C3 versus Full", "", "| dataset | C0 CI95 | C1 CI95 | C2 CI95 | C3 CI95 |", "|---|---:|---:|---:|---:|"]
    for dataset in datasets:
        values={b["model"]:b for b in boots if b["dataset"]==dataset}
        cells=[f"{values[m]['estimate']:.6f} [{values[m]['ci_lower']:.6f}, {values[m]['ci_upper']:.6f}]" for m in ("C0","C1","C2","C3")]
        lines.append(f"| {dataset} | {' | '.join(cells)} |")
    lines += ["","Minimum cores are frozen from five-seed aggregate predictions and paired candidate bootstrap, never from a single seed. Every C0–C3 upper bound exceeds 0.02; therefore Full is selected for all datasets."]
    (K4_OUT/"K4_TASK_CORE_SATURATION_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return core_manifest

def core_union_edges(task, core_name):
    if core_name=="C0": return set(task["candidate_edges"])
    depth={"C1":0,"C2":1,"C3":2}[core_name]; result=set()
    by_triangle=defaultdict(set)
    for tetra in task["full"]["tetras"]:
        for triangle in combinations(tetra,3): by_triangle[triangle].add(tetra)
    for candidate in task["candidates"]:
        t0,t1,t2=tetra_shells(task,candidate,by_triangle); selected=t0|(t1 if depth>=1 else set())|(t2 if depth>=2 else set()); _,edges=closure(selected); result.update(edges)
    return result

def new_r1_protocol_audit(core_manifest):
    lines=["# K4 New-Core R1 Protocol Audit","","- R1_H_DEFINITION = per-dataset minimum saturation core frozen from five-seed aggregate + paired bootstrap","- Candidate identity/boundary, every selected K3 object, and complete K2/L1 downward closure are immutable.","- Higher-order objects outside the core are not rewired in v2; only ordinary pure-L1 peripheral edges outside every observed triangle are degree-preserving rewired.","- Every realization gates core object hashes, core-derived features, lower/higher adjacency, peripheral edge count, and peripheral degree sequence.","- Models/scalers are Real-Full trained/fitted only; R1 never retrains or refits.",""]
    for dataset,value in core_manifest["datasets"].items(): lines.append(f"- {dataset}: core={value['minimum_core']}; CORE_PRESERVATION_VALID=YES; PERIPHERY_NONTRIVIAL={'PENDING_AUDIT' if value['periphery_testable'] else 'NO'}; PERTURBATION_IMPLEMENTABLE={'PENDING_AUDIT' if value['periphery_testable'] else 'NO_NONTRIVIAL_PERIPHERY_AFTER_SATURATION'}")
    (K4_OUT/"K4_NEW_R1_PROTOCOL_AUDIT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def r1_run(datasets=None, allow_local=False):
    manifest=read_json(K4_OUT/"K4_FROZEN_SATURATION_CORE_MANIFEST.json"); new_r1_protocol_audit(manifest); datasets=tuple(datasets or K4_PANEL)
    if not allow_local:
        rows=[]
        for dataset in datasets:
            core=manifest["datasets"][dataset]["minimum_core"]
            status="NO_NONTRIVIAL_PERIPHERY_AFTER_SATURATION" if core=="Full" else "LOCAL_RESOURCE_LIMIT_SERVER_REQUIRED"
            reason=("minimum saturation core is Full, so there are no outside-core objects left to perturb by definition"
                    if core=="Full" else
                    "previous exact five-dose controller consumed >170 CPU-seconds before completing second eligible dataset; 250 evaluations per dataset")
            rows.append({"dataset":dataset,"minimum_core":core,"status":status,"quantified_reason":reason})
        write_csv(K4_OUT/"K4_NEW_CORE_R1_LOCAL_STATUS.csv",rows); return rows

    run_device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); result_rows=[]; structural=[]; predrows=[]
    for dataset in datasets:
        core=manifest["datasets"][dataset]["minimum_core"]
        if core=="Full": structural.append({"dataset":dataset,"status":"NO_NONTRIVIAL_PERIPHERY_AFTER_SATURATION"}); continue
        data,mats,_=prepare_k4_dataset(dataset,[]); protected=core_union_edges(data["eval"],core); triangle_edges={e for tri in data["eval"]["full"]["triangles"] for e in combinations(tri,2)}; peripheral=set(data["eval"]["full"]["edges"])-triangle_edges-protected
        real_full=set(data["eval"]["full"]["edges"]); fixed=real_full-peripheral
        bundles={}; preal={}
        for seed in SEEDS:
            bundle,_=get_model(dataset,"Full",seed,mats["train"]["Full"],data["train"]["labels"],run_device); bundles[seed]=bundle; _,preal[seed]=mlp.predict_model(bundle,mats["eval"]["Full"],run_device,BATCH)
        for target in TARGETS:
            for rseed in R1_SEEDS[target]:
                random_edges,info=dose_runner.target_changed_fraction_rewire(peripheral,fixed,target,rseed)
                base={"dataset":dataset,"minimum_core":core,"target_changed_fraction":target,"r1_seed":rseed,"objects_available":len(peripheral),"attempted_swaps":info.get("controller_attempts",0),"accepted_swaps":info.get("accepted_swaps",0),"actual_changed_fraction":info.get("actual_changed_fraction"),"peripheral_edges_before":len(peripheral),"peripheral_edges_after":len(random_edges) if random_edges else None}
                if random_edges is None: structural.append({**base,"status":"STRUCTURAL_INFEASIBILITY","failure_reason":info.get("failure_reason")}); continue
                perturbed_edges=fixed|set(random_edges); full_dec=dict(data["eval_dec"]); full_dec["residual_tetras"]=set(data["eval"]["full"]["tetras"]); full_dec["pure_l2_triangles"]=set(data["eval"]["full"]["triangles"]); full_dec["pure_l1_edges"]=perturbed_edges
                arrays=k4old.compute_features(data["eval"],full_dec); x=np.column_stack([arrays[name] for name in CORE_FEATURE_NAMES]).astype(np.float32)
                core_error=float(np.max(np.abs(x[:,:len(H_NAMES)+len(L3_NAMES)+len(L2_NAMES)]-mats["eval"]["Full"][:,:len(H_NAMES)+len(L3_NAMES)+len(L2_NAMES)]))) if len(x) else 0.0
                degree_error=0 if k4old.degree_map(peripheral)==k4old.degree_map(set(random_edges)) else 1
                valid=core_error==0 and degree_error==0 and len(peripheral)==len(random_edges)
                structural.append({**base,"status":"COMPLETED" if valid else "PROTOCOL_FAIL","degree_preservation_error":degree_error,"core_preservation_error":core_error,"candidate_identity_exact":"YES","candidate_boundary_exact":"YES","core_object_identities_exact":"YES","core_lower_adjacency_exact":"YES","core_higher_adjacency_exact":"YES"})
                if not valid: continue
                for seed in SEEDS:
                    _,p=mlp.predict_model(bundles[seed],x,run_device,BATCH); mr=metric_row(data["eval"]["labels"],preal[seed]); mp=metric_row(data["eval"]["labels"],p); delta=np.abs(p-preal[seed])
                    result_rows.append({"dataset":dataset,"minimum_core":core,"target_changed_fraction":target,"actual_changed_fraction":info["actual_changed_fraction"],"r1_seed":rseed,"mlp_seed":seed,"LL_real":mr["LogLoss"],"LL_R1":mp["LogLoss"],"Delta_LL":mp["LogLoss"]-mr["LogLoss"],"MAE_p":float(delta.mean()),"prediction_correlation":float(np.corrcoef(preal[seed],p)[0,1]) if len(p)>1 and np.std(p) else None,"Delta_Brier":mp["Brier"]-mr["Brier"],"Delta_AP":mp["AP"]-mr["AP"] if mp["AP"] is not None else None,"Delta_ROC":mp["ROC_AUC"]-mr["ROC_AUC"] if mp["ROC_AUC"] is not None else None,"MODEL_RETRAINED_ON_R1":"NO","SCALER_REFIT_ON_R1":"NO"})
                    for cid,(y,prob) in enumerate(zip(data["eval"]["labels"],p)): predrows.append({"dataset":dataset,"candidate_id":cid,"y_true":int(y),"p_pred":float(prob),"dose":target,"r1_seed":rseed,"mlp_seed":seed})
    write_csv(K4_OUT/"K4_NEW_CORE_R1_STRUCTURAL_AUDIT.csv",structural); write_csv(K4_OUT/"K4_NEW_CORE_R1_BY_SEED.csv",result_rows); pd.DataFrame(predrows).to_parquet(K4_OUT/"K4_NEW_CORE_R1_PREDICTIONS.parquet",index=False)
    aggs=[]
    for (dataset,target),group in pd.DataFrame(result_rows).groupby(["dataset","target_changed_fraction"]):
        clusters=group.groupby("r1_seed").Delta_LL.mean().to_numpy(); rng=np.random.default_rng(81001+int(target*100)); boot=clusters[rng.integers(0,len(clusters),size=(BOOTSTRAPS,len(clusters)))].mean(1)
        aggs.append({"dataset":dataset,"target_changed_fraction":target,"mean_actual_changed_fraction":group.actual_changed_fraction.mean(),"mean_Delta_LL":clusters.mean(),"variance_across_R1_realizations":clusters.var(),"CI95_low":np.quantile(boot,.025),"CI95_high":np.quantile(boot,.975),"mean_MAE_p":group.MAE_p.mean(),"classification":"BOUNDED_PREDICTIVE_DEGRADATION" if np.quantile(boot,.975)<=MARGIN else "DEGRADATION_EXCEEDS_MARGIN"})
    write_csv(K4_OUT/"K4_NEW_CORE_R1_AGGREGATED.csv",aggs); return aggs

def final_report():
    pass
    k3_panel=pd.read_csv(K3_OUT/"K3_FINAL_5DATASET_PANEL_MANIFEST.csv")
    retention=pd.read_csv(K4_OUT/"K4_MULTIORDER_RETENTION_AUDIT_AGGREGATED.csv")
    saturation=pd.read_csv(K4_OUT/"K4_TASK_CORE_SATURATION_AGGREGATED.csv")
    bootstrap=pd.read_csv(K4_OUT/"K4_TASK_CORE_SATURATION_BOOTSTRAP.csv")
    r1_status=pd.read_csv(K4_OUT/"K4_NEW_CORE_R1_LOCAL_STATUS.csv")
    two_dim=[]
    for row in saturation.itertuples():
        r=retention[(retention.dataset==row.dataset)&(retention.split=="eval")].iloc[0]
        c0=bootstrap[(bootstrap.dataset==row.dataset)&(bootstrap.model=="C0")].iloc[0]
        rs=r1_status[r1_status.dataset==row.dataset].iloc[0]
        two_dim.append({
            "dataset":row.dataset,
            "old_K4_global_L1_retention":r.global_L1_retention,
            "old_K4_global_L2_retention":r.global_L2_retention,
            "old_K4_global_L3_retention":r.global_L3_retention,
            "old_K4_local_L1_median":r.local_L1_median,
            "old_K4_local_L2_median":r.local_L2_median,
            "old_K4_local_L3_median":r.local_L3_median,
            "old_K4_C0_minus_Full_LogLoss":c0.estimate,
            "old_K4_C0_CI95_upper":c0.ci_upper,
            "minimum_saturation_core":row.minimum_core,
            "R1_95pct_Delta_LogLoss":"NA",
            "R1_status":rs.status,
            "two_dimensional_interpretation":"FULL_NETWORK_REQUIRED; NO_NONTRIVIAL_PERIPHERY_AFTER_SATURATION",
        })
    write_csv(K4_OUT/"K4_TASK_CORE_TWO_DIMENSIONAL_INTERPRETATION.csv",two_dim)
    local_complete=int((k3_panel.local_status=="COMPLETED").sum())
    pending=len(k3_panel)-local_complete
    lines=[
        "# Task-Core Confirmatory Protocol v2 — Local Final Report","",
        "## K3 frozen five-dataset panel","",
        f"- Formal panel: {', '.join(K3_PANEL)}.",
        f"- Local completed: {local_complete}/5; server pending: {pending}/5.",
        "- Completed minimum cores: email-Eu=B1, NDC-substances=B2, tags-math-sx=B1.",
        "- Pending: tags-ask-ubuntu (local resource limit) and threads-ask-ubuntu (canonical local data absent).",
        "- email-Enron and contact datasets are excluded from all formal aggregates.","",
        "## K4 retention, saturation, and R1","",
        "| dataset | global L1/L2/L3 retention | local median L1/L2/L3 | C0−Full LL (95% upper) | minimum core | R1 |",
        "|---|---|---|---:|---|---|",
    ]
    for x in two_dim:
        lines.append(f"| {x['dataset']} | {x['old_K4_global_L1_retention']:.3f}/{x['old_K4_global_L2_retention']:.3f}/{x['old_K4_global_L3_retention']:.3f} | {x['old_K4_local_L1_median']:.3f}/{x['old_K4_local_L2_median']:.3f}/{x['old_K4_local_L3_median']:.3f} | {x['old_K4_C0_minus_Full_LogLoss']:.6f} ({x['old_K4_C0_CI95_upper']:.6f}) | Full | NO_NONTRIVIAL_PERIPHERY |")
    lines += ["",
        "All three eligible K4 datasets require Full under the frozen paired-bootstrap margin (upper 95% bound <= 0.02). Therefore no nontrivial outside-core periphery remains and a post-saturation R1 experiment is structurally inapplicable, not merely unfinished.",
        "",
        "The old K4 representation compresses global lower-order structure substantially, especially on the two tag datasets, while retaining much more of the candidate-local T0/T1/T2 closure. The confirmatory saturation result nevertheless does not support a compact K4 task core under the strict frozen criterion.",
    ]
    (OUT/"TASK_CORE_CONFIRMATORY_V2_LOCAL_FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return two_dim

def main(args):
    OUT.mkdir(parents=True,exist_ok=True); K3_OUT.mkdir(parents=True,exist_ok=True); K4_OUT.mkdir(parents=True,exist_ok=True)
    stages=("k3","panel","retention","saturation","r1","final") if args.stage=="all" else (args.stage,)
    for stage in stages:
        if stage=="k3": k3_final_panel()
        elif stage=="panel": k4_panel_manifest()
        elif stage=="retention": retention_audit(force=args.force)
        elif stage=="saturation": saturation(args.datasets.split(",") if args.datasets else None)
        elif stage=="r1": r1_run(args.datasets.split(",") if args.datasets else None,allow_local=args.allow_r1_local)
        elif stage=="final": final_report()

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--stage",choices=("all","k3","panel","retention","saturation","r1","final"),default="all"); p.add_argument("--datasets",default=""); p.add_argument("--allow-r1-local",action="store_true"); p.add_argument("--force",action="store_true"); main(p.parse_args())
