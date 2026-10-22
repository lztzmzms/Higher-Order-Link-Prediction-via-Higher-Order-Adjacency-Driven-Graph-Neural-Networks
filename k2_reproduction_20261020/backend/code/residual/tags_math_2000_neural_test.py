"""Causal, fixed-candidate pilot using repository DualMappingGNN and SparseSIN."""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import csv, hashlib, json, random, sys, time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
import networkx as nx
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import dual_mapping_from_raw as dual
import exp.run_all_sparse_sin as sin
from mp.models import SparseSIN
from simplicial_complex.data_loading import DataLoader
from simplicial_complex.simplex_utils import convert_graph_dataset_with_gudhi

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'input'
OUT = SOURCE / 'neural_pilot'
DS = 'tags-math-sx'
sys.path.insert(0, str(SOURCE / 'code'))
import easy_search as sample
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)


def atomic(name, obj):
    sample.s.atomic(OUT / name, obj)


def log(message):
    print(message, flush=True)
    atomic('STATUS.json', {'message':message, 'utc':sample.s.utc()})


def seed(value=41):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def metrics(y, scores):
    assert np.isfinite(scores).all() and len(y)==len(scores)
    ap = float(average_precision_score(y, scores))
    return {'AP':ap, 'Lift':ap/float(y.mean()), 'ROC_AUC':float(roc_auc_score(y,scores)),
            'count':len(y), 'positives':int(y.sum()), 'score_std':float(np.std(scores))}


def material():
    events = sample.s.load_events(ROOT/'data', DS)
    tasks = {}
    for name, fraction in [('dev_train',.4), ('final_train',.6)]:
        with np.load(SOURCE/DS/'cache'/f'{int(fraction*100)}.npz') as z:
            ids = sample.s.quota_sample(z['y'], 2000, 2026)
            tasks[name] = {'q':z['q'][ids], 'y':z['y'][ids], 'x':z['x'][ids],
                           'cut':int(z['cut']), 'label_end':int(z['label_end'])}
    with np.load(SOURCE/DS/'pools/60.npz') as z:
        ids, _ = sample.select_easy(z['q'],z['propensity'],z['margin'],z['learned'],
                                   1000,'easy_learned_70',2027,z['scores'].mean(axis=1))
        cut = int(z['cut']); end = sample.s.cutoff(events,.8)
        tasks['validation'] = {'q':z['q'][ids], 'y':sample.s.labels(events[cut:end],z['q'][ids]),
                               'cut':cut, 'label_end':end}
    with np.load(SOURCE/DS/'local_debug/easy_learned_70_2027/test_locked.npz') as z:
        tasks['test'] = {'q':z['q'], 'y':z['y'], 'x':z['x'], 'cut':int(z['prediction_cut']),
                         'label_end':len(events)}
    assert tasks['dev_train']['label_end'] <= tasks['validation']['cut']
    assert tasks['validation']['label_end'] <= tasks['test']['cut']
    assert tasks['final_train']['label_end'] <= tasks['test']['cut']
    assert len(tasks['test']['q'])==2000 and tasks['test']['y'].sum()==26
    for name,t in tasks.items():
        assert np.array_equal(sample.s.labels(events[t['cut']:t['label_end']],t['q']),t['y'])
        history = sample.s.History(events[:t['cut']])
        for q in map(tuple,t['q']):
            assert q not in history.closed and all(b in history.adj[a] for a,b in combinations(q,2))
        np.savez_compressed(OUT/f'{name}_manifest.npz',q=t['q'],y=t['y'],cut=t['cut'],label_end=t['label_end'])
    atomic('PROTOCOL.json', {k:{'count':len(t['q']), 'positives':int(t['y'].sum()),
        'prediction_cut':t['cut'],'label_end':t['label_end'],
        'candidate_sha256':sample.hashlib_array(t['q'])} for k,t in tasks.items()})
    return events,tasks


def dual_assets(events, tasks):
    for name,t in tasks.items():
        path=OUT/'cache'/f'dual_{name}.npz'
        if path.exists():
            with np.load(path) as z: t['dual']={k:z[k] for k in z.files}
            continue
        log(f'Dual-Mapping features: {name}')
        history=sample.s.History(events[:t['cut']])
        state={'edges':{e for a,ns in history.adj.items() for e in ((a,b) for b in ns if b>a)},
               'triangles':history.closed}
        q=list(map(tuple,t['q']))
        graph=dual.build_dual_graph(q)
        features,raw=dual.build_features(q,state,graph,return_raw=True)
        rw=dual.build_random_walk_structural_features(graph['edge_index'],len(q),rw_steps=2)
        t['dual']={'local':dual.select_feature_block(features,dual.FEATURE_NAMES,dual.local_feature_names()),
                   'heuristic':dual.build_heuristic_feature_block(raw), 'rw':rw,
                   'edge_index':graph['edge_index']}
        np.savez_compressed(path,**t['dual'])


def fit_dual(train, target, epochs, select):
    seed(); _,_,F,Model=dual.make_model_classes()
    tensors=[]
    for t in (train,target):
        tensors.append({k:torch.as_tensor(v,dtype=torch.long if k=='edge_index' else torch.float32,device=DEVICE)
                        for k,v in t['dual'].items()})
    for block in ('local','heuristic','rw'):
        scaler=StandardScaler().fit(train['dual'][block])
        for i,t in enumerate((train,target)):
            tensors[i][block]=torch.tensor(scaler.transform(t['dual'][block]),dtype=torch.float32,device=DEVICE)
    model=Model(tensors[0]['local'].shape[1],tensors[0]['heuristic'].shape[1],
                tensors[0]['rw'].shape[1],0,feature_proj_dim=16,hidden_dim=64,
                num_layers=2,dropout=.2,max_residual_alpha=1.).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    yy=torch.tensor(train['y'],dtype=torch.float32,device=DEVICE)
    weight=torch.tensor([(len(yy)-yy.sum())/yy.sum()],device=DEVICE)
    def predict(t):
        return model(t['local'],t['heuristic'],t['rw'],t['local'].new_empty((len(t['local']),0)),t['edge_index'])
    best=-1.; best_epoch=epochs; curve=[]
    start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(predict(tensors[0]),yy,pos_weight=weight)
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if select and epoch%5==0:
            model.eval()
            with torch.no_grad(): sc=torch.sigmoid(predict(tensors[1])).cpu().numpy()
            ap=metrics(target['y'],sc)['AP'];curve.append({'epoch':epoch,'val_AP':ap,'loss':float(loss)})
            log(f'Dual-Mapping epoch={epoch} validation AP={ap:.5f}')
            if ap>best:best=ap;best_epoch=epoch
    model.eval()
    with torch.no_grad():sc=torch.sigmoid(predict(tensors[1])).cpu().numpy()
    return sc,best_epoch,curve,time.perf_counter()-start,model.state_dict()


def sin_assets(events,tasks):
    for name,t in tasks.items():
        path=OUT/'cache'/f'sin_{name}.pt'
        if path.exists():
            t['complexes']=torch.load(path,map_location='cpu',weights_only=False);continue
        log(f'MLSHoL local complexes: {name}')
        history=sample.s.History(events[:t['cut']])
        graph=nx.Graph();graph.add_nodes_from(history.adj)
        graph.add_edges_from((a,b) for a,ns in history.adj.items() for b in ns if b>a)
        bynode=defaultdict(set)
        for _,_,vs in events[:t['cut']]:
            if len(vs)>=3:
                for v in vs:bynode[v].add(vs)
        pyg=[]
        positives=set(map(tuple,t['q'][t['y'].astype(bool)]))
        for i,tri in enumerate(map(tuple,t['q'])):
            random.seed(2026+i)
            local=sin.extract_k_hop_subgraph(graph,1,tri,max_nodes_per_step=30,enforce_node_cap=True).copy()
            nodes=set(local.nodes);old=set()
            for v in nodes:
                for sx in bynode[v]:
                    if sum(n in nodes for n in sx)>=3:old.add(sx)
            removed=sin.remove_target_link(local,tri)
            labeled=sin.construct_labeled_networkx_graph(removed,old,tri,positives,
                        disconnected_policy='unreachable_zero',observed_2_skeleton=True)
            pyg.append(sin.to_pyg_data(labeled))
            if i%500==0:log(f'MLSHoL {name} local graphs {i}/{len(t["q"])}')
        complexes=convert_graph_dataset_with_gudhi(pyg,expansion_dim=2,include_down_adj=False,
                                                 init_method='sum',expand_cliques=False)
        assert len(complexes)==len(t['q'])
        assert np.array_equal([int(c.y.item()) for c in complexes],t['y'])
        torch.save(complexes,path);t['complexes']=complexes


def fit_sin(train,target,epochs,select):
    seed();model=SparseSIN(num_input_features=1,num_classes=2,num_layers=3,hidden=16,
                           dropout_rate=.2,max_dim=2,readout='mean',final_readout='sum').to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.001)
    weights=sin.make_class_weights(train['complexes'],DEVICE)
    loader=DataLoader(train['complexes'],batch_size=64,shuffle=True,max_dim=2)
    eval_loader=DataLoader(target['complexes'],batch_size=64,shuffle=False,max_dim=2)
    def predict():
        model.eval();scores=[]
        with torch.no_grad():
            for data in eval_loader:
                scores.extend(torch.softmax(model(data.to(DEVICE),return_logits=True),dim=1)[:,1].cpu().numpy())
        return np.asarray(scores)
    best=-1.;best_epoch=epochs;curve=[];start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();loss_sum=0
        for data in loader:
            data=data.to(DEVICE);opt.zero_grad()
            loss=torch.nn.functional.cross_entropy(model(data,return_logits=True),data.y.view(-1).long(),weight=weights)
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();loss_sum+=float(loss)
        if select and epoch%5==0:
            ap=metrics(target['y'],predict())['AP'];curve.append({'epoch':epoch,'val_AP':ap,'loss':loss_sum/len(loader)})
            log(f'MLSHoL epoch={epoch} validation AP={ap:.5f}')
            if ap>best:best=ap;best_epoch=epoch
        elif epoch%5==0:log(f'MLSHoL final training epoch={epoch}/{epochs}')
    return predict(),best_epoch,curve,time.perf_counter()-start,model.state_dict()


def main():
    OUT.mkdir(exist_ok=True);(OUT/'cache').mkdir(exist_ok=True)
    started=time.perf_counter()
    atomic('PLAN.json',{'seed':41,'max_epochs':40,'validation_interval':5,'train_rows':2000,
        'validation_rows':1000,'test_rows':2000,'test_arm':'easy_learned_70_2027',
        'test_used_for_epoch_selection':False,'device':str(DEVICE),
        'dual':'repository DualMappingGNN, 2-layer residual GraphSAGE; local+heuristics+2-step RW stats; candidate-induced dual graph',
        'mlshol':'repository SparseSIN, 3 layers hidden16; TRNL, one hop max30 additional nodes, observed 2-skeleton',
        'full_frozen_configuration_reproduction':False,
        'source_sha256':{p:sample.s.digest(ROOT/p) for p in ['tags_math_2000_neural_test.py','dual_mapping_from_raw.py','mp/models.py','exp/run_all_sparse_sin.py']}})
    log(f'Preparing fixed temporal tasks; device={DEVICE}')
    events,tasks=material();rows=[]
    for name,build,fit in [('Dual-Mapping',dual_assets,fit_dual),('MLSHoL',sin_assets,fit_sin)]:
        marker=OUT/f'{name}_RESULT.json'
        if marker.exists():rows.append(sample.s.read(marker));continue
        start=time.perf_counter();build(events,tasks);preparation=time.perf_counter()-start
        _,best,curve,dev_seconds,_=fit(tasks['dev_train'],tasks['validation'],40,True)
        atomic(f'{name}_PARAMETER_LOCK.json',{'epochs':best,'validation_curve':curve,'final_labels_used':False})
        scores,_,_,final_seconds,state=fit(tasks['final_train'],tasks['test'],best,False)
        np.savez_compressed(OUT/f'{name}_predictions.npz',q=tasks['test']['q'],y=tasks['test']['y'],scores=scores)
        torch.save(state,OUT/f'{name}_checkpoint.pt')
        row={'model':name,**metrics(tasks['test']['y'],scores),'selected_epochs':best,
             'feature_preparation_seconds':preparation,'development_training_seconds':dev_seconds,
             'final_training_and_inference_seconds':final_seconds,
             'candidate_sha256':sample.hashlib_array(tasks['test']['q'])}
        atomic(f'{name}_RESULT.json',row);rows.append(row);log(str(row))
    with (OUT/'RESULTS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for row in rows:
        with np.load(OUT/f'{row["model"]}_predictions.npz') as z:
            assert np.array_equal(z['q'],tasks['test']['q']) and np.array_equal(z['y'],tasks['test']['y'])
            assert metrics(z['y'],z['scores'])=={k:row[k] for k in metrics(z['y'],z['scores'])}
    atomic('VERIFICATION.json',{'state':'PASSED','identical_2000_candidates_and_26_labels':True,
        'raw_event_labels_and_candidate_legality_checked':True,'metrics_recomputed':True,
        'temporal_training_gates_checked':True,'total_seconds':time.perf_counter()-started})
    lines=['# tags-math-sx 固定 2000 候选神经模型测试','',
        '两个模型完全使用 easy_learned_70_2027 的 2000 个测试候选、26 个正例。',
        '先用 40%→60% 的 2000 行训练，在 60%→80% 的 1000 行验证中选轮数；再从头用 60%→80% 的 2000 行训练，预测 80%→100%。',
        '训练样本按过去已知类别比例抽取，测试样本与标签保持不变。种子 41，最多 40 轮，每 5 轮按历史验证 AP 选训练轮数。','',
        '| 模型 | AP | Lift | ROC-AUC | 选定轮数 | 特征准备秒 | 开发训练秒 | 最终训练及推理秒 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["model"]} | {r["AP"]:.6f} | {r["Lift"]:.4f} | {r["ROC_AUC"]:.4f} | {r["selected_epochs"]} | {r["feature_preparation_seconds"]:.1f} | {r["development_training_seconds"]:.1f} | {r["final_training_and_inference_seconds"]:.1f} |')
    lines+=['','HG 上轮参考 AP=0.189557、Lift=14.5813，使用最多 25000 行历史训练，与本轮 2000 行训练不构成等训练预算比较。',
        '本轮实际训练了仓库 DualMappingGNN 和 SparseSIN；采用较小配置，未复现旧冻结完整模型。',
        'Dual-Mapping：2 层残差 GraphSAGE，局部结构/启发式/2 步随机游走结构统计。派生图仅包含当前抽取候选，未运行全候选图及学习式随机游走嵌入。',
        'MLSHoL：3 层 hidden16 的 SparseSIN，TRNL、1 跳局部图，每跳最多 30 个新增节点；历史观测 2-skeleton，无额外 RW/活动融合。',
        'GPU 加速；保存预测、检查点、时间门禁及参数锁定记录。最终标签在过往实验已查看，结果为探索性测试。']
    (OUT/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    log('COMPLETE: both neural models verified')


if __name__=='__main__':main()
