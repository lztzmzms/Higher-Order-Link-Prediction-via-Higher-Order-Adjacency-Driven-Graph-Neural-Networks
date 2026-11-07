from itertools import combinations
from pathlib import Path
import tempfile
import json
import time
import numpy as np
import networkx as nx
import torch
from sklearn.preprocessing import StandardScaler
from data.prepare import ROOT, config as dataset_config
from historical_components.interfaces import ComponentBatch, ComponentGenerator
from data.complexes import PackedComplexes
from models import dual, network
from train import roles_for, fit_model
from evaluate import evaluate
from pipelines.inputs import unpack, native_check
from historical_components.generators.engine import definitions
from historical_components.upstream.tag_sampling import capsnap
from historical_components.upstream import tag_predictor
from historical_components.encoders import ComponentPreprocessor


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def triples(path):
    return np.asarray([tuple(sorted(map(int,line.split()))) for line in Path(path).read_text().splitlines() if line.strip()],dtype=np.int64).reshape(-1,3)

def prepare(ds, dest):
    folder = unpack(ds)
    dest.mkdir(parents=True,exist_ok=True)
    proto = folder/'protocol'
    with np.load(proto/'shared_splits'/ds/'fit_val_indices.npz') as a:
        fit, val, y = a['fit_indices'].copy(), a['val_indices'].copy(), a['train_labels'].copy()
    order = np.r_[fit,val]
    for split,role in [('train','fit'),('test','target')]:
        start = time.monotonic()
        q = triples(proto/'dataset'/split/f'{ds}_openTriangles_{split}.txt')
        if role == 'fit':
            q = q[order]
        graph = nx.read_graphml(proto/'dataset'/split/f'{ds}_G_{split}.graphml')
        edges = {tuple(sorted((int(a),int(b)))) for a,b in graph.edges()}
        closed = set(map(tuple,triples(proto/'dataset'/split/f'{ds}_closeTriangles_{split}.txt')))
        dg = dual.build_dual_graph(list(map(tuple,q)))
        features, raw = dual.build_features(list(map(tuple,q)),dict(edges=edges,triangles=closed),dg,return_raw=True)
        rw = dual.build_random_walk_structural_features(dg['edge_index'],len(q),rw_steps=2)
        np.savez_compressed(dest/f'{role}.npz',q=q,
            local=dual.select_feature_block(features,dual.FEATURE_NAMES,dual.local_feature_names()),
            heuristic=dual.build_heuristic_feature_block(raw),rw=rw,edge_index=dg['edge_index'])
        print(json.dumps(dict(stage='structure_prepared',dataset=ds,role=role,N=len(q),edges=dg['edge_index'].shape[1],seconds=time.monotonic()-start)),flush=True)
    np.savez_compressed(dest/'training.npz',order=order,fit_indices=fit,val_indices=val,fit_y=y[fit])

class ReorderedComplexes:
    def __init__(self, packed, order):
        self.packed, self.order = packed, order
    def __getitem__(self, index):
        return self.packed[int(self.order[int(index)])]
    def __len__(self):
        return len(self.order)

def load_roles(ds,model,device,dest,components=None):
    blocks={}
    for role in ('fit','target'):
        with np.load(dest/f'{role}.npz') as a:
            blocks[role]={k:a[k].copy() for k in a.files}
    with np.load(dest/'training.npz') as a:
        order,y=a['order'].copy(),a['fit_y'].copy()
    q={r:blocks[r].pop('q') for r in blocks}
    rw={}
    for key in ('local','heuristic','rw'):
        scaler=StandardScaler().fit(blocks['fit'][key][:len(y)])
        for role in blocks:
            blocks[role][key]=torch.tensor(scaler.transform(blocks[role][key]),device=device,dtype=torch.float32)
            if key=='rw':rw[role]=blocks[role][key]
    for role in blocks:
        blocks[role]['edge_index']=torch.tensor(blocks[role]['edge_index'],device=device,dtype=torch.long)
    if model=='Dual-Mapping':
        structure=blocks
    else:
        packed={r:PackedComplexes(ROOT/'data'/ds/f'complexes_{r}.npz') for r in q}
        np.testing.assert_array_equal(packed['fit'].arrays['q'][order],q['fit'])
        np.testing.assert_array_equal(packed['target'].arrays['q'],q['target'])
        structure={'fit':ReorderedComplexes(packed['fit'],order),'target':packed['target']}
    return roles_for(model,structure,rw,q,components),y

def generate(ds, prepared, folder, cfg):
    folder.mkdir(parents=True,exist_ok=False)
    source=unpack(ds)/'protocol'
    raw=source/'dataset/raw_data'/ds
    times=np.loadtxt(raw/(ds+'-times.txt'),dtype=np.int64)
    counts=np.loadtxt(raw/(ds+'-nverts.txt'),dtype=np.int64)
    vertices=np.loadtxt(raw/(ds+'-simplices.txt'),dtype=np.int64)
    assert len(times)==len(counts) and np.sum(counts)==len(vertices)
    assert np.all(times[1:]>=times[:-1])
    offsets=np.r_[0,np.cumsum(counts)]
    ratios=json.loads((source/'READY.json').read_text())
    total=len(times)
    native_cuts={.6:int(total*ratios['train_ratio']),.8:int(total*ratios['test_ratio'])}
    def cut(f):
        f=round(f,2)
        if f in native_cuts:return native_cuts[f]
        n=int(total*f)
        while n and times[n-1]==times[n]:n-=1
        return n
    for boundary in native_cuts.values():
        if boundary and times[boundary-1]>=times[boundary]:
            raise ValueError('Native origin splits a timestamp block')
    maximum=native_cuts[.8]
    events=[(int(times[i]),i,tuple(sorted(set(map(int,vertices[offsets[i]:offsets[i+1]]))))) for i in range(maximum)]
    event=definitions('tag_features',['History'])
    sampling=definitions('temporal',['uniform_open_candidates'],dict(CAP=25000))
    temporal=definitions('temporal',['attach_labels'])
    windows=[]
    for f in (.1,.2,.3,.4,.5,.6):
        c,e=cut(f),cut(round(f+.2,2))
        start=time.monotonic()
        q,_,_=sampling.uniform_open_candidates(events[:c],25000)
        q=np.asarray(q,dtype=np.int64).reshape(-1,3)
        if not len(q):raise ValueError(f'Empty past candidate cohort at {f}')
        x,names=event.History(events[:c]).features(q)
        y=temporal.attach_labels(events[c:e],list(map(tuple,q)))
        record=dict(q=q,x=x,y=y,cut=c,label_end=e,names=names)
        windows.append(record)
        np.savez_compressed(folder/f'history_{round(f*100)}.npz',q=q,x=x,y=y,cut=c,label_end=e,names=np.asarray(names))
        print(json.dumps(dict(dataset=ds,stage='historical_snapshot',fraction=f,rows=len(q),positives=int(y.sum()),cut=c,label_end=e,seconds=time.monotonic()-start)),flush=True)
    gates={}
    components={}
    for role,f in [('fit',.6),('target',.8)]:
        c=native_cuts[f]
        with np.load(prepared/f'{role}.npz') as a:q=a['q'].copy()
        state=event.History(events[:c])
        for tri in map(tuple,q):
            if tri in state.closed or not all(b in state.adj[a] for a,b in combinations(tri,2)):
                raise ValueError('Native candidates do not match the raw temporal prefix')
        x,names=state.features(q)
        tx,y,gate=capsnap(windows,c,25000)
        if len(np.unique(y))!=2:raise ValueError('Original HistoryNet sampling has fewer than two past label classes')
        values=[]
        for seed in cfg['ensemble_seeds']:
            _,_,prediction,_=tag_predictor.history_fit(tx,y,x,cfg,seed,cfg['epochs'])
            values.append(prediction)
        components[role]=np.column_stack(values)
        gates[role]=gate
        np.savez_compressed(folder/f'{role}_raw.npz',q=q,component_x=components[role],names=np.asarray(['HistoryNet_logit_seed_41','HistoryNet_logit_seed_42']))
        print(json.dumps(dict(dataset=ds,stage='components_generated',role=role,rows=len(q),**gate)),flush=True)
    with np.load(prepared/'training.npz') as a:nfit=len(a['fit_y'])
    scaler=ComponentPreprocessor(['HistoryNet_logit_seed_41','HistoryNet_logit_seed_42']).fit(components['fit'],ids=np.arange(nfit))
    np.savez_compressed(folder/'scaled.npz',**{r:scaler.transform(x) for r,x in components.items()})
    write(folder/'preprocessing.json',scaler.state_dict())
    write(folder/'recipe.json',dict(recipe='tags-math-sx HistoryNet explicit transfer',parameters=cfg,
        historical_origins=[.1,.2,.3,.4,.5,.6],label_width_fraction=.2,
        historical_candidates='Original uniform reservoir open-triangle routine, cap=25000, seed=62026',
        completed_window_selection='Original last3 capsnap, proportional quota sampling cap=25000',
        max_label_end_used=maximum,native_cuts={str(k):v for k,v in native_cuts.items()},parameter_gates=gates,
        final_test_labels_used=False,joint_scaler_fits_original_fit_indices_only=True,
        status='new explicitly selected transfer recipe, not original NDC/threads H reconstruction'))
    return {r:scaler.transform(x) for r,x in components.items()}


class HistoryNetTransferGenerator(ComponentGenerator):
    def __init__(self, dataset, config=None):
        if dataset not in ('NDC-substances', 'threads-math-sx'):
            raise ValueError('HistoryNet transfer requires a configured NDC or threads cohort')
        self.dataset = dataset
        self.config = dict(dataset_config(dataset)['component_generator_config'])
        self.config.update(config or {})

    def generate_prepared(self, prepared, output_dir):
        torch.set_num_threads(1)
        prepared, output_dir = Path(prepared), Path(output_dir)
        generate(self.dataset, prepared, output_dir, self.config)
        recipe = json.loads((output_dir/'recipe.json').read_text(encoding='utf-8'))
        batches = {}
        for role in ('fit', 'target'):
            with np.load(output_dir/f'{role}_raw.npz', allow_pickle=False) as raw:
                with np.load(prepared/f'{role}.npz', allow_pickle=False) as cohort:
                    np.testing.assert_array_equal(raw['q'], cohort['q'])
                batches[role] = ComponentBatch(q=raw['q'].copy(), values=raw['component_x'].copy(),
                    names=raw['names'].tolist(), mask_indices=(),
                    metadata=dict(recipe=recipe['recipe'], role=role,
                                  **recipe['parameter_gates'][role]))
        return batches

    def generate(self, output_dir=None, context=None):
        if context is not None:
            raise ValueError('HistoryNet transfer is bound to the configured NDC/threads cohorts')
        if output_dir is None:
            with tempfile.TemporaryDirectory() as directory:
                return self.generate(output_dir=Path(directory)/'components')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        prepared = output_dir/'prepared'
        prepare(self.dataset, prepared)
        return self.generate_prepared(prepared, output_dir/'components')


def run(args, cfg):
    mode = args.mode or 'component_joint'
    if mode not in ('component_joint', 'structure_only'):
        raise ValueError('Unsupported prediction mode')
    torch.set_num_threads(1)
    network.DEVICE = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'config.json', dict(cfg, requested_mode=mode))
    prepared = args.output/'prepared'
    prepare(args.dataset, prepared)
    components = None
    if mode == 'component_joint':
        from historical_components.generators import create_generator
        generator = create_generator(cfg['component_generator'], cfg['component_generator_config'])
        generator.generate_prepared(prepared, args.output/'components')
        with np.load(args.output/'components'/'scaled.npz', allow_pickle=False) as values:
            components = {role: values[role].copy() for role in ('fit', 'target')}
    report = []
    for model in args.models:
        roles, labels = load_roles(args.dataset, model, network.DEVICE, prepared, components)
        for seed in ([args.seed] if args.seed is not None else cfg['models'][model]['seeds']):
            net, score = fit_model(model, roles, labels, cfg['models'][model], seed, mode, args.device)
            with np.load(unpack(args.dataset)/'selected_sample.npz', allow_pickle=False) as selected:
                ids, q, y = selected['base_indices'], selected['q'], selected['y']
                np.testing.assert_array_equal(roles['target']['q'][ids], q)
            native_check(args.dataset)
            np.savez_compressed(args.output/f'{model}_{seed}_all.npz', q=roles['target']['q'], logits=score)
            score = score[ids]
            np.savez_compressed(args.output/f'{model}_{seed}.npz', q=q, y=y, logits=score,
                                probabilities=torch.sigmoid(torch.from_numpy(score)).numpy(), predictions=score)
            torch.save(net.state_dict(), args.output/f'{model}_{seed}.pt')
            write(args.output/f'{model}_{seed}_training.json', net.training_history)
            row = dict(dataset=args.dataset, model=model, seed=seed, mode=mode, **evaluate(y, score))
            report.append(row)
            print(json.dumps(row), flush=True)
    write(args.output/'metrics.json', report)
