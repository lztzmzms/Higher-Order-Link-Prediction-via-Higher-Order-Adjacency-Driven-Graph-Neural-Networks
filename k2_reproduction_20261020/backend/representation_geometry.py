                                                                                 
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import networkx as nx
import torch

def candidate_rw(p,tasks,folder):
    pass                                                                            
    from sklearn.preprocessing import StandardScaler
    import ablation_embeddings as emb
    folder.mkdir(parents=True,exist_ok=True);path=folder/'RW.npz'
    if path.exists():
        z=np.load(path)
        for name,t in tasks.items():np.testing.assert_array_equal(t['q'],z[name+'_q'])
        return {name:z[name] for name in tasks}
    olddim=emb.DIM;emb.DIM=64;values={};states={};train_lookup={}
    try:
        for name,t in tasks.items():
            q=[tuple(map(int,tri)) for tri in t['q']];graph=p.dual.build_dual_graph(q);g=nx.Graph();g.add_nodes_from(range(len(q)));g.add_edges_from(np.asarray(graph['edge_index']).T.tolist())
            pairs=emb._pairs(emb._walks(g,list(range(len(q))),'DeepWalk',0),{i:i for i in range(len(q))})
            initial=None
            if name=='test':
                rng=np.random.default_rng(0);initial=(rng.normal(0,.02,(len(q),64)).astype(np.float32),np.zeros((len(q),64),np.float32))
                for i,tri in enumerate(q):
                    if tri in train_lookup:
                        ix=train_lookup[tri];initial[0][i]=states['train'][0][ix];initial[1][i]=states['train'][1][ix]
            states[name]=emb._sgns(pairs,len(q),0,initial);values[name]=states[name][0]
            if name=='train':train_lookup={tri:i for i,tri in enumerate(q)}
        scaler=StandardScaler().fit(values['train']);values={name:scaler.transform(v).astype(np.float32) for name,v in values.items()}
        tmp=folder/'RW.partial.npz';np.savez_compressed(tmp,**values,**{name+'_q':t['q'] for name,t in tasks.items()});tmp.replace(path)
    finally:emb.DIM=olddim
    return values

def build(p, events, tasks, method, maxdim, protocol, embfolder):
    maps={}
    if method!='MRNL':
        import ablation_embeddings as emb
        if not (embfolder/'audit.json').exists():
                                                                                     
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):emb.build_pair(p.DS,method,protocol,embfolder)
        for name in ('train','test'):
            z=np.load(embfolder/(name+'.npz'));assert z['vectors'].shape[1]==1;maps[name]=dict(zip(z['nodes'].tolist(),z['vectors']))
    for name,t in tasks.items():
        path=p.OUT/'cache'/('sin_%s.pt' % name)
        if path.exists():t['complexes']=torch.load(path,map_location='cpu');continue
        hist=p.sample.s.History(events[:t['cut']]);g=nx.Graph();g.add_nodes_from(hist.adj)
        g.add_edges_from((a,b) for a,ns in hist.adj.items() for b in ns if b>a)
        bynode=defaultdict(set)
        for _,_,vs in events[:t['cut']]:
            if len(vs)>=3:
                for v in vs:bynode[v].add(vs)
        pyg=[];positives=set(map(tuple,t['q'][t['y'].astype(bool)]))
        for i,tri in enumerate(map(tuple,t['q'])):
            random.seed(2026+i);local=p.sin.extract_k_hop_subgraph(g,1,tri,max_nodes_per_step=30,enforce_node_cap=True).copy()
            nodes=set(local.nodes);old={s for v in nodes for s in bynode[v] if sum(n in nodes for n in s)>=3}
            removed=p.sin.remove_target_link(local,tri)
            labeled=p.sin.construct_labeled_networkx_graph(removed,old,tri,positives,disconnected_policy='unreachable_zero',observed_2_skeleton=True)
            if method!='MRNL':
                for node in labeled.nodes:labeled.nodes[node]['feature']=maps[name][node].tolist()
            data=p.sin.to_pyg_data(labeled)
                                                                                               
            data.simplices=[s for s in data.simplices if len(s)<=maxdim+1]
            if maxdim==0:data.simplices=[[i] for i in range(data.num_nodes)]
            pyg.append(data)
        complexes=p.convert_graph_dataset_with_gudhi(pyg,expansion_dim=maxdim,include_down_adj=False,init_method='sum',expand_cliques=False)
        assert len(complexes)==len(t['q']);torch.save(complexes,path);t['complexes']=complexes
        assert all(c.dimension<=maxdim for c in complexes)
