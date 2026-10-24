                                                                                   
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.%s.tmp' % os.getpid())
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8'); os.replace(tmp, path)

def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def worker(job, seed, smoke=False):
    import torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score, roc_auc_score
    sys.path.insert(0, str(ROOT/'code/residual'))
    from scatter_compat_20261015 import ensure_scatter
    scatter_backend=ensure_scatter()
    print('scatter backend:',scatter_backend['backend'],flush=True)
    import tags_math_2000_neural_test as p
    e=job['effective']; ds=e['dataset']; model=e['model']
    d=ROOT/'results'/('smoke' if smoke else 'jobs')/job['id']/str(seed)
    d.mkdir(parents=True, exist_ok=True)
    if (d/('DONE.json' if smoke else 'RESULT.json')).exists(): return
    cfg=load(ROOT/job['source_config']); assert sha(ROOT/job['source_config'])==e['source_config_sha256']
    material=ROOT/'material'/ds
                                                                                          
    z=dict(np.load(material/'EVENTS.npz')); offsets=np.r_[0,np.cumsum(z['counts'])]
    events=[(t.item(),int(ix),tuple(map(int,z['vertices'][offsets[i]:offsets[i+1]]))) for i,(t,ix) in enumerate(zip(z['times'],z['indices']))]
    tasks={name:dict(np.load(material/(name+'.npz'))) for name in ('train','test')}
    actual_y=np.load(material/'EVALUATION.npz')['y']
                                                                     
    assert 'y' not in tasks['test']
    for t in tasks.values():
        t['cut']=int(t['cut']);t['label_end']=int(t['label_end'])
    assert tasks['train']['label_end']<=tasks['test']['cut']
    tasks['test']['y']=np.zeros(len(tasks['test']['q']),dtype=np.int64)
    if smoke:
        train=tasks['train']; ids=np.r_[np.flatnonzero(train['y'])[:2],np.flatnonzero(train['y']==0)[:2]]
        for name,ix in [('train',ids),('test',np.arange(4))]:
            tasks[name]={k:(v[ix] if k in ('q','y','x','base','base_probability') else v) for k,v in tasks[name].items()}
    train,target=tasks['train'],tasks['test']
    p.DS=ds;p.OUT=ROOT/'artifacts/residual'/ds/('smoke' if smoke else 'full')
    if model=='MLSHoL':p.OUT=p.OUT/('dim%s_%s%s' % (e['max_dim'],e['representation'],'_strict_v2' if e['max_dim']<2 else ''))
    (p.OUT/'cache').mkdir(parents=True,exist_ok=True)
    started=time.time()
    if ds=='tags-math-sx':
                                                                               
        cache=ROOT/'artifacts/teacher'/ds/('smoke' if smoke else 'full')
        cache.mkdir(parents=True,exist_ok=True)
        basefile=cache/'BASE.npz'
        if not basefile.exists():
            import tags_math_2000_neural_improve as m
            m.OUT=cache;m.DEVICE=p.DEVICE
            snapshots=[]
            for path in sorted((material/'history').glob('*.npz')):
                h=np.load(path);snapshots.append({'x':h['x'],'y':h['y'],'cut':int(h['cut']),'label_end':int(h['label_end']),'names':h['names'].tolist()})
            hcfg=copy.deepcopy(cfg['history'])
            if smoke:hcfg['epochs']=1
            m.history_ensemble(snapshots,target['cut'],[target],hcfg,'test')
            m.history_ensemble(snapshots,train['cut'],[train],hcfg,'training_covariate')
            tmp=cache/('BASE.%s.npz' % os.getpid());np.savez_compressed(tmp,train_q=train['q'],test_q=target['q'],train_base=train['base'],test_base=target['base']);os.replace(tmp,basefile)
        else:
            h=np.load(basefile);np.testing.assert_array_equal(h['train_q'],train['q']);np.testing.assert_array_equal(h['test_q'],target['q']);train['base']=h['train_base'];target['base']=h['test_base']
    epochs=1 if smoke else cfg['epochs']; scale=cfg['scale']; bounded=cfg['bounded_tanh']
    save(d/'CONFIG.json',{'locked_source':cfg,'effective':e,'seed':seed,'epochs_executed':epochs,'smoke':smoke,'source_sha256':e['source_config_sha256'],'scatter_backend':scatter_backend,'teacher_frozen':True,'training_residual_strength':1.,'inference_strength':scale})
    p.seed(seed);torch.set_num_threads(1)
    from frequency_hooks import residual_assets
    residual_assets(p,events,tasks,e['frequency'],model,d)
    if cfg.get('deterministic',False):torch.use_deterministic_algorithms(True)
    device=p.DEVICE; y=torch.tensor(train['y'],dtype=torch.float32,device=device)
    weight=torch.tensor([(len(y)-y.sum())/y.sum()],dtype=torch.float32,device=device)
    perturb=(lambda r:torch.tanh(r)) if bounded else (lambda r:r)
    if model=='Dual-Mapping':
        p.dual_assets(events,tasks);_,_,_,Class=p.dual.make_model_classes(); tensors=[]
        for t in (train,target):tensors.append({k:torch.as_tensor(v,dtype=torch.long if k=='edge_index' else torch.float32,device=device) for k,v in t['dual'].items()})
        for block in ('local','heuristic','rw'):
            scaler=StandardScaler().fit(train['dual'][block])
            for t,src in zip(tensors,(train,target)):t[block]=torch.as_tensor(scaler.transform(src['dual'][block]),dtype=torch.float32,device=device)
        if not e['rw']:
            for t in tensors:t['rw']=torch.zeros_like(t['rw'])
        net=Class(tensors[0]['local'].shape[1],tensors[0]['heuristic'].shape[1],tensors[0]['rw'].shape[1],0,feature_proj_dim=16,hidden_dim=64,num_layers=e['layers'],dropout=.2,max_residual_alpha=1.).to(device)
        for head in (net.base_head[-1],net.residual_head[-1]):torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
        def forward(t):return net(t['local'],t['heuristic'],t['rw'],t['local'].new_empty((len(t['local']),0)),t['edge_index'],t.get('edge_weight'))
        base=torch.tensor(train['base'],dtype=torch.float32,device=device)
        def predict():
            net.eval()
            with torch.no_grad():return perturb(forward(tensors[1])).cpu().numpy()
        def epoch(opt):
            net.train();opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(base+perturb(forward(tensors[0])),y,pos_weight=weight);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();return float(loss)
    else:
        maxdim=e['max_dim'];rep=e['representation']
        if rep not in ('native','MRNL'):
            import representation_geometry as geometry
            geometry.build(p,events,tasks,rep,maxdim,ROOT/'protocol'/ds,ROOT/'artifacts/embeddings'/ds/rep)
        elif maxdim==2:
            p.sin_assets(events,tasks)
        else:
            import representation_geometry as geometry
            geometry.build(p,events,tasks,'MRNL',maxdim,ROOT/'protocol'/ds,None)
                                                                    
        if ds=='tags-math-sx':p.seed(seed)
        core=p.SparseSIN(num_input_features=1,num_classes=2,num_layers=3,hidden=16,dropout_rate=.2,max_dim=maxdim,readout='mean',final_readout='sum').to(device)
        torch.nn.init.zeros_(core.lin2.weight);torch.nn.init.zeros_(core.lin2.bias)
        rw_values=None
        if e['rw']:
                                                                                   
            import representation_geometry as geometry
            rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            rw_values=geometry.candidate_rw(p,tasks,ROOT/'artifacts/residual_rw'/ds/('smoke' if smoke else 'full'))
            torch.set_rng_state(rng)
            if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
            class WithRW(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.core=core;self.head=torch.nn.Sequential(torch.nn.Linear(64,16),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(16,1))
                    torch.nn.init.zeros_(self.head[-1].weight);torch.nn.init.zeros_(self.head[-1].bias)
                def forward(self,batch):
                    logits=self.core(batch,return_logits=True);r=self.head(batch.cochains[0].candidate_rw).flatten()
                    return logits+torch.stack((torch.zeros_like(r),r),dim=1)
            net=WithRW().to(device)
        else:net=core
        for name,t in tasks.items():
            for i,(c,base) in enumerate(zip(t['complexes'],t['base'])):
                c.cochains[0].history_base_logit=torch.tensor([[base]],dtype=torch.float32)
                if rw_values is not None:c.cochains[0].candidate_rw=torch.tensor(rw_values[name][i:i+1],dtype=torch.float32)
        loader=p.DataLoader(train['complexes'],batch_size=64,shuffle=True,max_dim=maxdim);evaluation=p.DataLoader(target['complexes'],batch_size=64,shuffle=False,max_dim=maxdim)
        def predict():
            net.eval();values=[]
            with torch.no_grad():
                for batch in evaluation:
                    batch=batch.to(device);logits=net(batch) if e['rw'] else net(batch,return_logits=True);values.extend(perturb(logits[:,1]-logits[:,0]).cpu().numpy())
            return np.array(values)
        def epoch(opt):
            net.train();values=[]
            for batch in loader:
                batch=batch.to(device);opt.zero_grad();logits=net(batch) if e['rw'] else net(batch,return_logits=True);scores=batch.cochains[0].history_base_logit.flatten()+perturb(logits[:,1]-logits[:,0]);loss=torch.nn.functional.binary_cross_entropy_with_logits(scores,batch.y.view(-1).float(),pos_weight=weight);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();values.append(float(loss))
            return float(np.mean(values))
                                                                                
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
                                                                                  
                                                                                
    if ds=='tags-math-sx':
                                                                                   
        if model=='MLSHoL':assert torch.count_nonzero(core.lin2.weight)==0 and torch.count_nonzero(core.lin2.bias)==0
        else:
            for head in (net.base_head[-1],net.residual_head[-1]):assert torch.count_nonzero(head.weight)==0 and torch.count_nonzero(head.bias)==0
    else:
        initial=predict();assert np.max(np.abs(initial))<1e-6
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);gradient=False
    for ep in range(1,epochs+1):
        loss=epoch(opt);gradient |= any(v.grad is not None and float(v.grad.abs().sum())>0 for v in net.parameters())
        save(d/'PROGRESS.json',{'epoch':ep,'epochs':epochs,'loss':loss,'time':time.time()});print('epoch',ep,'loss',loss,flush=True)
    assert gradient
    torch.save(net.state_dict(),d/'CHECKPOINT.pt');residual=predict()
    score=1/(1+np.exp(-np.clip(target['base']+scale*residual,-40,40)))
    np.savez_compressed(d/'FORECAST.npz',triplets=target['q'],final=score,history_logit=target['base'],residual=residual)
    save(d/'FORECAST_LOCK.json',{'sha256':sha(d/'FORECAST.npz'),'test_labels_used_for_selection':False,'config_sha256':sha(d/'CONFIG.json')})
    resources={'runtime_sec':time.time()-started,'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None,'nonzero_training_gradient':gradient}
    save(d/'RESOURCE.json',resources)
    if smoke:save(d/'DONE.json',{'state':'SMOKE_PASSED',**resources});return
    np.savez_compressed(d/'PREDICTIONS.npz',triplets=target['q'],y_true=actual_y,native=residual,final=score,history_logit=target['base'],residual=residual)
    ap=float(average_precision_score(actual_y,score)); nap=float(average_precision_score(actual_y,residual))
    result={'id':job['id'],'dataset':ds,'model':model,'seed':seed,'AP':ap,'Lift':ap/actual_y.mean(),'native_AP':nap,'native_Lift':nap/actual_y.mean(),'ROC_AUC':float(roc_auc_score(actual_y,score)),'candidates':len(actual_y),'positives':int(actual_y.sum()),**resources,'prediction_sha256':sha(d/'PREDICTIONS.npz')}
    save(d/'RESULT.json',result)
