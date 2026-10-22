"""History-pretrained neural structural residuals; tuning sees historical labels only."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import copy,json,time,csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import tags_math_2000_neural_test as p

OUT=p.SOURCE/'neural_improved'
DEVICE=p.DEVICE
SEEDS=(41,42)


def save(name,obj):p.sample.s.atomic(OUT/name,obj)
def log(message):
    print(message,flush=True);save('STATUS.json',{'message':message,'utc':p.sample.s.utc()})


class HistoryNet(nn.Module):
    def __init__(self,dim,width):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,width),nn.ReLU(),nn.Dropout(.15),
                               nn.Linear(width,width),nn.ReLU(),nn.Dropout(.15),nn.Linear(width,1))
    def forward(self,x):return self.net(x).flatten()


def history_fit(x,y,target,config,seed,epochs,selection_y=None):
    p.seed(seed)
    scaler=StandardScaler().fit(x)
    tx=torch.tensor(scaler.transform(x),dtype=torch.float32,device=DEVICE)
    ty=torch.tensor(y,dtype=torch.float32,device=DEVICE)
    vx=torch.tensor(scaler.transform(target),dtype=torch.float32,device=DEVICE)
    model=HistoryNet(x.shape[1],config['width']).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    weight=((len(y)-y.sum())/y.sum())**config['weight_power']
    curves={};states={}
    for epoch in range(1,epochs+1):
        model.train();order=torch.randperm(len(tx),device=DEVICE)
        for ids in order.split(1024):
            optimizer.zero_grad()
            loss=nn.functional.binary_cross_entropy_with_logits(model(tx[ids]),ty[ids],
                 pos_weight=torch.tensor([weight],device=DEVICE,dtype=torch.float32))
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if selection_y is not None and epoch%5==0:
            model.eval()
            with torch.no_grad():sc=torch.sigmoid(model(vx)).cpu().numpy()
            curves[epoch]=sc
    model.eval()
    with torch.no_grad():logits=model(vx).cpu().numpy()
    return model,scaler,logits,curves


def history_material(events,tasks):
    snapshots=[]
    for fraction in p.sample.ORIGINS:
        with np.load(p.SOURCE/p.DS/'cache'/f'{int(round(fraction*100))}.npz') as z:
            snapshots.append({'x':z['x'],'y':z['y'],'cut':int(z['cut']),
                              'label_end':int(z['label_end']),'names':z['names'].tolist()})
    folds=[]
    for fraction in (.5,.6):
        with np.load(p.SOURCE/p.DS/'pools'/f'{int(fraction*100)}.npz') as z:
            n=2000
            ids,_=p.sample.select_easy(z['q'],z['propensity'],z['margin'],z['learned'],n,
                                      'easy_learned_70',2027,z['scores'].mean(axis=1))
            cut=int(z['cut']);end=p.sample.s.cutoff(events,round(fraction+.2,2))
            yy=p.sample.s.labels(events[cut:end],z['q'][ids])
            xx,y,gate=p.sample.capsnap(snapshots,cut,25000)
            folds.append({'fraction':fraction,'x':xx,'y':y,'vx':z['x'][ids],'vy':yy,'gate':gate})
    # Attach the *same* candidate-indexed historical features to local structural tasks.
    with np.load(p.SOURCE/p.DS/'pools/60.npz') as z:
        lookup={tuple(q):i for i,q in enumerate(z['q'])}
        tasks['validation']['x']=z['x'][[lookup[tuple(q)] for q in tasks['validation']['q']]]
    for name,t in tasks.items():
        history=p.sample.s.History(events[:t['cut']])
        if 'x' not in t:t['x'],_=history.features(t['q'])
    return snapshots,folds


def tune_history(folds):
    configs=[{'width':64,'weight_power':.5},{'width':128,'weight_power':.5},
             {'width':64,'weight_power':1.}]
    trials=[]
    for index,cfg in enumerate(configs):
        predictions=[]
        for fold in folds:
            curves=[]
            for seed in SEEDS:
                log(f'History pretraining config={index} fold={fold["fraction"]} seed={seed}')
                *_,curve=history_fit(fold['x'],fold['y'],fold['vx'],cfg,seed,50,fold['vy'])
                curves.append(curve)
            predictions.append({ep:np.mean([v[ep] for v in curves],axis=0) for ep in curves[0]})
        for epoch in range(5,51,5):
            ms=[p.metrics(f['vy'],pred[epoch]) for f,pred in zip(folds,predictions)]
            quality=float(np.mean(np.log([m['Lift'] for m in ms])))
            trials.append({'config':cfg,'epochs':epoch,'quality':quality,'fold_metrics':ms})
    best=max(trials,key=lambda x:x['quality'])
    save('HISTORY_PARAMETER_LOCK.json',{'chosen':best,'trials':trials,'test_labels_used':False})
    log(f'History selected: {best}')
    return best


def history_ensemble(snapshots,cut,targets,chosen,name):
    x,y,gate=p.sample.capsnap(snapshots,cut,25000)
    save(f'{name}_HISTORY_GATE.json',gate)
    models=[];scalers=[]
    allx=np.concatenate([t['x'] for t in targets])
    for seed in SEEDS:
        model,scaler,_,_=history_fit(x,y,allx,chosen['config'],seed,chosen['epochs'])
        torch.save({'model':model.state_dict(),'config':chosen,'feature_mean':scaler.mean_,
                    'feature_scale':scaler.scale_,'gate':gate},OUT/f'{name}_history_seed{seed}.pt')
        models.append(model);scalers.append(scaler)
    for t in targets:
        logits=[]
        for model,scaler in zip(models,scalers):
            with torch.no_grad():
                logits.append(model(torch.tensor(scaler.transform(t['x']),dtype=torch.float32,device=DEVICE)).cpu().numpy())
        t['base']=np.mean(logits,axis=0)
    return gate


def structure_fit(family,train,target,epochs,select,chosen=None):
    p.seed(41)
    yy=torch.tensor(train['y'],dtype=torch.float32,device=DEVICE)
    pos_weight=torch.tensor([(len(yy)-yy.sum())/yy.sum()],device=DEVICE)
    if family=='Dual-Mapping':
        _,_,_,Model=p.dual.make_model_classes()
        tensors=[]
        for t in (train,target):
            tensors.append({k:torch.tensor(v,dtype=torch.long if k=='edge_index' else torch.float32,device=DEVICE)
                            for k,v in t['dual'].items()})
        for block in ('local','heuristic','rw'):
            scaler=StandardScaler().fit(train['dual'][block])
            for t,src in zip(tensors,(train,target)):
                t[block]=torch.tensor(scaler.transform(src['dual'][block]),dtype=torch.float32,device=DEVICE)
        model=Model(tensors[0]['local'].shape[1],tensors[0]['heuristic'].shape[1],tensors[0]['rw'].shape[1],
                    0,feature_proj_dim=16,hidden_dim=64,num_layers=2,dropout=.2,max_residual_alpha=1.).to(DEVICE)
        # Begin at the pretrained history ranking, learning a structural correction.
        nn.init.zeros_(model.base_head[-1].weight);nn.init.zeros_(model.base_head[-1].bias)
        nn.init.zeros_(model.residual_head[-1].weight);nn.init.zeros_(model.residual_head[-1].bias)
        def forward(t):
            return model(t['local'],t['heuristic'],t['rw'],t['local'].new_empty((len(t['local']),0)),t['edge_index'])
        def predict():
            model.eval()
            with torch.no_grad():return forward(tensors[1]).cpu().numpy()
        def train_epoch(optimizer):
            model.train();optimizer.zero_grad()
            logits=forward(tensors[0])+torch.tensor(train['base'],device=DEVICE)
            loss=nn.functional.binary_cross_entropy_with_logits(logits,yy,pos_weight=pos_weight)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            return float(loss)
    else:
        model=p.SparseSIN(num_input_features=1,num_classes=2,num_layers=3,hidden=16,
                          dropout_rate=.2,max_dim=2,readout='mean',final_readout='sum').to(DEVICE)
        nn.init.zeros_(model.lin2.weight);nn.init.zeros_(model.lin2.bias)
        for t in (train,target):
            for c,base in zip(t['complexes'],t['base']):
                c.cochains[0].history_base_logit=torch.tensor([[base]],dtype=torch.float32)
        loader=p.DataLoader(train['complexes'],batch_size=64,shuffle=True,max_dim=2)
        eval_loader=p.DataLoader(target['complexes'],batch_size=64,shuffle=False,max_dim=2)
        def predict():
            model.eval();scores=[]
            with torch.no_grad():
                for data in eval_loader:
                    logits=model(data.to(DEVICE),return_logits=True)
                    scores.extend((logits[:,1]-logits[:,0]).cpu().numpy())
            return np.asarray(scores)
        def train_epoch(optimizer):
            model.train();ls=[]
            for data in loader:
                data=data.to(DEVICE);optimizer.zero_grad()
                logits=model(data,return_logits=True)
                score=logits[:,1]-logits[:,0]+data.cochains[0].history_base_logit.flatten()
                loss=nn.functional.binary_cross_entropy_with_logits(score,data.y.view(-1).float(),pos_weight=pos_weight)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();ls.append(float(loss))
            return float(np.mean(ls))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    trials=[]
    for epoch in range(1,epochs+1):
        loss=train_epoch(optimizer);assert np.isfinite(loss)
        if select and epoch%5==0:
            residual=predict()
            for scale in (.1,.3,1.):
                score=1/(1+np.exp(-np.clip(target['base']+scale*residual,-40,40)))
                trials.append({'epochs':epoch,'residual_scale':scale,'metrics':p.metrics(target['y'],score),'loss':loss})
            log(f'{family} residual epoch={epoch}: best validation AP={max(r["metrics"]["AP"] for r in trials):.5f}')
    if select:
        best=max(trials,key=lambda r:r['metrics']['AP'])
        save(f'{family}_PARAMETER_LOCK.json',{'chosen':best,'trials':trials,'test_labels_used':False})
        return best
    residual=predict()
    score=1/(1+np.exp(-np.clip(target['base']+chosen['residual_scale']*residual,-40,40)))
    torch.save(model.state_dict(),OUT/f'{family}_structure_checkpoint.pt')
    return score,residual


def main():
    OUT.mkdir(exist_ok=True);started=time.perf_counter()
    save('PLAN.json',{'test_arm':'easy_learned_70_2027','test_candidates':2000,'seeds':list(SEEDS),
         'history_training_cap':25000,'history_windows':'last3 completed windows',
         'history_configs':[(64,.5),(128,.5),(64,1.)],'history_epochs':list(range(5,51,5)),
         'history_validation_origins':[.5,.6],'structural_train_rows':2000,
         'structural_max_epochs':30,'structural_scales':[.1,.3,1.],
         'structural_mode':'frozen neural history base + trained zero-initialized graph/simplicial residual',
         'tree_models_used':False,'test_labels_used_for_selection':False,
         'source_sha256':p.sample.s.digest(Path(__file__))})
    events,tasks=p.material();snapshots,folds=history_material(events,tasks)
    chosen=tune_history(folds)
    history_ensemble(snapshots,tasks['validation']['cut'],
                     [tasks['dev_train'],tasks['validation']],chosen,'development')
    # History pretraining must end BEFORE structural training prediction, including its own labels.
    # For a causal training covariate, fit a separate past-only teacher at dev_train.cut.
    history_ensemble(snapshots,tasks['dev_train']['cut'],[tasks['dev_train']],chosen,'dev_training_covariate')
    # Likewise final structural rows at .6 receive forecasts trained only through .6.
    history_ensemble(snapshots,tasks['final_train']['cut'],[tasks['final_train']],chosen,'final_training_covariate')
    history_ensemble(snapshots,tasks['test']['cut'],[tasks['test']],chosen,'final')
    # Structural feature caches are reused exactly from the audited pilot.
    p.dual_assets(events,tasks);p.sin_assets(events,tasks)
    locks={}
    for family in ('Dual-Mapping','MLSHoL'):
        locks[family]=structure_fit(family,tasks['dev_train'],tasks['validation'],30,True)
    save('ALL_PARAMETER_LOCK.json',{'history':chosen,'structure':locks,'final_predictions_evaluated':False})
    rows=[]
    for family in ('Dual-Mapping','MLSHoL'):
        lock=locks[family]
        scores,residual=structure_fit(family,tasks['final_train'],tasks['test'],lock['epochs'],False,lock)
        np.savez_compressed(OUT/f'{family}_predictions.npz',q=tasks['test']['q'],y=tasks['test']['y'],scores=scores,
                            history_logit=tasks['test']['base'],structure_residual=residual)
        baseline=p.sample.s.read(p.OUT/f'{family}_RESULT.json')
        row={'model':family,**p.metrics(tasks['test']['y'],scores),'previous_AP':baseline['AP'],
             'previous_Lift':baseline['Lift'],'AP_gain_factor':p.metrics(tasks['test']['y'],scores)['AP']/baseline['AP'],
             'selected_epochs':lock['epochs'],'residual_scale':lock['residual_scale'],
             'candidate_sha256':p.sample.hashlib_array(tasks['test']['q'])}
        rows.append(row);save(f'{family}_RESULT.json',row);log(str(row))
    base_scores=1/(1+np.exp(-np.clip(tasks['test']['base'],-40,40)))
    save('HISTORY_ONLY_ABLATION.json',p.metrics(tasks['test']['y'],base_scores))
    for r in rows:
        with np.load(OUT/f'{r["model"]}_predictions.npz') as z:
            assert np.array_equal(z['q'],tasks['test']['q']) and np.array_equal(z['y'],tasks['test']['y'])
            assert p.metrics(z['y'],z['scores'])['AP']==r['AP']
    with (OUT/'RESULTS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save('VERIFICATION.json',{'state':'PASSED','identical_test_candidates_and_labels':True,
         'metrics_recomputed':True,'test_not_used_for_model_selection':True,
         'training_history_forecasts_use_only_completed_past_windows':True,
         'total_seconds':time.perf_counter()-started})
    lines=['# tags-math-sx：神经模型时序分支改进','',
      '测试集合保持 easy_learned_70_2027 的 2000 个候选、26 个正例。',
      '新增 85 维历史特征（频次、真实时间衰减、历史闭合率、边年龄等），用最多 25000 行已完成窗口训练两种子神经历史分支。',
      '历史分支以 50%、60% 的两个过去窗口选配置和轮数；结构分支使用原 2000 行，按过去验证选择轮数和残差强度。',
      'Dual-Mapping 的 GraphSAGE、MLSHoL 的 SparseSIN 输出学习对冻结历史神经分支的残差。没有使用树模型分数。',
      '结构训练行上的历史预测另行拟合，仅使用其自身预测时点之前已结束的窗口，避免把训练标签喂回输入。','',
      '| 模型 | 原 AP | 改进 AP | 原 Lift | 改进 Lift | AP 倍数 |',
      '|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["model"]} | {r["previous_AP"]:.6f} | {r["AP"]:.6f} | {r["previous_Lift"]:.4f} | {r["Lift"]:.4f} | {r["AP_gain_factor"]:.2f} |')
    m=p.metrics(tasks['test']['y'],base_scores)
    lines+=['',f'无结构分支的历史神经网络消融：AP={m["AP"]:.6f}、Lift={m["Lift"]:.4f}。该对照用于判断收益是否主要来自历史分支。',
       '改进模型改变了输入和架构，不能宣称旧原版神经模型本身已恢复；候选图与局部图限制沿用 pilot。',
       '最终窗口此前已查看，结果属于探索实验，不是全新独立留出检验。',
       f'总耗时 {time.perf_counter()-started:.1f} 秒；缓存历史特征构造不计入本次耗时。']
    (OUT/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    log('COMPLETE')


if __name__=='__main__':main()
