                                                                                         

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

