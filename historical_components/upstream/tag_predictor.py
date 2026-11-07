from types import SimpleNamespace
import random
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)
p=SimpleNamespace(seed=seed)
DEVICE=torch.device("cpu")

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
