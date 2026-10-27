"""Final selected F training; no search, diagnostics, or test-time selection."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import random
import numpy as np
import torch
from torch import nn
from data.prepare import config, material, geometry
from models.network import Network
from models.singleton_batchnorm import install

def seed_everything(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def optimizer(net, cfg):
    core = list(net.core.parameters())
    ids = {id(p) for p in core}
    return torch.optim.AdamW([
        dict(params=[p for p in net.parameters() if id(p) not in ids and p.requires_grad], lr=cfg['lr']),
        dict(params=[p for p in core if p.requires_grad], lr=cfg['lr']*cfg['structure_lr_ratio'])],
        weight_decay=cfg['weight_decay'])

def update(net, opt, labels, h, teacher):
    net.train()
    opt.zero_grad(set_to_none=True)
    ix = (np.random.choice(len(labels), min(net.cfg['batch_size'],len(labels)), replace=False)
          if net.model_name == 'MLSHoL' and not teacher else np.arange(len(labels)))
    prediction = net('fit', ix, teacher=teacher)
    target = h[torch.as_tensor(ix, device=net.device),2]*6
    if teacher:
        loss = nn.functional.mse_loss(prediction, target)
    else:
        y = torch.tensor(labels[ix], dtype=torch.float32, device=net.device)
        weight = min(10., ((len(labels)-labels.sum())/labels.sum())**.5)
        loss = nn.functional.binary_cross_entropy_with_logits(prediction, y,
            pos_weight=torch.tensor(weight, device=net.device))*net.cfg['loss_weight']
        count = min(512, 2*len(ix))
        ii = torch.randint(len(ix), (count,), device=net.device)
        jj = torch.randint(len(ix), (count,), device=net.device)
        loss = loss + net.cfg['rank_weight']*nn.functional.binary_cross_entropy_with_logits(
            prediction[ii]-prediction[jj], torch.sigmoid((target[ii]-target[jj]).detach()))
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training objective')
    loss.backward()
    if any(not torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None):
        raise FloatingPointError('Nonfinite gradient')
    if not teacher and net.cfg['teacher_policy'] == 'freeze':
        for p in net.h_encoder.parameters():
            p.grad = None
    nn.utils.clip_grad_norm_(net.parameters(), 1.)
    opt.step()

def train(dataset, model, seed, device='auto', prepared=None):
    spec = config(dataset)
    cfg = spec['models'][model]
    if seed not in cfg['seeds']:
        raise ValueError('Only the original three seeds are supported')
    seed_everything(seed)
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
    data = material(dataset, model) if prepared is None else prepared
    structure, rw = geometry(dataset, model, device)
    if any(len(structure[r]['local'] if model=='Dual-Mapping' else structure[r]) != len(data[p+'_q'])
           for r,p in [('fit','train'),('target','target')]):
        raise ValueError('Structural input/candidate alignment changed')
    h = {r:torch.tensor(data[p+'_h4'], dtype=torch.float32, device=device)
         for r,p in [('fit','train'),('target','target')]}
    seed_everything(seed)
    net = Network(model, structure, h, cfg, device, rw)
    opt = optimizer(net, cfg)
    _, restore = install()
    try:
        for _ in range(cfg['teacher_steps']):
            update(net, opt, data['train_y'], h['fit'], True)
        for _ in range(cfg['selected_joint_steps'][str(seed)]):
            update(net, opt, data['train_y'], h['fit'], False)
        net.eval()
        with torch.no_grad():
            structure = net.encode_structure('target', None)
            predictions = net.fuse(h['target'], structure).cpu().numpy().astype(np.float64)
        if predictions.shape != (len(data['target_q']),) or not np.isfinite(predictions).all():
            raise FloatingPointError('Invalid real model predictions')
        return data['target_q'], predictions
    finally:
        restore()
