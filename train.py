import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models import network

def seed_everything(seed):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def optimizer(net, cfg):
    core = list(net.core.parameters())
    core_ids = {id(p) for p in core}
    return torch.optim.AdamW([
        dict(params=[p for p in net.parameters() if p.requires_grad and id(p) not in core_ids], lr=cfg['lr']),
        dict(params=[p for p in core if p.requires_grad], lr=cfg['lr'] * cfg['structure_lr_ratio']),
    ], weight_decay=cfg['weight_decay'])

def roles_for(model, structure, rw, q, components=None, mask_indices=()):
    roles = {}
    for role in ('fit', 'target'):
        roles[role] = dict(q=q[role], rw=rw[role],
                           block=structure[role] if model == 'Dual-Mapping' else {'rw': rw[role]},
                           complexes=None if model == 'Dual-Mapping' else structure[role])
        if components is not None:
            roles[role].update(x=torch.as_tensor(components[role], dtype=torch.float64, device=network.DEVICE),
                               component_masks=list(mask_indices))
    return roles

def update(net, opt, role, labels):
    pass

    net.train()
    opt.zero_grad(set_to_none=True)
    n = len(labels)
    prevalence = float(np.mean(labels))
    if not 0 < prevalence < 1:
        raise ValueError('Training candidates must contain both true-link classes')
    batch = int(net.cfg.get('batch_size', 32))
    if batch < 1:
        raise ValueError('batch_size must be positive')
    weight = float(net.cfg.get('loss_weight', 1.))
    if weight <= 0:
        raise ValueError('loss_weight must be positive')
    pos_weight = torch.tensor(min(10., ((1. - prevalence) / prevalence) ** .5),
                              device=network.DEVICE, dtype=torch.float64)
    order = np.random.permutation(n)
    total = 0.
    backward_calls = 0
    if net.model == 'Dual-Mapping':
        score = net(role, np.arange(n))
        loss = weight * F.binary_cross_entropy_with_logits(score,
            torch.as_tensor(labels, device=network.DEVICE, dtype=torch.float64),
            pos_weight=pos_weight, reduction='sum') / n
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite true-link BCE')
        loss.backward()
        backward_calls = 1
        total = float(loss.detach())
    else:
        for start in range(0, n, batch):
            indices = order[start:start+batch]
            loss = weight * F.binary_cross_entropy_with_logits(net(role, indices),
                torch.as_tensor(labels[indices], device=network.DEVICE, dtype=torch.float64),
                pos_weight=pos_weight, reduction='sum') / n
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite true-link BCE')
            loss.backward()
            backward_calls += 1
            total += float(loss.detach())
    net.epoch_execution = dict(forward_batches=backward_calls, backward_calls=backward_calls,
        gradient_scope='full_graph' if net.model == 'Dual-Mapping' else 'microbatch_forwards',
        effective_batch_size=n if net.model == 'Dual-Mapping' else min(batch, n))
    if any(not torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None):
        raise FloatingPointError('Nonfinite gradient')
    nn.utils.clip_grad_norm_(net.parameters(), 1.)
    opt.step()
    return total

def predict(net, role):
    net.eval()
    indices = np.arange(len(role['q']))
    if not len(indices):
        raise ValueError('No prediction candidates')
    batch = int(net.cfg.get('batch_size', 32)) if net.model == 'MLSHoL' else len(indices)
    with torch.no_grad():
        score = np.concatenate([net(role, indices[i:i+batch]).cpu().numpy()
                                for i in range(0, len(indices), batch)])
    if not np.isfinite(score).all():
        raise FloatingPointError('Nonfinite predictions')
    return score

def fit_model(model, roles, labels, cfg, seed, mode='component_joint', device='cpu'):
    from models.singleton_batchnorm import install
    network.DEVICE = torch.device(device)
    seed_everything(seed)
    if mode not in ('component_joint', 'structure_only'):
        raise ValueError(f'Unsupported prediction mode: {mode}')
    net = (network.Network(model, roles['fit'], cfg) if mode == 'component_joint'
           else network.StructureNetwork(model, roles['fit'], cfg))
    opt = optimizer(net, cfg)
    _, restore = install()
    try:
        epochs = int(cfg['epochs'])
        if epochs < 1:
            raise ValueError('epochs must be positive; joint_steps is no longer a training budget')
        net.training_history = []
        for epoch in range(1, epochs + 1):
            loss = update(net, opt, roles['fit'], labels)
            net.training_history.append(dict(epoch=epoch, weighted_BCE=loss,
                candidate_visits=epoch * len(labels), optimizer_updates=epoch,
                positives_per_epoch=int(np.sum(labels)), batch_size=int(cfg.get('batch_size',32)),
                **net.epoch_execution))
        score = predict(net, roles['target'])
    finally:
        restore()
    return net, score
