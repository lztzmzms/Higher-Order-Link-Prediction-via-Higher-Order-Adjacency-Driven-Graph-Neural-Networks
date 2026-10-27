"""Regenerate the original tags-math H input from its completed historical windows."""
import random
from pathlib import Path
import numpy as np
import torch
from pipelines.inputs import ROOT, unpack
from pipelines.history_sampling import capsnap
from pipelines import history

def regenerate(device='auto'):
    folder=unpack('tags-math-sx')/'material'
    snapshots=[]
    for path in sorted((folder/'history').glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            snapshots.append(dict(x=z['x'],y=z['y'],cut=int(z['cut']),label_end=int(z['label_end'])))
    from data.prepare import config
    cfg=config('tags-math-sx')['original_history']
    device=torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
    history.DEVICE=device
    old_determinism=torch.are_deterministic_algorithms_enabled()
    numpy_state,python_state=np.random.get_state(),random.getstate()
    output={}
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(False)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            for split in ('train','test'):
                with np.load(folder/f'{split}.npz') as z:target={k:z[k] for k in z.files}
                x,y,gate=capsnap(snapshots,int(target['cut']),25000)
                if gate['max_training_label_end']>int(target['cut']):
                    raise ValueError('Historical H training crosses its prediction cut')
                logits=[]
                for seed in cfg['ensemble_seeds']:
                    _,_,prediction,_=history.history_fit(x,y,target['x'],cfg,seed,cfg['epochs'])
                    logits.append(prediction)
                mean=np.mean(logits,axis=0)
                output[split]=dict(q=target['q'],h=1/(1+np.exp(-np.clip(mean,-40,40))))
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.use_deterministic_algorithms(old_determinism)
    return output
