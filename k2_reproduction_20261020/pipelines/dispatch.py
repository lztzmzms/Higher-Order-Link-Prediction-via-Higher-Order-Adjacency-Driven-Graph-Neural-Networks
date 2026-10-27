import os
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from data.prepare import ROOT, config, material, evaluation_labels
from train import train

def run_model(dataset,model,seed,device='auto',prepared=None):
    if config(dataset).get('pipeline')!='native':
        q,pred=train(dataset,model,seed,device,prepared)
        return evaluation_labels(dataset,q),pred
    cache=ROOT/'.cache'/'predictions'
    cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        output=Path(directory)/'actual_predictions.npz'
        command=[sys.executable,'-m','pipelines.native_runner','--dataset',dataset,
                 '--model',model,'--seed',str(seed),'--device',device,'--output',str(output)]
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        completed=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        if completed.returncode:
            raise RuntimeError(f'{dataset}/{model}/{seed} failed:\n'+completed.stdout[-8000:])
        with np.load(output,allow_pickle=False) as z:
            return z['y'].copy(),z['predictions'].copy()
