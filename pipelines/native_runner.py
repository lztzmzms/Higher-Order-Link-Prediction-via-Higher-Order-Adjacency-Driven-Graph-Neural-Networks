import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from data.complexes import PackedComplexes
from data.prepare import config
from pipelines.inputs import ROOT, native_check, triples
from models.singleton_batchnorm import install

def run(dataset, model, seed, device, output, epochs=None):
    spec = config(dataset)
    if spec.get('pipeline')!='native' or seed not in spec['models'][model]['seeds']:
        raise ValueError('Unknown original native dataset/model/seed')
    folder = native_check(dataset)
    proto = folder/'protocol'
    ratios = json.loads((proto/'READY.json').read_text())
    cfg = dict(dual={dataset:spec['models']['Dual-Mapping']['parameters']},
               mlshol={dataset:spec['models']['MLSHoL']['parameters']},
               model_seeds=[seed],rw_seed=spec['rw_seed'])
    torch.set_num_threads(1)
    if device=='auto':
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda:'):
        torch.cuda.set_device(int(device.split(':')[1]))
    cache=ROOT/'.cache'/'runs'
    cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        work=Path(temporary)
        cfgpath=work/'config.json'
        cfgpath.write_text(json.dumps(cfg))
        arguments=['adapter','--dataset',dataset,'--data-root',str(proto),
                   '--results-root',str(work/'output'),'--config',str(cfgpath),'--device',device]
        if epochs is not None:
            arguments+=['--epochs-override',str(epochs)]
        if model=='Dual-Mapping':
            from pipelines.native import dual_adapter as adapter, dual as engine
            engine.load_priority_mask_for_train=lambda args,n:torch.zeros(n,dtype=torch.bool)
            original=engine.main_rwsearch
            def execute():
                for flag,key in [('--train_strength_context_ratio','train_ratio'),('--test_strength_context_ratio','test_ratio')]:
                    sys.argv[sys.argv.index(flag)+1]=str(ratios[key])
                original()
            engine.main_rwsearch=execute
            arguments+=['--model-layers',str(spec['models'][model]['layers']),
                        '--shared-rw-cache',str(folder/'rw/dual')]
        else:
            from pipelines.native import mlshol_adapter as adapter, mlshol as engine
            original=engine.main
            def execute():
                for flag,key in [('--train-history-context-ratio','train_ratio'),('--test-history-context-ratio','test_ratio')]:
                    sys.argv[sys.argv.index(flag)+1]=str(ratios[key])
                original()
            engine.main=execute
            def complexes(ds,split,args):
                role='fit' if split=='train' else 'target'
                packed=PackedComplexes(ROOT/'data'/ds/f'complexes_{role}.npz')
                q=packed.arrays['q']
                np.testing.assert_array_equal(q,triples(proto/'dataset'/split/f'{ds}_openTriangles_{split}.txt'))
                positive=set(map(tuple,triples(proto/'dataset'/split/f'{ds}_new_closures_{split}.txt')))
                items=[]
                for i in range(len(packed)):
                    c=packed[i]
                    c.y=torch.tensor([int(tuple(q[i]) in positive)],dtype=torch.long)
                    c.cochains[0].original_node_ids=torch.from_numpy(packed.original_nodes(i))
                    items.append(c)
                engine.attach_aux_features(items,q,split,args)
                return items,json.loads(str(packed.arrays['stats'].item()))
            engine.load_or_build_complexes=complexes
            arguments+=['--run-name','run','--max-dim','2','--cache-dir',str(work/'complexes'),
                        '--rw-cache-dir',str(work/'rw')]
        _,restore=install()
        sys.argv=arguments
        try:
            adapter.main()
        finally:
            restore()
        paths=list((work/'output').rglob('*scores*seed*'+str(seed)+'.npz'))
        if len(paths)!=1:
            raise RuntimeError('The original engine did not export exactly one prediction file')
        q=triples(proto/'dataset/test'/f'{dataset}_openTriangles_test.txt')
        positive=set(map(tuple,triples(proto/'dataset/test'/f'{dataset}_new_closures_test.txt')))
        y=np.asarray([tuple(t) in positive for t in q],dtype=np.int64)
        with np.load(paths[0],allow_pickle=False) as values:
            order={tuple(sorted(map(int,t))):i for i,t in enumerate(values['target_triplets'])}
            indices=np.asarray([order[tuple(t)] for t in q])
            key='y_score_target' if model=='Dual-Mapping' else 'y_score'
            pred=values[key][indices]
        with np.load(folder/'selected_sample.npz',allow_pickle=False) as selected:
            ids=selected['base_indices']
            np.testing.assert_array_equal(q[ids],selected['q'])
            np.testing.assert_array_equal(y[ids],selected['y'])
            q,y,pred=q[ids],y[ids],pred[ids]
        if not np.isfinite(pred).all():
            raise FloatingPointError('Nonfinite actual native predictions')
        np.savez_compressed(output,q=q,y=y,predictions=pred)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--model',required=True,choices=['Dual-Mapping','MLSHoL'])
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=None)
    args=parser.parse_args()
    run(args.dataset,args.model,args.seed,args.device,args.output,args.epochs)
