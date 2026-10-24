                                                          
import hashlib, importlib.util, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
 p=Path(p)
 if os.name=='nt' and not str(p).startswith('\\\\?\\'):p=Path('\\\\?\\'+str(p.resolve()))
 return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2),encoding='utf-8');os.replace(t,p)

def triples(p):
 import numpy as np
 return np.array([list(map(int,l.split())) for l in p.read_text().splitlines() if l.strip()],dtype=np.int64)

def expected(j):
 import numpy as np
 ds=j['effective']['dataset'];proto=ROOT/j['protocol'];q=triples(proto/'dataset/test'/(ds+'_openTriangles_test.txt'));pos=set(map(tuple,triples(proto/'dataset/test'/(ds+'_new_closures_test.txt'))));y=np.array([tuple(t) in pos for t in q],int)
 if ds in ('NDC-substances','threads-math-sx'):
  z=np.load(ROOT/'reference'/ds/'SELECTED_SAMPLE.npz');np.testing.assert_array_equal(q[z['base_indices']],z['q']);np.testing.assert_array_equal(y[z['base_indices']],z['y']);return z['q'],z['y'],len(q)
 return q,y,len(q)

def frozen_worker(j,seed):
 import numpy as np,torch
 from scipy.stats import rankdata
 from sklearn.metrics import average_precision_score
 from frequency_hooks import configure,patch_native_dual
 ds='contact-high-school';e=j['effective'];model=e['model'];code=ROOT/j['code'];proto=ROOT/j['protocol'];d=OUT/'jobs'/j['id']/str(seed);d.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(code));cfg=read(ROOT/j['source_config']);cfg['model_seeds']=[seed];configure(cfg['dual' if model=='Dual-Mapping' else 'mlshol'][ds],e['frequency'],model);save(d/'CONFIG.json',cfg)
 path=code/('run_dual_seed.py' if model=='Dual-Mapping' else 'run_mlshol_seed.py');s=importlib.util.spec_from_file_location('adapter',path);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
 args=['adapter','--dataset',ds,'--data-root',str(proto),'--results-root',str(d/'native'),'--config',str(d/'CONFIG.json'),'--device',os.environ.get('DEVICE','cuda:0'),'--run-name','run']
 if model=='Dual-Mapping':
  args=args[:-2]                                                 
  import node_classfication_embedding_original_rwsearch as eng
  patch_native_dual(eng,e['frequency'])
 else:
  import exp.run_all_sparse_sin as eng
  args+=['--max-nodes-per-step',str(read(ROOT/'configs/highschool_pipeline.json').get('mlshol_max_nodes_per_step',{}).get(ds,35))]
 sys.argv=args;save(d/'MODEL_COMMAND.json',dict(argv=args));start=time.time()
 if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
 restored=torch.nn.Module.load_state_dict
 def capture(self,*aa,**kw):
  result=restored(self,*aa,**kw)
  if Path(sys._getframe(1).f_code.co_filename).resolve()==Path(eng.__file__).resolve():torch.save(self.state_dict(),d/'CHECKPOINT.pt')
  return result
 torch.nn.Module.load_state_dict=capture
 try:a.main()
 finally:torch.nn.Module.load_state_dict=restored
 paths=list((d/'native').rglob('*scores*seed*'+str(seed)+'.npz'));assert len(paths)==1
 z=np.load(paths[0]);q,y,_=expected(j)
 if 'target_triplets' in z:
  ix={tuple(sorted(t)):i for i,t in enumerate(z['target_triplets'])};order=[ix[tuple(t)] for t in q]
 else:
  assert model=='Dual-Mapping';np.testing.assert_array_equal(z['target_node_ids'],np.arange(len(q)));order=np.arange(len(q))
 np.testing.assert_array_equal(z['y_true_target' if model=='Dual-Mapping' else 'y_true'][order],y)
 score=z['y_score_target' if model=='Dual-Mapping' else 'y_score'][order];h=np.load(proto/'HISTORY.npz')['dual_rank' if model=='Dual-Mapping' else 'mlshol_rank'];m=read(ROOT/'configs/highschool_pipeline.json');w=m['historical_fusion'][ds][model];rank=lambda x:rankdata(x,method='average')/len(x)
 final=(1-w)*(rank(h) if m['history_rerank'].get(ds,True) else h)+w*rank(score)
 np.savez_compressed(d/'PREDICTIONS.npz',triplets=q,y_true=y,native=score,final=final);ap=float(average_precision_score(y,final));save(d/'RESULT.json',dict(id=j['id'],dataset=ds,model=model,seed=seed,AP=ap,Lift=ap/y.mean(),runtime_sec=time.time()-start,peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else None,prediction_sha256=sha(d/'PREDICTIONS.npz')))

def worker(j,seed):
 import torch
 from scatter_compat_20261015 import ensure_scatter
 ensure_scatter()
 if os.environ.get('DEVICE','').startswith('cuda:'):torch.cuda.set_device(int(os.environ['DEVICE'].split(':')[1]))
 if j['effective']['pipeline']=='residual':
  import residual_worker as r
  r.worker(j,seed)
 elif j['effective']['pipeline']=='frozen':frozen_worker(j,seed)
 else:
  import native_worker as n
  if j['effective']['model']=='MLSHoL':
   import singleton_batchnorm
   stats,restore=singleton_batchnorm.install()
   try:n.worker(j['id'],seed)
   finally:restore();save(OUT/'jobs'/j['id']/str(seed)/'RUNTIME_REPAIR.json',stats)
  else:n.worker(j['id'],seed)
