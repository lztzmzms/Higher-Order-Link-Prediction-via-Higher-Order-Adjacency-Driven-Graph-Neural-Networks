import os,sys,json,csv,hashlib,importlib.util,shutil,time
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
BUNDLE=ROOT/'reference'
DS=('email-Enron', 'contact-primary-school', 'email-Eu', 'tags-ask-ubuntu', 'tags-math-sx', 'NDC-substances')
MODELS=('Dual-Mapping','MLSHoL')
FILES=('MAIN_K2_RESULTS', 'RW_ABLATION', 'DUAL_DEPTH_ABLATION', 'MRNL_COMPARISON', 'MAX_SIMPLEX_DIMENSION_ABLATION')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8');t.replace(p)
def csvout(p,rows,fields=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.csv.tmp')
 fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 with t.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields or ['state']);w.writeheader();w.writerows(rows)
 t.replace(p)
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def triples(p): return [tuple(sorted(map(int,s.split()))) for s in Path(p).read_text().splitlines() if s.strip()]
def worker(jobid,seed,smoke=False):
 from scipy.stats import rankdata
 from sklearn.metrics import average_precision_score
 import torch
 plan=read(OUT/'PLAN.json');job=globals().get('JOB_OVERRIDE') or next(j for j in plan['jobs'] if j['id']==jobid);e=job['effective'];ds=e['dataset'];model=e['model'];code=ROOT/job['code'];proto=ROOT/job['protocol'];d=OUT/('smoke' if smoke else 'jobs')/jobid/str(seed);d.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(ROOT));sys.path.insert(0,str(code));assert sha(ROOT/job['source_config'])==e['source_config_sha256'];cfg=read(ROOT/job['source_config']);cfg['model_seeds']=[seed]
 meta=read(proto/'READY.json');spec=cfg['dual' if model==MODELS[0] else 'mlshol'][ds]
 if ds in ('email-Eu','tags-ask-ubuntu') or e['no_activity_inputs']:
  spec.update(prior='none',activity_prior=False,temporal_activity_alpha=0,validation_select_activity_alpha=False,validation_select_temporal_alpha=False)
 if e['no_activity_inputs']:spec['strength_feature']=False
 from frequency_hooks import configure
 configure(spec,e['frequency'],model)
 cfgpath=d/'CONFIG.json';save(cfgpath,cfg);save(d/'EFFECTIVE.json',e)
 out=ROOT/'artifacts'/('s' if smoke else 'j')/jobid/str(seed);args=['--dataset',ds,'--data-root',str(proto),'--results-root',str(out),'--config',str(cfgpath),'--device',os.environ.get('DEVICE',('cuda:0' if torch.cuda.is_available() else 'cpu')),'--run-name','run']
 if smoke:args+=['--epochs-override','1']
 if model==MODELS[0]:
  adapter=mod('dual_adapter',code/'run_dual_seed.py');import node_classfication_embedding_original_rwsearch as engine
  assert Path(engine.__file__).resolve().is_relative_to(code.resolve())
  engine.load_priority_mask_for_train=lambda args,n:torch.zeros(n,dtype=torch.bool)
  from frequency_hooks import patch_native_dual
  patch_native_dual(engine,e['frequency'])
  original=engine.main_rwsearch
  def main():
   for flag,key in [('--train_strength_context_ratio','train_ratio'),('--test_strength_context_ratio','test_ratio')]:sys.argv[sys.argv.index(flag)+1]=str(meta[key])
   if not e['rw']:sys.argv+=['--disable-rw']
   save(d/'MODEL_COMMAND.json',{'argv':sys.argv,'engine':engine.__file__,'engine_sha256':sha(engine.__file__),'adapter_sha256':sha(code/'run_dual_seed.py'),'driver_sha256':sha(__file__)})
   original()
  engine.main_rwsearch=main;args+=['--model-layers',str(e['layers']),'--shared-rw-cache',str(ROOT/'artifacts/rw'/ds/'dual')]
 else:
  adapter=mod('mls_adapter',code/'run_mlshol_seed.py');import exp.run_all_sparse_sin as engine
  assert Path(engine.__file__).resolve().is_relative_to(code.resolve())
                                                                               
  def causal_rw(dataset,args,device):
   maps={}
   for split in ('train','test'):
    qs=engine.read_triangles(engine.split_paths(dataset,split,args)['open']);qs=sorted(set(qs));index={q:i for i,q in enumerate(qs)}
    cache=ROOT/'artifacts/rw'/dataset/('mls_'+split+'_'+digest({'q':qs,'seed':args.rw_seed,'dim':args.rw_dim,'walks':args.rw_walks_per_node,'epochs':args.rw_epochs})[:16]+'.npz');cache.parent.mkdir(parents=True,exist_ok=True)
    v=None
    if ds in ('NDC-substances','threads-math-sx'):assert cache.exists(), 'Locked RW cache missing: '+str(cache)
    if cache.exists():
     try:v=np.load(cache)['embedding']
     except Exception as exc:os.replace(cache,cache.with_name(cache.name+'.corrupt-'+str(os.getpid())));print('Rebuilding corrupt RW cache:',exc,flush=True)
    if v is None:
     edges=engine.dual_edges_for_candidates(qs,index);ip,ix=engine.build_csr_adj(sorted(edges),len(qs));v=engine.train_aligned_dual_rw(ip,ix,len(qs),np.zeros(len(qs),dtype=bool),args,device);tmp=cache.with_name(cache.stem+'.partial-'+str(os.getpid())+'.npz');np.savez_compressed(tmp,triangles=qs,embedding=v);os.replace(tmp,cache)
    maps[split]={q:v[i] for i,q in enumerate(qs)}
   return maps
  original=engine.main
  def main():
   for flag,key in [('--train-history-context-ratio','train_ratio'),('--test-history-context-ratio','test_ratio')]:sys.argv[sys.argv.index(flag)+1]=str(meta[key])
   sys.argv[sys.argv.index('--rw-source')+1]='aligned_union'
   if e['representation'] not in ('native','MRNL'):sys.argv[sys.argv.index('--num-input-features')+1]='1'
   if not e['rw']:sys.argv[sys.argv.index('--rw-fusion')+1]='none'
   save(d/'MODEL_COMMAND.json',{'argv':sys.argv,'engine':engine.__file__,'engine_sha256':sha(engine.__file__),'adapter_sha256':sha(code/'run_mlshol_seed.py'),'driver_sha256':sha(__file__),'rw_repair':'separate context graph embeddings; no label priority'})
   original()
                                                                       
  source=Path(engine.__file__).read_text(encoding='utf-8')
  start=source.index('            feature_map, args.resolved_rw_cache = load_or_build_aligned_dual_rw(')
  end=source.index('        elif args.rw_source == "dual_mapping_full":',start)
  source=source[:start]+'            args.dual_rw_feature_maps = causal_rw(dataset,args,device)\n            args.resolved_rw_cache = "causal_separate_no_label_priority"\n'+source[end:]
  exec(compile(source,engine.__file__,'exec'),engine.__dict__);engine.causal_rw=causal_rw;original=engine.main;engine.main=main
  cfg['mlshol'][ds]['rw_source']='aligned_union';save(cfgpath,cfg)
  args+=['--max-dim',str(e['max_dim']),'--cache-dir',str(ROOT/'artifacts/locked_geometry'/ds if ds in ('NDC-substances','threads-math-sx') else ROOT/'artifacts/cache'/jobid),'--rw-cache-dir',str(ROOT/'artifacts/rw'/ds/'mls')]
  if e['representation'] not in ('native','MRNL'):
   emb=mod('alternative_embeddings',ROOT/'ablation_embeddings.py');folder=ROOT/'artifacts/embeddings'/ds/e['representation']
   if not (folder/'audit.json').exists():emb.build_pair(ds,e['representation'],proto,folder)
   args+=['--node-embedding-train-npz',str(folder/'train.npz'),'--node-embedding-test-npz',str(folder/'test.npz')]
 sys.argv=['adapter']+args;started=time.time();torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
 saved_load=torch.nn.Module.load_state_dict
 def save_restored(self,*a,**kw):
  value=saved_load(self,*a,**kw)
  caller=sys._getframe(1)
  if model==MODELS[0] and Path(caller.f_code.co_filename).resolve()==Path(engine.__file__).resolve():torch.save(self.state_dict(),d/'CHECKPOINT.pt')
  return value
 torch.nn.Module.load_state_dict=save_restored
 try:adapter.main()
 finally:torch.nn.Module.load_state_dict=saved_load
 if model==MODELS[1]:
  checkpoints=list(out.rglob('best_checkpoint.pt'))
  if checkpoints:shutil.copy2(checkpoints[0],d/'CHECKPOINT.pt')
 save(d/'RESOURCE.json',{'runtime_sec':time.time()-started,'peak_gpu_allocated_mb':(torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None)})
 if smoke:save(d/'DONE.json',{'state':'SMOKE_PASSED'});return
 paths=list(out.rglob('*scores*seed*'+str(seed)+'.npz'));assert len(paths)==1,paths
 z=np.load(paths[0]);qs=triples(proto/'dataset/test'/f'{ds}_openTriangles_test.txt');pos=set(triples(proto/'dataset/test'/f'{ds}_new_closures_test.txt'));y=np.array([q in pos for q in qs],int);ix={tuple(sorted(q)):i for i,q in enumerate(z['target_triplets'])};order=[ix[q] for q in qs]
 key=('y_score_target_unfused' if model==MODELS[0] else 'y_score_unfused') if ds in ('email-Eu','tags-ask-ubuntu') else ('y_score_target' if model==MODELS[0] else 'y_score')
 if key not in z:raise RuntimeError(f'Required export missing {key}: {z.files}')
 native=np.asarray(z[key])[order];r=lambda x:rankdata(x,method='average')/len(x)
 if ds in ('email-Eu','tags-ask-ubuntu'):final=.8*r(np.load(proto/'HISTORY.npz')['history'])+.2*r(native)
 else:final=native
 if ds in ('NDC-substances','threads-math-sx'):
  selection=np.load(ROOT/'reference'/ds/'SELECTED_SAMPLE.npz');ids=selection['base_indices'];np.testing.assert_array_equal(np.asarray(qs)[ids],selection['q']);np.testing.assert_array_equal(y[ids],selection['y'])
  np.savez_compressed(d/'CONTEXT_PREDICTIONS.npz',triplets=qs,y_true=y,native=native,final=final)
  qs=selection['q'];y=selection['y'];native=native[ids];final=final[ids]
 np.savez_compressed(d/'FORECAST.npz',triplets=qs,final=final)
 save(d/'FORECAST_LOCK.json',{'sha256':sha(d/'FORECAST.npz'),'config_sha256':sha(cfgpath),'test_parameter_search':False})
 np.savez_compressed(d/'PREDICTIONS.npz',triplets=qs,y_true=y,native=native,final=final)
 ap=float(average_precision_score(y,final));nap=float(average_precision_score(y,native));resource=read(d/'RESOURCE.json')
 result={'id':jobid,'dataset':ds,'model':model,'seed':seed,'AP':ap,'Lift':ap/y.mean(),'native_AP':nap,'native_Lift':nap/y.mean(),'candidates':len(y),'positives':int(y.sum()),**resource,'peak_process_gpu_mb':0,'prediction_sha256':sha(d/'PREDICTIONS.npz')};save(d/'RESULT.json',result)
