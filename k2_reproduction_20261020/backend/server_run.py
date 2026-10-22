"""Paired, resumable 8-dataset frequency runs. References are never measurements."""
import argparse,csv,hashlib,importlib.util,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 p=Path(p)
 if os.name=='nt' and not str(p).startswith('\\\\?\\'):p=Path('\\\\?\\'+str(p.resolve()))
 return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2),encoding='utf-8');os.replace(t,p)
def csvout(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)) or ['state']);w.writeheader();w.writerows(rows)
def triples(p):
 import numpy as np
 return np.array([list(map(int,l.split())) for l in p.read_text().splitlines() if l.strip()],dtype=np.int64)
def expected(j):
 import numpy as np
 ds=j['effective']['dataset'];proto=ROOT/j['protocol'];q=triples(proto/'dataset/test'/(ds+'_openTriangles_test.txt'));pos=set(map(tuple,triples(proto/'dataset/test'/(ds+'_new_closures_test.txt'))));y=np.array([tuple(t) in pos for t in q],int)
 if ds in ('NDC-substances','threads-math-sx'):
  z=np.load(ROOT/'reference'/ds/'SELECTED_SAMPLE.npz');np.testing.assert_array_equal(q[z['base_indices']],z['q']);np.testing.assert_array_equal(y[z['base_indices']],z['y']);return z['q'],z['y'],len(q)
 return q,y,len(q)
def preflight():
 import numpy as np,torch
 counts={'email-Enron':(2574,119),'contact-primary-school':(4184,22),'contact-high-school':(25056,201),'email-Eu':(3222,31),'tags-ask-ubuntu':(4414,36),'tags-math-sx':(2000,26),'NDC-substances':(1750,23),'threads-math-sx':(28872,27)}
 for name,h in read(ROOT/'PACKAGE_FILES.json').items():assert sha(ROOT/name)==h,'Package changed: '+name
 p=read(ROOT/'PLAN.json');assert sum(len(j['seeds']) for j in p['jobs'])==192
 rows=[]
 for ds in counts:
  j=next(j for j in p['jobs'] if j['effective']['dataset']==ds);q,y,context=expected(j);assert (len(q),int(y.sum()))==counts[ds];assert len(set(map(tuple,q)))==len(q)
  if ds=='threads-math-sx':assert context==30000
  if ds=='NDC-substances':assert context==6000
  proto=ROOT/j['protocol']
  if ds!='contact-high-school':
   m=read(proto/'READY.json');ts=np.loadtxt(proto/'dataset/raw_data'/ds/(ds+'-times.txt'),dtype=np.int64);assert np.all(ts[1:]>=ts[:-1])
   for cut in (m['train_cut'],m['test_cut']):assert ts[cut-1]!=ts[cut]
  rows.append(dict(dataset=ds,candidates=len(q),positives=int(y.sum()),context=context,seeds=j['seeds'],order='frozen raw' if ds=='contact-high-school' else 'timestamp chronological'))
 OUT.mkdir(exist_ok=True);save(OUT/'PLAN.json',p);save(OUT/'PREFLIGHT.json',dict(state='PASSED',datasets=rows,torch=torch.__version__,cuda=torch.cuda.is_available()));print(json.dumps(rows),flush=True)
def report(verify=False):
 import numpy as np
 from sklearn.metrics import average_precision_score
 p=read(ROOT/'PLAN.json');rows=[];results={};groups=[];deltas=[]
 for j in p['jobs']:
  e=j['effective'];actual=[]
  for seed in j['seeds']:
   d=OUT/'jobs'/j['id']/str(seed);r=dict(dataset=e['dataset'],model=e['model'],arm=e['frequency'],seed=seed,id=j['id'],state='PENDING')
   if (d/'FAILURE.json').exists():r['state']='FAILED'
   if (d/'RESULT.json').exists():
    v=read(d/'RESULT.json');z=np.load(d/'PREDICTIONS.npz');q,y,ctx=expected(j);np.testing.assert_array_equal(q,z['triplets']);np.testing.assert_array_equal(y,z['y_true']);assert np.isfinite(z['final']).all();assert sha(d/'PREDICTIONS.npz')==v['prediction_sha256'];ap=float(average_precision_score(y,z['final']));assert abs(ap-v['AP'])<1e-10 and abs(ap/y.mean()-v['Lift'])<1e-8
    if ctx!=len(y):
     zz=np.load(d/'CONTEXT_PREDICTIONS.npz');sel=np.load(ROOT/'reference'/e['dataset']/'SELECTED_SAMPLE.npz');assert len(zz['triplets'])==ctx;np.testing.assert_array_equal(zz['final'][sel['base_indices']],z['final'])
    r.update(v,state='COMPLETE',arm=e['frequency']);actual.append(r);results[(e['dataset'],e['model'],e['frequency'],seed)]=r
   rows.append(r)
  g=dict(dataset=e['dataset'],model=e['model'],arm=e['frequency'],n=len(actual),expected_n=3,state='COMPLETE' if len(actual)==3 else 'PARTIAL' if actual else 'PENDING')
  for key in ('AP','Lift'):
   g[key+'_mean']=float(np.mean([r[key] for r in actual])) if actual else None;g[key+'_sample_sd']=float(np.std([r[key] for r in actual],ddof=1)) if len(actual)>1 else None
  groups.append(g)
 for j in p['jobs']:
  e=j['effective'];arm=e['frequency']
  if arm not in ('count','log'):continue
  for base in ('control','unit'):
   vv=[]
   for seed in j['seeds']:
    a=results.get((e['dataset'],e['model'],arm,seed));b=results.get((e['dataset'],e['model'],base,seed))
    if a and b:vv.append(a['Lift']-b['Lift'])
   deltas.append(dict(dataset=e['dataset'],model=e['model'],arm=arm,baseline=base,n=len(vv),Lift_delta_mean=float(np.mean(vv)) if vv else None,Lift_delta_sample_sd=float(np.std(vv,ddof=1)) if len(vv)>1 else None))
 csvout(OUT/'PER_SEED.csv',rows);csvout(OUT/'SUMMARY.csv',groups);csvout(OUT/'PAIRED_DELTAS.csv',deltas)
 references=read(ROOT/'reference/USER_REFERENCE_LIFT.json');comparison=[]
 for g in groups:
  if g['arm']!='control':continue
  ref=references[g['dataset']][0 if g['model']=='Dual-Mapping' else 1]
  comparison.append(dict(**g,prescribed_reference_Lift=ref,measured_minus_reference=g['Lift_mean']-ref if g['n'] else None,reference_is_three_seed_mean=g['dataset'] in ('email-Enron','contact-primary-school','NDC-substances','threads-math-sx'),numeric_pass_tolerance=None))
 csvout(OUT/'CONTROL_COMPARISON.csv',comparison)
 state=dict(complete=sum(r['state']=='COMPLETE' for r in rows),failed=sum(r['state']=='FAILED' for r in rows),total=192)
 save(OUT/('VERIFICATION.json' if verify else 'STATUS.json'),dict(state='PASSED_FOR_COMPLETED',all_complete=state['complete']==192,**state));print(json.dumps(state),flush=True)
 if verify and state['failed']:raise RuntimeError('Failed tasks remain; inspect FAILURE.json')
 return state
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
  args=args[:-2]  # Frozen Dual adapter has no --run-name option.
  import node_classfication_embedding_original_rwsearch as eng
  patch_native_dual(eng,e['frequency'])
 else:
  import exp.run_all_sparse_sin as eng
  args+=['--max-nodes-per-step',str(read(ROOT/'reference/HIGH_MANIFEST.json').get('mlshol_max_nodes_per_step',{}).get(ds,35))]
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
 score=z['y_score_target' if model=='Dual-Mapping' else 'y_score'][order];h=np.load(proto/'HISTORY.npz')['dual_rank' if model=='Dual-Mapping' else 'mlshol_rank'];m=read(ROOT/'reference/HIGH_MANIFEST.json');w=m['historical_fusion'][ds][model];rank=lambda x:rankdata(x,method='average')/len(x)
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
def run(args):
 import psutil
 preflight();lock=(OUT/'RUN.lock').open('a+b')
 if os.name=='posix':
  import fcntl
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Recover surviving workers after a supervisor restart.
 for proc in psutil.process_iter(['pid','cmdline']):
  cmd=proc.info['cmdline'] or []
  if proc.pid!=os.getpid() and 'worker' in cmd and str(ROOT/'server_run.py') in cmd:
   while proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE:time.sleep(10)
 jobs=read(ROOT/'PLAN.json')['jobs'];jobs.sort(key=lambda j:(j['effective']['frequency']!='control',j['effective']['dataset'],j['effective']['model']))
 for j in jobs:
  if args.dataset and j['effective']['dataset']!=args.dataset:continue
  if args.arm and j['effective']['frequency']!=args.arm:continue
  for seed in j['seeds']:
   d=OUT/'jobs'/j['id']/str(seed);d.mkdir(parents=True,exist_ok=True)
   if (d/'RESULT.json').exists():continue
   cmd=[sys.executable,'-u',str(ROOT/'server_run.py'),'worker','--job',j['id'],'--seed',str(seed)];start=time.time();peak=0;gpu=None
   with (d/'run.log').open('a',encoding='utf-8') as log:
    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name=='posix')
    while child.poll() is None:
     try:peak=max(peak,psutil.Process(child.pid).memory_info().rss)
     except psutil.Error:pass
     try:
      s=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3);v=[float(l.split(',')[1]) for l in s.stdout.splitlines() if l.split(',')[0].strip()==str(child.pid)]
      if v:gpu=max([gpu or 0]+v)
     except Exception:pass
     save(OUT/'WATCHDOG.json',dict(job=j['id'],seed=seed,dataset=j['effective']['dataset'],arm=j['effective']['frequency'],worker_pid=child.pid,updated=time.time()))
     time.sleep(10)
   save(d/'RESOURCE_PROCESS.json',dict(runtime_sec=time.time()-start,peak_rss_mb=peak/2**20,peak_process_gpu_mb=gpu,returncode=child.returncode))
   if child.returncode or not (d/'RESULT.json').exists():save(d/'FAILURE.json',dict(returncode=child.returncode,command=cmd))
   elif (d/'FAILURE.json').exists():(d/'FAILURE.json').unlink()
   report()
 report(True)
def main():
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['preflight','run','worker','report','verify']);a.add_argument('--job');a.add_argument('--seed',type=int);a.add_argument('--dataset');a.add_argument('--arm',choices=['control','unit','count','log']);args=a.parse_args()
 if args.mode=='preflight':preflight()
 elif args.mode in ('report','verify'):report(args.mode=='verify')
 elif args.mode=='run':run(args)
 else:
  j=next(j for j in read(ROOT/'PLAN.json')['jobs'] if j['id']==args.job);assert args.seed in j['seeds'];worker(j,args.seed)
if __name__=='__main__':main()
