"""Only adjacency frequency changes. Frozen history, features and RW stay fixed."""
from collections import Counter
from itertools import combinations
import json, math
import numpy as np
def value(c,arm):return 1. if arm=='unit' else float(c) if arm=='count' else math.log1p(c)
def configure(spec,arm,model):
 if arm=='control':return
 spec['strength_edge']=True
 if model=='Dual-Mapping':spec['strength_edge_mode']=arm
 else:spec['adjacency_mode']={'unit':'weighted-unit','count':'weighted-count','log':'weighted-log'}[arm]
def patch_native_dual(engine,arm):
 if arm=='control':return
 def weights(triplets,edges,pair_counts,scale=1.):
  out=[]
  for a,b in np.asarray(edges,dtype=np.int64):
   face=tuple(sorted(set(map(int,triplets[a]))&set(map(int,triplets[b]))))
   out.append(value(pair_counts.get(face,0),arm) if len(face)==2 else 1.)
  return np.asarray(out,dtype=np.float32)/(float(scale) if scale>1e-12 else 1.)
 engine._derived_edge_strength_weights=weights
def residual_assets(p,events,tasks,arm,model,d):
 if arm=='control':return
 import torch
 def counts(t):
  out=Counter()
  for _,_,vs in events[:t['cut']]:out.update(combinations(sorted(set(vs)),2))
  return out
 audit=[]
 if model=='Dual-Mapping':
  old=p.dual_assets
  def build(ev,ts):
   old(ev,ts)
   for name,t in ts.items():
    c=counts(t);q=t['q'];ei=t['dual']['edge_index'];w=[]
    for a,b in ei.T:
     face=tuple(sorted(set(map(int,q[a]))&set(map(int,q[b]))));assert len(face)==2
     w.append(value(c.get(face,0),arm))
    t['dual']['edge_weight']=np.array(w,dtype=np.float32);audit.append(dict(split=name,edges=len(w),minimum=min(w,default=0),maximum=max(w,default=0)))
   (d/'FREQUENCY_AUDIT.json').write_text(json.dumps(audit),encoding='utf-8')
  p.dual_assets=build
 else:
  old=p.sin_assets
  def build(ev,ts):
   old(ev,ts)
   for name,t in ts.items():
    c=counts(t);n=0
    for comp in t['complexes']:
     ch=comp.cochains[0];ix=ch.upper_index
     if ix is None:continue
     original=ch.original_node_ids.flatten().tolist()
     w=[value(c.get(tuple(sorted((int(original[a]),int(original[b])))),0),arm) for a,b in ix.T.tolist()]
     ch.upper_weight=torch.tensor(w,dtype=torch.float32);n+=len(w)
    audit.append(dict(split=name,messages=n))
   (d/'FREQUENCY_AUDIT.json').write_text(json.dumps(audit),encoding='utf-8')
  p.sin_assets=build
