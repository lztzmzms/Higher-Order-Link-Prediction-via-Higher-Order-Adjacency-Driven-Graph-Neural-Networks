                                                                                         
import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def ensure_scatter():
    try:
        module=importlib.import_module('torch_scatter')
        return {'backend':'installed_torch_scatter','module':getattr(module,'__file__',None)}
    except ModuleNotFoundError as exc:
        if exc.name!='torch_scatter':raise
                                                                                
    path=ROOT/'code/current/torch_scatter/__init__.py'
    if not path.is_file():raise RuntimeError('Packaged torch_scatter compatibility file missing: '+str(path))
    spec=importlib.util.spec_from_file_location('torch_scatter',path,submodule_search_locations=[str(path.parent)])
    module=importlib.util.module_from_spec(spec);sys.modules['torch_scatter']=module
    try:spec.loader.exec_module(module)
    except BaseException:sys.modules.pop('torch_scatter',None);raise
    return {'backend':'packaged_torch_native_compatibility','module':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'uses_no_labels':True,'sum_mean_backend':'torch.index_add','csr_backend_not_used_by_dense_edge_index_protocol':True}
