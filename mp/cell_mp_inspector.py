import inspect
from collections import OrderedDict
from typing import Dict, Any, Callable
from torch_geometric.inspector import Inspector

class CellularInspector(Inspector):
    pass

    def __init__(self, cls):
        super().__init__(cls)
        self.cls = cls
        self.params = {}

    def __implements__(self, cls, func_name: str) -> bool:
        if cls.__name__ == 'CochainMessagePassing':
            return False
        if func_name in cls.__dict__.keys():
            return True
        return any(self.__implements__(c, func_name) for c in cls.__bases__)

    def implements(self, func_name: str) -> bool:
        return self.__implements__(self.cls.__class__, func_name)

    def inspect(self, func: Callable, pop_first_n: int = 0) -> Dict[str, Any]:
        params = inspect.signature(func).parameters
        params = OrderedDict(params)
        for _ in range(pop_first_n):
            params.popitem(last=False)
        self.params[func.__name__] = params

    def keys(self, func_names):
        out = set()
        for name in func_names:
            out.update(self.params.get(name, {}).keys())
        return out

    def distribute(self, func_name, coll_dict):
        return {
            key: coll_dict[key]
            for key in self.params.get(func_name, {})
            if key in coll_dict
        }
