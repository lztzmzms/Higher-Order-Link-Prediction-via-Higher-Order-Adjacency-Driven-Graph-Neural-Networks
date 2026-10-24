"""
Based on https://github.com/rusty1s/pytorch_geometric/blob/76d61eaa9fc8702aa25f29dfaa5134a169d0f1f6/torch_geometric/nn/conv/utils/inspector.py

MIT License

Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
Copyright (c) 2021 The CWN Project Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import inspect
from collections import OrderedDict
from typing import Dict, Any, Callable
try:
    from torch_geometric.nn.conv.utils.inspector import Inspector
except ModuleNotFoundError:
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
