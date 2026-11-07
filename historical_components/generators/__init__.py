from importlib import import_module

from historical_components.interfaces import ComponentBatch, CandidateContext, ComponentGenerator

def create_generator(name, config=None):
    from .builtin import DATASETS, DatasetSpecificComponentGenerator
    if name in DATASETS:
        return DatasetSpecificComponentGenerator(name, config)
    if name in ('NDC-substances', 'threads-math-sx'):
        from pipelines.historynet_joint import HistoryNetTransferGenerator
        return HistoryNetTransferGenerator(name, config)
    if ':' not in str(name):
        raise ValueError(f'No original component recipe for {name!r}; configure a module:ComponentGeneratorClass explicitly.')
    module, class_name = str(name).split(':', 1)
    cls = getattr(import_module(module), class_name)
    generator = cls(config=config or {})
    if not isinstance(generator, ComponentGenerator):
        raise TypeError('Custom component generators must implement ComponentGenerator')
    return generator

__all__ = ['create_generator', 'ComponentBatch', 'CandidateContext', 'ComponentGenerator']
