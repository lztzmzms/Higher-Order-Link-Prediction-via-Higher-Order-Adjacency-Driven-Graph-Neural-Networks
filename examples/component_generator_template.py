from historical_components.interfaces import ComponentGenerator, ComponentBatch

class MyGenerator(ComponentGenerator):
    def __init__(self, config=None):
        self.config = config or {}

    def generate(self, output_dir=None, context=None):
        if context is None:
            raise ValueError('A new-dataset prefix context is required')

        raise NotImplementedError('Provide a real component prediction recipe; no frequency fallback is supplied')
