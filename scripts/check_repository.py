import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NATIVE = {'NDC-substances', 'threads-math-sx'}
COMPONENTS = {'email-Enron': 7, 'email-Eu': 5, 'contact-high-school': 4,
              'contact-primary-school': 3, 'tags-ask-ubuntu': 5, 'tags-math-sx': 2}
LOCAL = {'data', 'models', 'mp', 'pipelines', 'simplicial_complex',
         'historical_components', 'preprocessing', 'train', 'evaluate', 'run', 'temporal_pipeline'}

def main():
    errors, imports = [], 0
    sources = sorted(ROOT.rglob('*.py'))
    for path in sources:
        source = path.read_text(encoding='utf-8-sig')
        try:
            tree = ast.parse(source, filename=str(path.relative_to(ROOT)))
        except SyntaxError as exc:
            errors.append(str(exc))
            continue
        if re.search(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/](?!/)|/(?:home|Users|mnt)/', source):
            errors.append(f'Absolute private path: {path.relative_to(ROOT)}')
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
                if node.level:
                    base = path.parent
                    for _ in range(node.level-1):
                        base = base.parent
                    target = base.joinpath(*node.module.split('.'))
                    if not target.is_dir() and not target.with_suffix('.py').is_file():
                        errors.append(f'Missing relative import: {path.relative_to(ROOT)}: {node.module}')
                    imports += 1
                    continue
            else:
                continue
            for module in modules:
                if module.split('.')[0] in LOCAL:
                    target = ROOT.joinpath(*module.split('.'))
                    imports += 1
                    if not target.is_dir() and not target.with_suffix('.py').is_file():
                        errors.append(f'Missing local module: {module}')
    jsons = sorted(ROOT.rglob('*.json'))
    for path in jsons:
        json.loads(path.read_text(encoding='utf-8-sig'))
    for dataset in NATIVE | COMPONENTS.keys():
        cfg = json.loads((ROOT/'configs'/f'{dataset}.json').read_text())
        assert set(cfg['models']) == {'Dual-Mapping', 'MLSHoL'}
        if dataset in NATIVE:
            assert cfg['pipeline'] == 'historynet_joint' and cfg['use_history']
            assert cfg['component_joint']['dimensions'] == 2
            assert cfg['component_generator'] == dataset
            assert cfg['models']['Dual-Mapping']['gnn'] == 'GAT'
            assert {(m['epochs'], m['batch_size']) for m in cfg['models'].values()} == {(25, 32)}
            assert cfg['component_generator_config'] == dict(ensemble_seeds=[41,42], epochs=20, width=128, weight_power=.5)
        else:
            assert cfg['pipeline'] == 'component_joint'
            assert cfg['component_generator'] == dataset
            assert cfg['component_joint']['dimensions'] == COMPONENTS[dataset]
            assert cfg['models']['Dual-Mapping']['gnn'] == 'GAT'
            assert cfg['component_generator_config']['recipe_policy'] == 'frozen_fit'
            assert {(m['epochs'], m['batch_size']) for m in cfg['models'].values()} == {(25, 32)}
    print(json.dumps(dict(passed=not errors, scope='static source/configuration inspection',
        python_sources=len(sources), json_files=len(jsons), local_imports=imports, errors=errors), indent=2))
    if errors:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
