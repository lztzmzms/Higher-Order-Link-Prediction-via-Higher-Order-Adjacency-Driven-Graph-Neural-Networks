import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/software_checks.json')
    args = parser.parse_args()
    checks = []
    completed = False
    def execute(name, arguments):
        start = time.monotonic()
        proc = subprocess.run([sys.executable, '-B', *arguments], cwd=ROOT, capture_output=True, text=True)
        checks.append(dict(name=name, passed=proc.returncode == 0,
            seconds=round(time.monotonic()-start,3),
            output=(proc.stdout+proc.stderr).replace(str(ROOT), '<release>').replace(str(Path(sys.executable)), 'python')))
        if proc.returncode:
            raise RuntimeError(checks[-1]['output'])
    try:
        execute('unit_and_real_cached_structure_training', ['-m','unittest','discover','-s','tests','-p','test_*.py','-v'])
        with tempfile.TemporaryDirectory(prefix='component-joint-check-', dir=ROOT) as directory:
            folder = Path(directory)
            assert folder.resolve().parent == ROOT.resolve()
            events = [dict(timestamp=i,nodes=list(e)) for i,e in enumerate(itertools.combinations(range(5),2))]
            events += [dict(timestamp=12,nodes=[0,1,2]), dict(timestamp=15,nodes=[3,4]), dict(timestamp=22,nodes=[0,1,3])]
            (folder/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events), encoding='utf-8')
            cfg = dict(K=2, input='events.jsonl', seed=41, observation_start=-1, train_cut=10,
                prediction_cut=20,prediction_end=30,future_labels_available=False,
                component_generator='tests.smoke_generator:SyntheticRule', history_windows=[],
                model=dict(hidden=32,depth=3,dropout=0.,gat_heads=4,gnn='GAT',lr=.001,
                    structure_lr_ratio=.1,weight_decay=0.,batch_size=3,epochs=2,loss_weight=1.,rw_enabled=True))
            config = folder/'config.json'
            config.write_text(json.dumps(cfg),encoding='utf-8')
            for mode in ('structure_only','component_joint'):
                execute('actual_new_event_cli_'+mode, ['run.py','--config',str(config),'--mode',mode,
                    '--model','both','--output',str(folder/mode)])
                for model in ('Dual-Mapping','MLSHoL'):
                    for suffix in ('.pt','.npz','_training.json'):
                        assert (folder/mode/(model+'_41'+suffix)).is_file()
        completed = True
    finally:

        report = dict(scope='bounded real execution of software contracts',
            passed=completed and bool(checks) and all(c['passed'] for c in checks), checks=checks,
            synthetic_cli_models=['GAT Dual-Mapping','SparseSIN MLSHoL'],
            benchmark_metrics_claimed=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,checks=len(checks))))

if __name__ == '__main__':
    main()
