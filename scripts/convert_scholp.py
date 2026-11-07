import argparse
import gzip
from pathlib import Path
import numpy as np

def integers(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return np.asarray([int(line.strip()) for line in f if line.strip()], dtype=np.int64)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--times', type=Path, required=True)
    p.add_argument('--nverts', type=Path, required=True)
    p.add_argument('--simplices', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Refusing to overwrite an existing event file')
    times, counts, vertices = (integers(path) for path in (a.times, a.nverts, a.simplices))
    if (len(times) != len(counts) or np.any(counts < 0) or
            counts.sum() != len(vertices) or np.any(times[1:] < times[:-1])):
        raise ValueError('Invalid offsets or nonchronological source events')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, times=times, counts=counts, vertices=vertices)

if __name__ == '__main__':
    main()
