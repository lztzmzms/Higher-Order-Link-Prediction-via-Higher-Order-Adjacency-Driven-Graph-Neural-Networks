"""Original native protocol files and fixed candidate masks."""
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def triples(path):
    return np.asarray([tuple(map(int,line.split())) for line in Path(path).read_text().splitlines() if line.strip()], dtype=np.int64)

def unpack(dataset):
    source = ROOT/'data'/dataset/'inputs.zip'
    signature = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = ROOT/'.cache'/'inputs'
    target = cache/dataset
    marker = target/'.complete'
    if marker.exists() and marker.read_text() == signature:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        temporary = Path(temporary)
        with zipfile.ZipFile(source) as archive:
            for item in archive.infolist():
                dest = (temporary/item.filename).resolve()
                if not dest.is_relative_to(temporary.resolve()):
                    raise ValueError('Input archive path escaped its destination')
                archive.extract(item, temporary)
        if target.exists():
            if not target.resolve().is_relative_to(cache.resolve()) or target.resolve()==cache.resolve():
                raise ValueError('Invalid cache destination')
            shutil.rmtree(target)
        target.mkdir()
        for item in temporary.iterdir():
            shutil.move(str(item), str(target/item.name))
        marker.write_text(signature)
    return target

def native_check(dataset):
    folder = unpack(dataset)
    proto = folder/'protocol'
    q = triples(proto/'dataset/test'/f'{dataset}_openTriangles_test.txt')
    positives = set(map(tuple, triples(proto/'dataset/test'/f'{dataset}_new_closures_test.txt')))
    labels = np.asarray([tuple(t) in positives for t in q], dtype=np.int64)
    with np.load(folder/'selected_sample.npz', allow_pickle=False) as selected:
        ids = selected['base_indices']
        np.testing.assert_array_equal(q[ids], selected['q'])
        np.testing.assert_array_equal(labels[ids], selected['y'])
    train_q = triples(proto/'dataset/train'/f'{dataset}_openTriangles_train.txt')
    train_pos = set(map(tuple, triples(proto/'dataset/train'/f'{dataset}_new_closures_train.txt')))
    with np.load(proto/'shared_splits'/dataset/'fit_val_indices.npz') as split:
        y = np.asarray([tuple(t) in train_pos for t in train_q], dtype=np.int64)
        np.testing.assert_array_equal(y, split['train_labels'])
        fit, val = split['fit_indices'], split['val_indices']
        np.testing.assert_array_equal(np.sort(np.r_[fit,val]), np.arange(len(train_q)))
        if np.intersect1d(fit,val).size or len(np.unique(y[fit]))!=2 or len(np.unique(y[val]))!=2:
            raise ValueError('Original shared fit/validation split changed')
    return folder

def native_rw_maps(dataset, args, device):
    folder = Path(args.data_root).parent
    maps = {}
    for split in ('train','test'):
        q = sorted({tuple(map(int,t)) for t in triples(folder/'protocol/dataset'/split/f'{dataset}_openTriangles_{split}.txt')})
        identity = dict(q=q,seed=args.rw_seed,dim=args.rw_dim,walks=args.rw_walks_per_node,epochs=args.rw_epochs)
        digest = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
        path = folder/'rw'/f'mls_{split}_{digest}.npz'
        with np.load(path,allow_pickle=False) as payload:
            values = payload['embedding']
            np.testing.assert_array_equal(payload['triangles'], q)
            if values.shape!=(len(q),args.rw_dim) or not np.isfinite(values).all():
                raise ValueError('Original separate-prefix RW input changed')
            maps[split] = {t:values[i] for i,t in enumerate(q)}
    return maps
