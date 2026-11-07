def clock_features(prefix, queries, units_per_second, hours):
    pass
    last = prefix[-1][0]
    edges = defaultdict(list)
    nodes = defaultdict(list)
    groups = defaultdict(list)
    for timestamp, _, vs in prefix:
        for v in vs:
            nodes[v].append(timestamp)
        for edge in combinations(vs, 2):
            edges[edge].append(timestamp)
            if len(vs) >= 3:
                groups[edge].append(timestamp)
    pairs = [list(combinations(q, 2)) for q in queries]
    out = {'event_same_day_recent_edges': np.asarray([sum(((last - edges[e][-1]) / units_per_second < 86400 for e in es)) for es in pairs])}
    for label, entities, mapping in (('node', queries, nodes), ('edge', pairs, edges), ('group', pairs, groups)):
        requested = {v for q in entities for v in q}
        ages = {v: (last - np.asarray(mapping[v], dtype=np.float64)) / units_per_second for v in requested}
        for h in hours:
            decay = {v: np.exp(-age / (h * 3600)).sum() for v, age in ages.items()}
            recent = {v: float((age < h * 3600).sum()) for v, age in ages.items()}
            b.r.aggregate(np.asarray([[decay[v] for v in q] for q in entities]), f'clock_{label}_decay_{h:g}h', out)
            b.r.aggregate(np.asarray([[recent[v] for v in q] for q in entities]), f'clock_{label}_recent_{h:g}h', out)
        bins = {}
        for v in requested:
            phase = ((np.asarray(mapping[v], dtype=np.float64) - prefix[0][0]) / units_per_second % 86400 / 3600).astype(int)
            counts = np.bincount(phase, minlength=24).astype(float)
            bins[v] = counts / max(1.0, counts.sum())
        phase = np.asarray([[bins[v] for v in q] for q in entities])
        out[f'clock_{label}_daily_phase_triple_overlap'] = phase.min(axis=1).sum(axis=1)
    return out

def community_features(prefix, queries):
    pass
    counts, group_counts = (Counter(), Counter())
    adj = defaultdict(set)
    for _, _, vs in prefix:
        for u, v in combinations(vs, 2):
            counts[u, v] += 1
            adj[u].add(v)
            adj[v].add(u)
            if len(vs) >= 3:
                group_counts[u, v] += 1
    graph = nx.Graph()
    graph.add_nodes_from(sorted(adj))
    graph.add_weighted_edges_from(((u, v, np.log1p(w)) for (u, v), w in sorted(counts.items())))
    communities = nx.community.louvain_communities(graph, seed=41, resolution=1.0)
    membership = {v: i for i, c in enumerate(communities) for v in c}
    sizes = {v: len(c) for c in communities for v in c}
    same, witness, group_witness = ([], [], [])
    for q in queries:
        same.append(sum((membership[u] == membership[v] for u, v in combinations(q, 2))))
        shared = set.intersection(*(adj[v] for v in q)) - set(q)
        witness.append(len(shared))
        group_witness.append(sum((min((group_counts[tuple(sorted((v, w)))] for v in q)) for w in shared)))
    out = {'community_same_pairs': np.asarray(same), 'community_common_witness': np.asarray(witness), 'community_group_witness_mass': np.asarray(group_witness)}
    b.r.aggregate(np.asarray([[sizes[v] for v in q] for q in queries]), 'community_size', out)
    return out

def feature_indices(names, cfg):
    if cfg['view'] == 'base':
        return [i for i, n in enumerate(names) if n.startswith('event_')]
    if cfg['view'] == 'temporal':
        return [i for i, n in enumerate(names) if n.startswith(('event_decay_', 'event_age_', 'continue_decay_', 'continue_edge_sync_', 'clock_', 'community_'))]
    return list(range(len(names)))

def training(target, cfg):
    allowed = [v for f, v in DATA[CURRENT]['snaps'].items() if f >= 0.2 and v['label_end'] is not None and (int(v['label_end']) <= int(target['cut']))]
    if cfg.get('origin_policy') == 'nonoverlap':
        allowed = [v for v in allowed if v['fraction'] in (0.2, 0.4, 0.6)]
    if not allowed:
        raise ValueError('No completed historical horizon')
    names = target['names'].tolist()
    indices = feature_indices(names, cfg)
    x = np.concatenate([v['x'][:, indices] for v in allowed])
    xv = target['x'][:, indices].copy()
    if cfg['units'] == 'legacy':
        j = names.index('event_same_day_recent_edges')
        if j in indices:
            x[:, indices.index(j)] = np.concatenate([v['old_x'][:, j] for v in allowed])
            xv[:, indices.index(j)] = target['old_x'][:, j]
    y = np.concatenate([v['y'] for v in allowed])
    if len(np.unique(y)) < 2:
        raise ValueError('Historical training needs both classes')
    q = np.concatenate([v['q'] for v in allowed])
    origins = np.concatenate([np.full(len(v['y']), i) for i, v in enumerate(allowed)])
    ends = np.concatenate([np.full(len(v['y']), int(v['label_end'])) for v in allowed])
    weights = np.exp(cfg['recent_weight'] * (ends / int(target['cut']) - 1.0))
    if cfg.get('duration_match'):
        differences = np.concatenate([np.full(len(v['y']), np.log(max(1.0, v['completed_label_duration_seconds']) / target['duration_proxy_seconds'])) for v in allowed])
        weights *= np.exp(-0.5 * differences ** 2)
    if cfg['deduplicate']:
        _, inv, counts = np.unique(np.column_stack([q, y]), axis=0, return_inverse=True, return_counts=True)
        weights /= counts[inv]
    weights /= weights.mean()
    if cfg.get('balanced'):
        prevalence = np.average(y, weights=weights)
        weights *= np.where(y == 1, 0.5 / prevalence, 0.5 / (1 - prevalence))
    return (x, y, weights, origins, xv, {'max_training_label_end': int(ends.max()), 'prediction_cut': int(target['cut']), 'train_rows': len(y), 'train_positives': int(y.sum()), 'distinct_positive_triangles': len(np.unique(q[y == 1], axis=0)), 'feature_count': len(indices), 'units': cfg['units'], 'overlap_weights': cfg['deduplicate'], 'duration_match': cfg.get('duration_match', False), 'origin_policy': cfg.get('origin_policy', 'dense')})

def pair_indices(y, origins, hardness, sampler, seed):
    rng = np.random.default_rng(seed)
    left, right = ([], [])
    for origin in np.unique(origins):
        positive = np.flatnonzero((origins == origin) & (y == 1))
        negative = np.flatnonzero((origins == origin) & (y == 0))
        if not len(positive) or not len(negative):
            continue
        for p in positive:
            random = rng.choice(negative, 32, replace=True)
            if sampler == 'hard':
                hard = negative[np.argsort(hardness[negative], kind='stable')[-max(1, len(negative) // 4):]]
                random[16:] = rng.choice(hard, 16, replace=True)
            left.extend([p] * len(random))
            right.extend(random.tolist())
    if not left:
        raise ValueError('No within-origin ranking pairs')
    ids = rng.choice(len(left), min(12000, len(left)), replace=False)
    return (np.asarray(left)[ids], np.asarray(right)[ids])

def robust_quality(lifts):
    logs = np.log(np.maximum(np.asarray(lifts), 1e-12))
    return float(np.median(logs) + 0.5 * np.min(logs))

def stack_predict(target, cfg, seeds):
    recipes = [{**cfg, 'family': family} for family in ('ET', 'HG', 'PAIR_LR')]
    eligible = [v for f, v in DATA[CURRENT]['snaps'].items() if f in (0.4, 0.45, 0.5, 0.55, 0.6) and v['label_end'] <= int(target['cut'])]
    oof = []
    for val in eligible:
        scores = np.column_stack([rankdata(predict(val, recipe, seeds)[0]) / len(val['q']) for recipe in recipes])
        oof.append((scores, val['y']))
    grid = [np.asarray(v) / 4 for v in product(range(5), repeat=3) if sum(v) == 4]
    weights = np.ones(3) / 3
    if oof:
        weights = max(grid, key=lambda w: robust_quality([b.r.score_metrics(y, scores @ w)['Lift'] for scores, y in oof]))
    outputs = [predict(target, recipe, seeds) for recipe in recipes]
    score = np.column_stack([rankdata(p) / len(p) for p, _ in outputs]) @ weights
    return (score, {**outputs[0][1], 'oof_windows': len(oof), 'oof_weights': weights.tolist(), 'oof_max_label_end': max((int(v['label_end']) for v in eligible), default=0)})

def predict(target, cfg, seeds=(41,)):
    if cfg['family'] == 'OOF_STACK':
        return stack_predict(target, cfg, seeds)
    x, y, weights, origins, xv, gate = training(target, cfg)
    if cfg.get('input_rank'):
        sorted_x = np.sort(x, axis=0)
        x, xv = [np.column_stack([np.searchsorted(sorted_x[:, j], matrix[:, j], side='right') / len(sorted_x) for j in range(matrix.shape[1])]).astype(np.float32) for matrix in (x, xv)]
    if cfg['family'] in ('PAIR_LR', 'PAIR_MLP', 'PAIR_AP'):
        ids = b.r.top_features(x, y, [str(i) for i in range(x.shape[1])], min(80, x.shape[1]))
        scaler = StandardScaler().fit(x[:, ids])
        xt = np.clip(scaler.transform(x[:, ids]), -8, 8).astype(np.float32)
        xtest = np.clip(scaler.transform(xv[:, ids]), -8, 8).astype(np.float32)
    values = []
    names = target['names'].tolist()
    hardness = np.concatenate([v['x'][:, names.index('event_edge_count_harmonic')] for f, v in DATA[CURRENT]['snaps'].items() if f >= 0.2 and v['label_end'] is not None and (int(v['label_end']) <= int(target['cut'])) and (cfg.get('origin_policy') != 'nonoverlap' or f in (0.2, 0.4, 0.6))])
    for seed in seeds:
        if cfg['family'].startswith('PAIR_'):
            pos, neg = pair_indices(y, origins, hardness, cfg['sampler'], seed)
            pw = weights[pos]
            if cfg['family'] == 'PAIR_LR':
                diff = xt[pos] - xt[neg]
                model = LogisticRegression(C=cfg['C'], fit_intercept=False, max_iter=400, random_state=seed)
                model.fit(np.r_[diff, -diff], np.r_[np.ones(len(diff)), np.zeros(len(diff))], sample_weight=np.r_[pw, pw])
                prediction = model.decision_function(xtest)
            else:
                import torch
                torch.set_num_threads(1)
                torch.manual_seed(seed)
                torch.use_deterministic_algorithms(True)
                model = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
                optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001 / cfg['C'])
                tx, tw = (torch.from_numpy(xt), torch.from_numpy(pw.astype(np.float32)))
                pp, nn = (torch.from_numpy(pos), torch.from_numpy(neg))
                rng = np.random.default_rng(seed)
                for _ in range(cfg['epochs']):
                    if cfg['family'] == 'PAIR_AP':
                        for origin in rng.permutation(np.unique(origins)):
                            positive = np.flatnonzero((origins == origin) & (y == 1))
                            negative = np.flatnonzero((origins == origin) & (y == 0))
                            if not len(positive) or not len(negative):
                                continue
                            negative_sample = rng.choice(negative, min(768, len(negative)), replace=False)
                            ps = model(tx[positive]).ravel()
                            ns = model(tx[negative_sample]).ravel()
                            rank_pos = 1.0 + (torch.sigmoid((ps[None, :] - ps[:, None]) / 0.5).sum(1) - 0.5)
                            rank_neg = torch.sigmoid((ns[None, :] - ps[:, None]) / 0.5).sum(1) * (len(negative) / len(negative_sample))
                            pw_origin = torch.from_numpy(weights[positive].astype(np.float32))
                            loss = 1.0 - (rank_pos / (rank_pos + rank_neg) * pw_origin).sum() / pw_origin.sum()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    else:
                        order = rng.permutation(len(pos))
                        for start in range(0, len(order), 1024):
                            ix = torch.from_numpy(order[start:start + 1024])
                            loss = (torch.nn.functional.softplus(-(model(tx[pp[ix]]) - model(tx[nn[ix]])).ravel()) * tw[ix]).mean()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                with torch.no_grad():
                    prediction = model(torch.from_numpy(xtest)).ravel().numpy()
            gate['ranking_pairs_per_seed'] = len(pos) if cfg['family'] != 'PAIR_AP' else 0
            if cfg['family'] == 'PAIR_AP':
                gate['objective'] = 'historical within-origin differentiable average precision'
        else:
            chosen = []
            rng = np.random.default_rng(seed)
            for origin in np.unique(origins):
                pos = np.flatnonzero((origins == origin) & (y == 1))
                neg = np.flatnonzero((origins == origin) & (y == 0))
                budget = min(len(neg), max(256, 32 * len(pos)))
                if cfg['sampler'] == 'full':
                    sampled = neg
                elif cfg['sampler'] == 'hard':
                    hard = neg[np.argsort(hardness[neg], kind='stable')[-budget // 2:]]
                    rest = np.setdiff1d(neg, hard)
                    sampled = np.r_[hard, rng.choice(rest, budget - len(hard), replace=False)]
                else:
                    sampled = rng.choice(neg, budget, replace=False)
                chosen.extend(np.r_[pos, sampled].tolist())
            chosen = np.asarray(chosen)
            if cfg['family'] in ('ET', 'ET_FULL'):
                model = ExtraTreesClassifier(n_estimators=100 if cfg['family'] == 'ET_FULL' else 80, max_depth=8, max_features=0.3, min_samples_leaf=cfg['leaf'], n_jobs=1, random_state=seed)
            else:
                full = cfg['family'] == 'HG_FULL'
                model = HistGradientBoostingClassifier(max_iter=150 if full else 120, learning_rate=0.02 if full else 0.04, max_leaf_nodes=43 if full else 23, min_samples_leaf=cfg['leaf'], l2_regularization=0.0 if full else 1 / cfg['C'], random_state=seed)
            model.fit(x[chosen], y[chosen], sample_weight=weights[chosen])
            prediction = model.predict_proba(xv)[:, 1]
            gate['sampled_train_rows'] = len(chosen)
        values.append(prediction if cfg.get('probability_ensemble') else rankdata(prediction) / len(prediction))
    score = np.mean(values, axis=0)
    if cfg['fusion']:
        group = target['x'][:, names.index('continue_edge_sync_triple_group_64')]
        score = (1 - cfg['fusion']) * score + cfg['fusion'] * rankdata(group) / len(group)
    assert np.all(np.isfinite(score))
    return (score, gate)
