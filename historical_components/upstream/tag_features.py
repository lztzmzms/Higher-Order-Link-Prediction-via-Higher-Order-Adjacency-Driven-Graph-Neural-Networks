class History:
    pass

    def __init__(self, prefix):
        self.adj = defaultdict(set)
        self.closed = set()
        self.node = Counter()
        self.edge = Counter()
        self.group = Counter()
        self.nc = Counter()
        self.ec = Counter()
        self.last = {}
        self.decay = {t: Counter() for t in (0.01, 0.05, 0.2, 1.0)}
        self.start = prefix[0][0]
        self.end = prefix[-1][0]
        self.span = max(1, self.end - self.start)
        for timestamp, iterator in groupby(prefix, key=lambda v: v[0]):
            block = list(iterator)
            new = {q for _, _, vs in block for q in combinations(vs, 3)} - self.closed
            eligible = [q for q in new if all((b in self.adj[a] for a, b in combinations(q, 2)))]
            for q in eligible:
                self.nc.update(q)
                self.ec.update(combinations(q, 2))
            for _, _, vs in block:
                self.node.update(vs)
                edges = list(combinations(vs, 2))
                self.edge.update(edges)
                if len(vs) >= 3:
                    self.group.update(edges)
                for edge in edges:
                    x, y = edge
                    self.adj[x].add(y)
                    self.adj[y].add(x)
                    self.last[edge] = timestamp
                    for tau, count in self.decay.items():
                        count[edge] += math.exp(-(self.end - timestamp) / (tau * self.span))
            self.closed.update(new)

    def candidates(self):
        result = []
        for x in sorted(self.adj):
            for y in sorted((v for v in self.adj[x] if v > x)):
                for z in sorted((v for v in self.adj[x] & self.adj[y] if v > y)):
                    q = (x, y, z)
                    if q not in self.closed:
                        result.append(q)
        return np.asarray(result, dtype=np.int64).reshape(-1, 3)

    def proxies(self, q):
        activity, recent = ([], [])
        for tri in q:
            edges = list(combinations(map(int, tri), 2))
            activity.append(sum((math.log1p(self.edge[e]) for e in edges)))
            recent.append(sum((self.decay[0.05][e] for e in edges)))
        return (np.asarray(activity), np.asarray(recent))

    def features(self, queries):
        output = defaultdict(list)
        for tri in queries:
            tri = tuple(map(int, tri))
            edges = list(combinations(tri, 2))
            groups = {'edge_count': [self.edge[e] for e in edges], 'edge_group': [self.group[e] for e in edges], 'node_count': [self.node[v] for v in tri], 'node_degree': [len(self.adj[v]) for v in tri], 'node_closure': [self.nc[v] for v in tri], 'edge_closure': [self.ec[e] for e in edges], 'edge_age': [(self.end - self.last[e]) / self.span for e in edges], 'edge_common': [len(self.adj[x] & self.adj[y]) for x, y in edges], 'edge_jaccard': [len(self.adj[x] & self.adj[y]) / max(1, len(self.adj[x] | self.adj[y])) for x, y in edges], 'edge_coactivity': [self.edge[x, y] / math.sqrt(max(1, self.node[x] * self.node[y])) for x, y in edges]}
            for tau, counts in self.decay.items():
                groups[f'decay_{tau}'] = [counts[e] for e in edges]
            for name, values in groups.items():
                v = np.asarray(values, dtype=float)
                for suffix, val in [('min', v.min()), ('mean', v.mean()), ('max', v.max()), ('std', v.std()), ('logsum', np.log1p(v).sum()), ('harmonic', 3 / np.sum(1 / (v + 1e-12)))]:
                    output[name + '_' + suffix].append(val)
            output['common_three'].append(len(set.intersection(*(self.adj[v] for v in tri))))
        names = sorted(output)
        x = np.column_stack([output[n] for n in names])
        return (np.log1p(np.maximum(x, 0)).astype(np.float32), names)
