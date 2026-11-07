class History:

    def __init__(self):
        self.adj = defaultdict(set)
        self.closed = set()
        self.nodes = Counter()
        self.edges = Counter()
        self.node_groups = Counter()
        self.edge_groups = Counter()
        self.node_closures = Counter()
        self.edge_closures = Counter()
        self.node_times = defaultdict(list)
        self.edge_times = defaultdict(list)
        self.group_times = defaultdict(list)
        self.first = None
        self.last = None
        self.rows = 0

    def update(self, block, eligible):
        for timestamp, _, vs in block:
            if self.first is None:
                self.first = timestamp
            self.last = timestamp
            self.rows += 1
            self.nodes.update(vs)
            for v in vs:
                self.node_times[v].append(timestamp)
            for a, b in combinations(vs, 2):
                edge = (a, b)
                self.edges[edge] += 1
                self.edge_times[edge].append(timestamp)
                self.adj[a].add(b)
                self.adj[b].add(a)
                if len(vs) >= 3:
                    self.edge_groups[edge] += 1
                    self.group_times[edge].append(timestamp)
            if len(vs) >= 3:
                self.node_groups.update(vs)
            self.closed.update(combinations(vs, 3))
        for tri in eligible:
            self.node_closures.update(tri)
            self.edge_closures.update(combinations(tri, 2))

    def open_queries(self):
        return [(a, b, c) for a in sorted(self.adj) for b in sorted((v for v in self.adj[a] if v > a)) for c in sorted((v for v in self.adj[a] & self.adj[b] if v > b)) if (a, b, c) not in self.closed]

    def features(self, queries):
        out = {}
        span = max(1, self.last - self.first)
        pairs = [tuple(combinations(q, 2)) for q in queries]
        for label, qs, values in (('node_count', queries, self.nodes), ('edge_count', pairs, self.edges), ('node_group', queries, self.node_groups), ('edge_group', pairs, self.edge_groups), ('node_closures', queries, self.node_closures), ('edge_closures', pairs, self.edge_closures), ('node_degree', queries, {v: len(a) for v, a in self.adj.items()})):
            a = np.asarray([[values.get(v, 0) for v in q] for q in qs], dtype=float)
            s.r.aggregate(a, 'event_' + label, out)
            s.r.aggregate(a / max(1, self.rows), 'event_rate_' + label, out)
        for label, qs, counts, groups, closures in (('node', queries, self.nodes, self.node_groups, self.node_closures), ('edge', pairs, self.edges, self.edge_groups, self.edge_closures)):
            for suffix, source in (('group_fraction', groups), ('closure_fraction', closures)):
                a = np.asarray([[source[v] / max(1, counts[v]) for v in q] for q in qs])
                s.r.aggregate(a, f'event_{label}_{suffix}', out)
        for label, qs, mapping in (('node', queries, self.node_times), ('edge', pairs, self.edge_times), ('edge_group', pairs, self.group_times)):
            requested = {v for q in qs for v in q}
            ages = {v: np.asarray([(self.last - t) / span for t in mapping[v]], dtype=float) for v in requested}
            for tau in (0.005, 0.02, 0.05, 0.2, 1.0):
                weights = {v: float(np.exp(-a / tau).sum()) for v, a in ages.items()}
                s.r.aggregate(np.asarray([[weights[v] for v in q] for q in qs]), f'event_decay_{label}_{tau}', out)
            age = {v: a[-1] if len(a) else 1.0 for v, a in ages.items()}
            tenure = {v: a[0] if len(a) else 0.0 for v, a in ages.items()}
            s.r.aggregate(np.asarray([[age[v] for v in q] for q in qs]), f'event_age_{label}', out)
            s.r.aggregate(np.asarray([[tenure[v] for v in q] for q in qs]), f'event_tenure_{label}', out)
        common = []
        jaccard = []
        coactivity = []
        for es in pairs:
            common.append([len(self.adj[a] & self.adj[b]) for a, b in es])
            jaccard.append([len(self.adj[a] & self.adj[b]) / max(1, len(self.adj[a] | self.adj[b])) for a, b in es])
            coactivity.append([self.edges[a, b] / math.sqrt(max(1, self.nodes[a] * self.nodes[b])) for a, b in es])
        for label, values in (('common', common), ('jaccard', jaccard), ('coactivity', coactivity)):
            s.r.aggregate(np.asarray(values), f'event_{label}', out)
        out['event_same_day_recent_edges'] = np.asarray([sum((self.last - self.edge_times[edge][-1] < 86400 for edge in es)) for es in pairs], dtype=float)
        return out

def prefix_state(prefix):
    state = History()
    for timestamp, iterator in groupby(prefix, key=lambda row: row[0]):
        block = list(iterator)
        new = {q for _, _, vs in block for q in combinations(vs, 3)} - state.closed
        eligible = [q for q in new if all((b in state.adj.get(a, set()) for a, b in combinations(q, 2)))]
        state.update(block, eligible)
    return state
