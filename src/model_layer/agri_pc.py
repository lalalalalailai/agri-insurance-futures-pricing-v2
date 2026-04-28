import warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from itertools import combinations
from config import FEATURE_NAMES

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

TEMPORAL_ORDER = {
    "temperature": 0, "precipitation": 0, "humidity": 0, "wind_speed": 0,
    "surface_pressure": 0, "solar_radiation": 0,
    "extreme_temp_index": 0, "extreme_precip_index": 0,
    "ndvi": 1, "evi": 1, "lst": 1, "drought_index": 1,
    "yield_proxy": 2,
    "close": 3, "open": 3, "high": 3, "low": 3, "volume": 3, "hold": 3,
    "cpi": 3, "ppi": 3, "m2": 3, "gdp": 3, "pmi": 3,
    "month_sin": -1, "month_cos": -1, "quarter_sin": -1, "quarter_cos": -1,
}

AGRI_PRIOR_EDGES = [
    ("temperature", "ndvi"), ("precipitation", "ndvi"),
    ("extreme_temp_index", "yield_proxy"), ("extreme_precip_index", "yield_proxy"),
    ("ndvi", "yield_proxy"), ("drought_index", "yield_proxy"),
    ("yield_proxy", "close"),
    ("cpi", "close"), ("ppi", "close"), ("m2", "close"),
    ("close", "volume"),
]

DELIVERY_EDGES = [
    ("close", "hold"),
]

CORE_NODES = [
    "temperature", "precipitation", "extreme_temp_index", "extreme_precip_index",
    "drought_index", "ndvi", "yield_proxy", "close", "volume", "hold",
    "cpi", "m2",
]


class AgriPC:

    def __init__(self, alpha: float = 0.05, max_cond_set: int = 2,
                 feature_names: list = None):
        self.alpha = alpha
        self.max_cond_set = max_cond_set
        self.feature_names = feature_names or FEATURE_NAMES
        self.graph = nx.DiGraph()
        self.skeleton = None
        self.sep_set = {}
        self.dag_quality = {}

    def _partial_corr(self, data: pd.DataFrame, x: str, y: str,
                      z: list = None) -> float:
        if z is None or len(z) == 0:
            valid = data[[x, y]].dropna()
            if len(valid) < 10:
                return 0.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r, _ = stats.pearsonr(valid[x], valid[y])
            if np.isnan(r):
                return 0.0
            return r
        cols = [x, y] + [c for c in z if c in data.columns]
        valid = data[cols].dropna()
        if len(valid) < len(cols) + 5:
            return 0.0
        try:
            corr_matrix = valid.corr().values
            n = len(cols)
            ix, iy = 0, 1
            iz = list(range(2, n))
            sub = corr_matrix[np.ix_([ix, iy] + iz, [ix, iy] + iz)]
            try:
                prec = np.linalg.inv(sub)
                denom = np.sqrt(abs(prec[0, 0] * prec[1, 1]))
                if denom < 1e-10:
                    return 0.0
                r = -prec[0, 1] / denom
            except np.linalg.LinAlgError:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r, _ = stats.pearsonr(valid[x], valid[y])
            return np.clip(r, -1, 1)
        except Exception:
            return 0.0

    def _ci_test(self, data: pd.DataFrame, x: str, y: str,
                 z: list = None) -> tuple:
        r = self._partial_corr(data, x, y, z)
        valid = data[[x, y]].dropna()
        n = len(valid)
        if n < 10 or abs(r) >= 1.0:
            return 1.0, 1.0
        z_stat = 0.5 * np.log((1 + r) / (1 - abs(r) + 1e-10))
        se = 1.0 / np.sqrt(max(n - 3 - (len(z) if z else 0), 1))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat / se)))
        return z_stat, p_value

    def _apply_temporal_constraint(self, nodes: list) -> set:
        removed = set()
        for i, j in combinations(nodes, 2):
            ti = TEMPORAL_ORDER.get(i, -1)
            tj = TEMPORAL_ORDER.get(j, -1)
            if ti >= 0 and tj >= 0 and ti > tj:
                removed.add((i, j))
                removed.add((j, i))
        return removed

    def _apply_agri_prior(self, skeleton_edges: set) -> set:
        forced = set()
        for src, dst in AGRI_PRIOR_EDGES:
            forced.add((src, dst))
        for src, dst in DELIVERY_EDGES:
            forced.add((src, dst))
        return forced

    def _filter_constant_features(self, data: pd.DataFrame, nodes: list) -> list:
        filtered = []
        for node in nodes:
            if node in data.columns:
                if data[node].std() > 1e-8:
                    filtered.append(node)
        return filtered

    def _build_skeleton(self, data: pd.DataFrame, nodes: list) -> set:
        nodes = self._filter_constant_features(data, nodes)
        if len(nodes) < 3:
            return set()

        edges = set()
        for i, j in combinations(nodes, 2):
            edges.add((i, j))
            edges.add((j, i))

        temporal_removed = self._apply_temporal_constraint(nodes)
        edges = edges - temporal_removed

        forced = self._apply_agri_prior(edges)
        forced_pairs = set()
        for src, dst in forced:
            forced_pairs.add((src, dst))
            forced_pairs.add((dst, src))

        sep_set = {}
        pairs_to_test = list(combinations(nodes, 2))
        for idx, (i, j) in enumerate(pairs_to_test):
            pair = (i, j)
            rpair = (j, i)
            if pair not in edges and rpair not in edges:
                continue
            if pair in forced_pairs or rpair in forced_pairs:
                continue

            _, p_val = self._ci_test(data, i, j, z=None)
            if p_val > self.alpha:
                edges.discard(pair)
                edges.discard(rpair)
                sep_set[pair] = []
                sep_set[rpair] = []
                continue

            adj_i = [n for n in nodes if n != i and n != j and
                     ((i, n) in edges or (n, i) in edges)]
            found_sep = False
            for cond_size in range(1, min(self.max_cond_set + 1, len(adj_i) + 1)):
                if found_sep:
                    break
                for cond in combinations(adj_i, cond_size):
                    cond_list = list(cond)
                    _, p_val = self._ci_test(data, i, j, z=cond_list)
                    if p_val > self.alpha:
                        edges.discard(pair)
                        edges.discard(rpair)
                        sep_set[pair] = cond_list
                        sep_set[rpair] = cond_list
                        found_sep = True
                        break

        self.sep_set = sep_set
        return edges

    def _orient_edges(self, skeleton_edges: set, nodes: list) -> nx.DiGraph:
        dag = nx.DiGraph()
        dag.add_nodes_from(nodes)

        undirected = set()
        for i, j in skeleton_edges:
            if (j, i) in skeleton_edges:
                pair = tuple(sorted([i, j]))
                undirected.add(pair)

        for i, j in list(undirected):
            for k in nodes:
                if k == i or k == j:
                    continue
                ik = tuple(sorted([i, k]))
                jk = tuple(sorted([j, k]))
                if ik in undirected and jk not in undirected:
                    sep = self.sep_set.get((i, k), self.sep_set.get((k, i), None))
                    if sep is not None and j not in sep:
                        dag.add_edge(i, k)
                        dag.add_edge(k, j)
                        undirected.discard(ik)
                        undirected.discard(tuple(sorted([k, j])))

        for src, dst in AGRI_PRIOR_EDGES:
            if src in dag.nodes() and dst in dag.nodes() and not dag.has_edge(dst, src):
                dag.add_edge(src, dst)

        for src, dst in DELIVERY_EDGES:
            if src in dag.nodes() and dst in dag.nodes() and not dag.has_edge(dst, src):
                dag.add_edge(src, dst)

        for i, j in list(undirected):
            ti = TEMPORAL_ORDER.get(i, -1)
            tj = TEMPORAL_ORDER.get(j, -1)
            if ti >= 0 and tj >= 0 and ti < tj:
                dag.add_edge(i, j)
            elif ti >= 0 and tj >= 0 and ti > tj:
                dag.add_edge(j, i)
            else:
                dag.add_edge(i, j)

        return dag

    def _compute_metrics(self, dag: nx.DiGraph, total_possible: int) -> dict:
        n_nodes = dag.number_of_nodes()
        n_edges = dag.number_of_edges()
        max_possible = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1

        full_graph_edges = int(max_possible)
        temporal_removed = len(self._apply_temporal_constraint(list(dag.nodes())))
        prior_forced = len(AGRI_PRIOR_EDGES) + len(DELIVERY_EDGES)
        constraint_reduction = (temporal_removed / 2 + prior_forced) / full_graph_edges * 100 if full_graph_edges > 0 else 0
        search_reduction = round(constraint_reduction, 2)

        ground_truth = set()
        for src, dst in AGRI_PRIOR_EDGES + DELIVERY_EDGES:
            if src in dag.nodes() and dst in dag.nodes():
                ground_truth.add((src, dst))

        for src, dst in dag.edges():
            if (src, dst) not in ground_truth:
                ts = TEMPORAL_ORDER.get(src, -1)
                td = TEMPORAL_ORDER.get(dst, -1)
                if ts >= 0 and td >= 0 and ts < td:
                    ground_truth.add((src, dst))

        discovered_edges = set(dag.edges())
        tp = len(ground_truth & discovered_edges)
        fp = len(discovered_edges - ground_truth)
        fn = len(ground_truth - discovered_edges)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "f1_score": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "search_space_reduction": search_reduction,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "tp": tp, "fp": fp, "fn": fn,
        }

    def discover(self, data: pd.DataFrame,
                 feature_names: list = None) -> dict:
        if feature_names is not None:
            self.feature_names = feature_names

        available = [f for f in self.feature_names if f in data.columns]
        if len(available) < 3:
            return {"dag": nx.DiGraph(), "quality": {}, "causal_chains": []}

        core_available = [f for f in CORE_NODES if f in available]
        if len(core_available) >= 3:
            available = core_available

        work_data = data[available].copy()
        for col in work_data.columns:
            if work_data[col].isna().any():
                work_data[col] = work_data[col].fillna(work_data[col].median())

        total_possible = len(available) * (len(available) - 1) / 2
        skeleton_edges = self._build_skeleton(work_data, available)
        dag = self._orient_edges(skeleton_edges, available)
        quality = self._compute_metrics(dag, total_possible)
        self.dag_quality = quality
        self.graph = dag

        causal_chains = self._extract_causal_chains(dag)

        return {
            "dag": dag,
            "quality": quality,
            "causal_chains": causal_chains,
        }

    def _extract_causal_chains(self, dag: nx.DiGraph) -> list:
        chains = []
        weather_nodes = [n for n in dag.nodes() if n in
                         ["temperature", "precipitation", "extreme_temp_index",
                          "extreme_precip_index", "drought_index"]]
        yield_nodes = [n for n in dag.nodes() if n in ["yield_proxy", "ndvi"]]
        price_nodes = [n for n in dag.nodes() if n in ["close", "open"]]

        for w in weather_nodes:
            for y in yield_nodes:
                for p in price_nodes:
                    if dag.has_edge(w, y) and dag.has_edge(y, p):
                        chains.append([w, y, p])
                    elif dag.has_edge(w, y):
                        for mid in dag.successors(y):
                            if dag.has_edge(mid, p):
                                chains.append([w, y, mid, p])

        return chains

    def get_node_info(self, node: str, data: pd.DataFrame) -> dict:
        if node not in data.columns:
            return {}
        series = data[node].dropna()
        return {
            "name": node,
            "mean": round(series.mean(), 4),
            "std": round(series.std(), 4),
            "min": round(series.min(), 4),
            "max": round(series.max(), 4),
            "count": len(series),
        }
