"""GraphSVX-style explanations for asd_exp DGL ClassicGNN models.

This is a DGL/regression port of the core GraphSVX pipeline:

1. Pick a graph and one WCD output target.
2. Treat the target flow-node features and the graph nodes as players.
3. Generate binary coalitions with GraphSVX samplers.
4. Convert each simplified coalition z into a perturbed DGL graph z'.
5. Query the trained ClassicGNN on z'.
6. Fit a weighted linear surrogate with the GraphSVX Shapley kernel.

It does not import ``GraphSVX.src.explainers.GraphSVX`` because the original
implementation is tied to PyTorch Geometric classification models. The
algorithmic pieces below are ported/adapted to the DGL WCD regression setting.

Example:
    python explain_graphsvx_dgl.py \
        --dataset_path data/ffw_origin \
        --model_path exp2/model_weights/best_model_GCN.pth \
        --classicGNN_type GCN \
        --graph_index 0 \
        --target_index 0 \
        --num_samples 800 \
        --coal Smarter \
        --feat All \
        --surrogate WLS
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import dgl
import numpy as np
import torch as th
from scipy.special import binom
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parent
METHOD_DIR = ROOT / "method"
EXP2_DIR = ROOT / "exp2"
for import_dir in (EXP2_DIR, METHOD_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from classical_gnn import ClassicGNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain asd_exp ClassicGNN WCD predictions with a GraphSVX-style DGL port."
    )
    parser.add_argument("--dataset_path", default="data/ffw_origin")
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--classicGNN_type",
        default="GCN",
        choices=["GCN", "GAT", "EdgeGAT", "GINE", "GatedGCN"],
    )
    parser.add_argument("--graph_index", type=int, default=0)
    parser.add_argument(
        "--target_index",
        type=int,
        default=0,
        help="Which WCD output / flow label to explain.",
    )
    parser.add_argument("--num_samples", type=int, default=800)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--baseline",
        default="mean",
        choices=["mean", "zero"],
        help="Feature replacement used when a feature/node is switched off.",
    )
    parser.add_argument(
        "--feat",
        default="All",
        choices=["All", "Null", "Expectation", "None"],
        help=(
            "GraphSVX feature selection on the target flow node. "
            "All=all target-node feature dimensions, Null=non-zero dims, "
            "Expectation=dims away from dataset mean, None=node-only explanation."
        ),
    )
    parser.add_argument(
        "--coal",
        default="Smarter",
        choices=["Smarter", "Smart", "Random", "All", "SmarterSeparate", "NewSmarterSeparate"],
        help="GraphSVX coalition sampler.",
    )
    parser.add_argument("--S", type=int, default=3, help="Maximum special coalition size.")
    parser.add_argument(
        "--surrogate",
        default="WLS",
        choices=["WLS", "WLR_sklearn", "Ridge", "Lasso"],
        help="Weighted explanation model used to estimate Shapley values.",
    )
    parser.add_argument("--alpha", type=float, default=1e-6, help="Ridge/Lasso regularization.")
    parser.add_argument(
        "--regu",
        type=float,
        default=None,
        help="Optional GraphSVX-style balance: 1=feature-only, 0=node-only.",
    )
    parser.add_argument(
        "--mask_nodes",
        default="isolate",
        choices=["isolate", "neutral"],
        help=(
            "isolate removes incident edges of excluded nodes, matching GraphSVX compute_pred; "
            "neutral keeps edges but replaces excluded node features."
        ),
    )
    parser.add_argument(
        "--fullempty",
        action="store_true",
        help="Set full/empty coalition weights to 0 instead of high GraphSVX weight.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return ROOT / candidate


def dataset_path_for_dgl(dataset_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    graph_file = dataset_path / "graphs.csv"
    legacy_graph_file = dataset_path / "graphso.csv"
    if graph_file.exists() or not legacy_graph_file.exists():
        return dataset_path, None

    temp_dir = tempfile.TemporaryDirectory(prefix="asd_graphsvx_")
    mirror = Path(temp_dir.name) / dataset_path.name
    shutil.copytree(dataset_path, mirror)
    shutil.copyfile(mirror / "graphso.csv", mirror / "graphs.csv")
    return mirror, temp_dir


def load_dataset(dataset_path: Path):
    readable_path, temp_dir = dataset_path_for_dgl(dataset_path)
    return dgl.data.CSVDataset(str(readable_path)), temp_dir


def load_model(model_path: Path, model_type: str, device: th.device) -> ClassicGNN:
    model = ClassicGNN(model_type=model_type).to(device)
    state_dict = th.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def add_self_loop_with_features(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    try:
        return dgl.add_self_loop(graph, fill_data=0.0)
    except TypeError:
        return dgl.add_self_loop(graph)


def predict_target(model: ClassicGNN, graph: dgl.DGLGraph, target_index: int, device: th.device) -> float:
    graph = add_self_loop_with_features(graph).to(device)
    num_nodes = th.tensor([graph.num_nodes()], device=device)
    with th.no_grad():
        scores = model(graph, num_nodes).float()[0]
    if target_index >= scores.shape[0]:
        raise IndexError(
            f"target_index={target_index} is out of range for model output length {scores.shape[0]}."
        )
    return float(scores[target_index].detach().cpu())


def graph_label_at(dataset_item, target_index: int) -> float | None:
    if not isinstance(dataset_item, tuple) or len(dataset_item) < 2:
        return None
    label = dataset_item[1]
    if target_index >= len(label):
        return None
    value = label[target_index]
    if float(value) == -1.0:
        return None
    return float(value)


def flow_node_index(graph: dgl.DGLGraph, target_index: int) -> int:
    labels = graph.ndata["label"].detach().cpu()
    matches = (labels == target_index).nonzero(as_tuple=False).flatten()
    if matches.numel() == 0:
        raise ValueError(f"No flow node with graph.ndata['label'] == {target_index}.")
    return int(matches[0])


def feature_baseline(values: th.Tensor, baseline: str) -> th.Tensor:
    if baseline == "mean":
        return values.float().mean(dim=0)
    return th.zeros(values.shape[1], dtype=values.dtype)


def select_target_features(
    graph: dgl.DGLGraph,
    target_node: int,
    feat_mode: str,
    baseline: str,
) -> tuple[list[int], list[int]]:
    feat = graph.ndata["feat"].float()
    if feat_mode == "None":
        return [], list(range(feat.shape[1]))
    if feat_mode == "All":
        return list(range(feat.shape[1])), []
    if feat_mode == "Null":
        selected = feat[target_node].nonzero(as_tuple=False).flatten().tolist()
        discarded = [idx for idx in range(feat.shape[1]) if idx not in selected]
        return selected, discarded

    std = feat.std(dim=0)
    mean = feature_baseline(feat, baseline)
    target_feat = feat[target_node]
    away = (target_feat < mean - 0.25 * std) | (target_feat > mean + 0.25 * std)
    selected = away.nonzero(as_tuple=False).flatten().tolist()
    discarded = (~away).nonzero(as_tuple=False).flatten().tolist()
    return selected, discarded


class GraphSVXDGLExplainer:
    def __init__(
        self,
        model: ClassicGNN,
        graph: dgl.DGLGraph,
        target_index: int,
        device: th.device,
        baseline: str,
        mask_nodes: str,
    ):
        self.model = model
        self.graph = graph.cpu()
        self.target_index = target_index
        self.device = device
        self.baseline = baseline
        self.mask_nodes = mask_nodes
        self.target_node = flow_node_index(graph, target_index)
        self.node_players = [idx for idx in range(graph.num_nodes()) if idx != self.target_node]
        self.feature_players: list[int] = []
        self.discarded_features: list[int] = []
        self.F = 0
        self.D = len(self.node_players)
        self.M = self.D

    def configure_players(self, feat_mode: str, regu: float | None) -> None:
        self.feature_players, self.discarded_features = select_target_features(
            self.graph, self.target_node, feat_mode, self.baseline
        )
        self.F = len(self.feature_players)
        if regu == 0 or self.F == 0:
            self.F = 0
            self.feature_players = []
        if regu == 1 or self.D == 0:
            self.D = 0
            self.node_players = []
        self.M = self.F + self.D
        if self.M == 0:
            raise ValueError("No feature or node players selected for explanation.")

    def mask_generation(
        self,
        num_samples: int,
        coal: str,
        max_size: int,
        regu: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if coal in {"SmarterSeparate", "NewSmarterSeparate"}:
            if self.F == 0 or self.D == 0:
                coalitions = self.Smarter(num_samples, max_size)
                weights = self.shapley_kernel(coalitions.sum(axis=1), self.M)
                return coalitions, weights

            if regu is not None:
                num_feat = int(num_samples * regu)
            else:
                num_feat = int(0.5 * num_samples / 2 + 0.5 * num_samples * self.F / self.M)
            num_feat = max(2, min(num_samples - 2, num_feat))

            feat_only = self.SmarterSeparate(num_feat, max_size, feature_side=True)
            node_only = self.SmarterSeparate(num_samples - num_feat, max_size, feature_side=False)
            coalitions = np.zeros((num_samples, self.M), dtype=np.float64)
            coalitions[:num_feat, : self.F] = feat_only
            coalitions[num_feat:, :] = 1.0
            coalitions[num_feat:, self.F :] = node_only

            weights = np.zeros(num_samples, dtype=np.float64)
            weights[:num_feat] = self.shapley_kernel(feat_only.sum(axis=1), self.F)
            weights[num_feat:] = self.shapley_kernel(node_only.sum(axis=1), self.D)
        else:
            if coal == "All":
                num_samples = min(10000, 2**self.M)
            sampler = getattr(self, coal)
            coalitions = sampler(num_samples, max_size)
            weights = self.shapley_kernel(coalitions.sum(axis=1), self.M)

        order = np.random.permutation(coalitions.shape[0])
        return coalitions[order], weights[order]

    def shapley_kernel(self, sizes: np.ndarray, players: int) -> np.ndarray:
        values = []
        for size in sizes.astype(int):
            if size == 0 or size == players:
                values.append(1000.0)
            elif binom(players, size) == float("+inf"):
                values.append(1.0 / (players**2))
            else:
                values.append((players - 1) / (binom(players, size) * size * (players - size)))
        return np.maximum(np.asarray(values, dtype=np.float64), 1.0e-40)

    def Random(self, num_samples: int, *_unused) -> np.ndarray:
        return np.random.randint(0, 2, size=(num_samples, self.M)).astype(np.float64)

    def All(self, num_samples: int, *_unused) -> np.ndarray:
        coalitions = np.zeros((num_samples, self.M), dtype=np.float64)
        row = 0
        for size in range(self.M + 1):
            for combo in combinations(range(self.M), size):
                if row >= num_samples:
                    return coalitions
                coalitions[row, list(combo)] = 1.0
                row += 1
        return coalitions

    def Smart(self, num_samples: int, max_size: int) -> np.ndarray:
        coalitions = np.ones((num_samples, self.M), dtype=np.float64)
        coalitions[1::2] = 0.0
        row = 2
        size = 1
        while row < num_samples:
            if row + 2 * self.M < num_samples and size == 1:
                coalitions[row : row + self.M] = 1.0
                np.fill_diagonal(coalitions[row : row + self.M], 0.0)
                row += self.M
                coalitions[row : row + self.M] = 0.0
                np.fill_diagonal(coalitions[row : row + self.M], 1.0)
                row += self.M
                size += 1
            elif size < max_size:
                cutoff = row + 4 * (num_samples - row) // 5
                combos = list(combinations(range(self.M), size))[: max(0, cutoff - row + 1)]
                random.shuffle(combos)
                for combo in combos:
                    coalitions[row, list(combo)] = 0.0
                    row += 1
                    if row >= cutoff:
                        coalitions[row:] = self.Random(num_samples - row)
                        return coalitions
                    coalitions[row] = 0.0
                    coalitions[row, list(combo)] = 1.0
                    row += 1
                    if row >= cutoff:
                        coalitions[row:] = self.Random(num_samples - row)
                        return coalitions
                size += 1
            else:
                coalitions[row:] = self.Random(num_samples - row)
                return coalitions
        return coalitions

    def Smarter(self, num_samples: int, max_size: int) -> np.ndarray:
        coalitions = np.ones((num_samples, self.M), dtype=np.float64)
        coalitions[1::2] = 0.0
        row = 2
        size = 1
        while row < num_samples:
            if row + 2 * self.M < num_samples and size == 1:
                coalitions[row : row + self.M] = 1.0
                np.fill_diagonal(coalitions[row : row + self.M], 0.0)
                row += self.M
                coalitions[row : row + self.M] = 0.0
                np.fill_diagonal(coalitions[row : row + self.M], 1.0)
                row += self.M
                size += 1
            else:
                cutoff = row + 9 * (num_samples - row) // 10
                while row < cutoff and size <= max_size:
                    feature_combos = list(combinations(range(self.F), size))
                    node_combos = list(combinations(range(self.F, self.M), size))
                    combos = feature_combos + node_combos
                    random.shuffle(combos)
                    for combo in combos[: max(0, cutoff - row + 1)]:
                        coalitions[row, list(combo)] = 0.0
                        row += 1
                        if row >= cutoff:
                            coalitions[row:] = self.Random(num_samples - row)
                            return coalitions
                        coalitions[row] = 0.0
                        coalitions[row, list(combo)] = 1.0
                        row += 1
                        if row >= cutoff:
                            coalitions[row:] = self.Random(num_samples - row)
                            return coalitions
                    size += 1
                coalitions[row:] = self.Random(num_samples - row)
                return coalitions
        return coalitions

    def SmarterSeparate(self, num_samples: int, max_size: int, feature_side: bool) -> np.ndarray:
        players = self.F if feature_side else self.D
        coalitions = np.ones((num_samples, players), dtype=np.float64)
        coalitions[1::2] = 0.0
        row = 2
        size = 1
        while row < num_samples:
            if row + 2 * players < num_samples and size == 1:
                coalitions[row : row + players] = 1.0
                np.fill_diagonal(coalitions[row : row + players], 0.0)
                row += players
                coalitions[row : row + players] = 0.0
                np.fill_diagonal(coalitions[row : row + players], 1.0)
                row += players
                size += 1
            else:
                cutoff = num_samples if not feature_side else row + 9 * (num_samples - row) // 10
                while row < cutoff and size <= min(max_size, players):
                    combos = list(combinations(range(players), size))
                    random.shuffle(combos)
                    for combo in combos[: max(0, cutoff - row + 1)]:
                        coalitions[row, list(combo)] = 0.0
                        row += 1
                        if row >= cutoff:
                            coalitions[row:] = np.random.randint(0, 2, size=(num_samples - row, players))
                            return coalitions
                        coalitions[row] = 0.0
                        coalitions[row, list(combo)] = 1.0
                        row += 1
                        if row >= cutoff:
                            coalitions[row:] = np.random.randint(0, 2, size=(num_samples - row, players))
                            return coalitions
                    size += 1
                coalitions[row:] = np.random.randint(0, 2, size=(num_samples - row, players))
                return coalitions
        return coalitions

    def perturb_graph(self, coalition: np.ndarray) -> dgl.DGLGraph:
        src, dst = self.graph.edges()
        keep_edges = th.ones(src.shape[0], dtype=th.bool)

        feature_mask = coalition[: self.F]
        node_mask = coalition[self.F :]
        excluded_nodes = [self.node_players[i] for i, keep in enumerate(node_mask) if keep == 0]

        if self.mask_nodes == "isolate" and excluded_nodes:
            excluded_tensor = th.tensor(excluded_nodes, dtype=src.dtype)
            keep_edges = ~(th.isin(src, excluded_tensor) | th.isin(dst, excluded_tensor))

        new_graph = dgl.graph((src[keep_edges], dst[keep_edges]), num_nodes=self.graph.num_nodes())

        for key, value in self.graph.ndata.items():
            new_graph.ndata[key] = value.clone()
        for key, value in self.graph.edata.items():
            new_graph.edata[key] = value[keep_edges].clone()

        if "feat" in new_graph.ndata:
            node_feat = new_graph.ndata["feat"].clone()
            node_baseline = feature_baseline(self.graph.ndata["feat"], self.baseline)
            for pos, feature_id in enumerate(self.feature_players):
                if feature_mask[pos] == 0:
                    node_feat[self.target_node, feature_id] = node_baseline[feature_id]
            if self.mask_nodes == "neutral":
                for node_idx in excluded_nodes:
                    node_feat[node_idx] = node_baseline
            new_graph.ndata["feat"] = node_feat

        return new_graph

    def compute_fz(self, coalitions: np.ndarray) -> np.ndarray:
        fz = np.zeros(coalitions.shape[0], dtype=np.float64)
        for row, coalition in enumerate(coalitions):
            perturbed = self.perturb_graph(coalition)
            fz[row] = predict_target(self.model, perturbed, self.target_index, self.device)
        return fz

    def fit_surrogate(
        self,
        coalitions: np.ndarray,
        weights: np.ndarray,
        fz: np.ndarray,
        surrogate: str,
        alpha: float,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        if surrogate == "WLS":
            x = np.concatenate([coalitions, np.ones((coalitions.shape[0], 1))], axis=1)
            weights = np.nan_to_num(weights, nan=0.0, posinf=1000.0, neginf=0.0)
            sqrt_w = np.sqrt(np.maximum(weights, 0.0))[:, None]
            xw = x * sqrt_w
            yw = fz * sqrt_w.flatten()
            try:
                params = np.linalg.lstsq(xw, yw, rcond=None)[0]
            except np.linalg.LinAlgError:
                xtwx = xw.T @ xw
                inv = np.linalg.inv(xtwx + np.diag(1e-5 * np.random.randn(xtwx.shape[0])))
                params = inv @ (xw.T @ yw)
            pred = x @ params
            return params[:-1], float(params[-1]), pred

        if surrogate == "WLR_sklearn":
            reg = LinearRegression()
        elif surrogate == "Lasso":
            reg = Lasso(alpha=alpha)
        else:
            reg = Ridge(alpha=alpha)
        reg.fit(coalitions, fz, sample_weight=weights)
        pred = reg.predict(coalitions)
        return reg.coef_.astype(np.float64), float(reg.intercept_), pred

    def explain(
        self,
        num_samples: int,
        feat_mode: str,
        coal: str,
        max_size: int,
        surrogate: str,
        alpha: float,
        regu: float | None,
        fullempty: bool,
    ) -> dict[str, object]:
        self.configure_players(feat_mode, regu)
        coalitions, weights = self.mask_generation(num_samples, coal, max_size, regu)
        if fullempty:
            weights = np.where(weights == 1000.0, 0.0, weights)
        fz = self.compute_fz(coalitions)
        phi, base_value, pred = self.fit_surrogate(coalitions, weights, fz, surrogate, alpha)

        full_value = predict_target(self.model, self.graph, self.target_index, self.device)
        empty = np.zeros(self.M, dtype=np.float64)
        empty_value = predict_target(self.model, self.perturb_graph(empty), self.target_index, self.device)
        weighted_r2 = r2_score(fz, pred, sample_weight=weights) if np.sum(weights) > 0 else float("nan")
        raw_r2 = r2_score(fz, pred)
        return {
            "phi": phi,
            "base_value": base_value,
            "full_value": full_value,
            "empty_value": empty_value,
            "approx_full": base_value + float(phi.sum()),
            "weighted_r2": weighted_r2,
            "r2": raw_r2,
            "fz": fz,
        }


def print_explanation(
    graph: dgl.DGLGraph,
    explainer: GraphSVXDGLExplainer,
    result: dict[str, object],
    label: float | None,
    args: argparse.Namespace,
    dataset_path: Path,
    model_path: Path,
) -> None:
    phi = result["phi"]
    assert isinstance(phi, np.ndarray)

    node_ids = graph.nodes().detach().cpu().numpy()
    feature_phi = phi[: explainer.F]
    node_phi = phi[explainer.F :]

    print("GraphSVX-DGL full explanation")
    print(f"dataset_path: {dataset_path}")
    print(f"model_path: {model_path}")
    print(f"model_type: {args.classicGNN_type}")
    print(f"graph_index: {args.graph_index}")
    print(f"target_index: {args.target_index}")
    print(f"target_flow_node_id: {int(node_ids[explainer.target_node])}")
    print(f"target_flow_local_index: {explainer.target_node}")
    if label is not None:
        print(f"target_label: {label:.6f}")
    print(f"prediction(full graph): {float(result['full_value']):.6f}")
    print(f"empty_value(all players off): {float(result['empty_value']):.6f}")
    print(f"surrogate_base_value: {float(result['base_value']):.6f}")
    print(f"surrogate_base + sum(phi): {float(result['approx_full']):.6f}")
    print(f"weighted_r2: {float(result['weighted_r2']):.6f}")
    print(f"r2: {float(result['r2']):.6f}")
    print(f"players: {explainer.F} target-node features + {explainer.D} graph nodes = {explainer.M}")
    print()

    if explainer.F:
        feature_order = np.argsort(np.abs(feature_phi))[::-1]
        print(f"Target flow-node feature contributions by absolute value:")
        for rank, local_pos in enumerate(feature_order, start=1):
            feature_id = explainer.feature_players[local_pos]
            print(
                f"{rank:>2}. feature_index={feature_id} "
                f"contribution={feature_phi[local_pos]:+.6f}"
            )
        print()

    node_order = np.argsort(np.abs(node_phi))[::-1]
    print(f"Top {min(args.top_k, len(node_order))} node contributions by absolute value:")
    for rank, local_pos in enumerate(node_order[: args.top_k], start=1):
        graph_local_idx = explainer.node_players[local_pos]
        print(
            f"{rank:>2}. node_id={int(node_ids[graph_local_idx])} "
            f"local_index={int(graph_local_idx)} "
            f"contribution={node_phi[local_pos]:+.6f}"
        )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    device = th.device(args.device if args.device == "cuda" and th.cuda.is_available() else "cpu")
    dataset_path = resolve_path(args.dataset_path)
    model_path = resolve_path(args.model_path)

    dataset, temp_dir = load_dataset(dataset_path)
    try:
        item = dataset[args.graph_index]
        graph = item[0] if isinstance(item, tuple) else item
        label = graph_label_at(item, args.target_index)
        model = load_model(model_path, args.classicGNN_type, device)
        explainer = GraphSVXDGLExplainer(
            model=model,
            graph=graph,
            target_index=args.target_index,
            device=device,
            baseline=args.baseline,
            mask_nodes=args.mask_nodes,
        )
        result = explainer.explain(
            num_samples=args.num_samples,
            feat_mode=args.feat,
            coal=args.coal,
            max_size=args.S,
            surrogate=args.surrogate,
            alpha=args.alpha,
            regu=args.regu,
            fullempty=args.fullempty,
        )
        print_explanation(graph, explainer, result, label, args, dataset_path, model_path)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
