# DistillationSVX

This workspace contains two related parts:

1. **`GraphSVX/`**  
   The original PyTorch Geometric GraphSVX codebase, used for GraphSVX demos and reference implementation details.

2. **`asd_exp/`**  
   DGL-based WCD regression experiments for feedforward/non-feedforward industrial control network graphs. This is the main project code for training ClassicGNN models and running the adapted GraphSVX-style explanation module.

## Environment

### macOS arm64 / CPU environment

```bash
conda create -n dis_svx_env python=3.10
conda activate dis_svx_env
pip install -r requirements.txt
```

### Windows environment, e.g. Ryzen 9700X + RTX 5070

```bash
conda create -n dis_svx_env_win python=3.10
conda activate dis_svx_env_win
python -m pip install --upgrade pip
pip install -r requirements_windows.txt
```

> **Note for Windows:**  
> `requirements_windows.txt` installs PyTorch CUDA 12.8 wheels for new NVIDIA GPUs. Native Windows DGL CUDA support may lag behind PyTorch/CUDA support, so DGL may run on CPU. For reliable DGL GPU acceleration, WSL2/Linux is recommended.

## Data

The `asd_exp/data` directory contains DGL CSV datasets:

```text
asd_exp/data/ffw_origin
asd_exp/data/nffw_origin
asd_exp/data/ffw_distill
asd_exp/data/nffw_distill
```

`ffw` means feedforward traffic pattern.  
`nffw` means non-feedforward traffic pattern.

Each dataset contains:

```text
meta.yaml
nodes.csv
edges.csv
graphs.csv
```

### `nodes.csv`

Columns:

```csv
graph_id,node_id,label,feat
```

Meaning:

| Column | Meaning |
| --- | --- |
| `graph_id` | which network snapshot this node belongs to |
| `node_id` | node id inside the graph |
| `label` | `-1` for non-target/server nodes; `0,1,2,...` for flow nodes |
| `feat` | node feature vector |

In this WCD setting, nodes represent both server nodes and flow nodes. Server node features encode service-curve-related parameters. Flow node features encode arrival-curve-related parameters.

### `edges.csv`

Columns:

```csv
graph_id,src_id,dst_id,feat
```

Meaning:

| Column | Meaning |
| --- | --- |
| `graph_id` | which network snapshot this edge belongs to |
| `src_id` / `dst_id` | directed edge endpoints |
| `feat` | path-ordering positional encoding for the flow-server relation |

The edge features encode the order in which a flow traverses servers, which is important for WCD prediction.

### `graphs.csv`

Columns:

```csv
graph_id,label
```

The graph label is a vector of flow-specific WCD values. For example:

```csv
0,"452.84, 134.55, 514.94, -1, -1, -1, -1"
```

This means graph `0` has valid WCD labels for flow `0`, flow `1`, and flow `2`. Values `-1` are padding and are ignored during training/evaluation.

## Training ClassicGNN Models

The main training script is:

```text
asd_exp/exp2/exp1.py
```

Run from the repository root:

```bash
conda activate dis_svx_env
python asd_exp/exp2/exp1.py \
  --dataset_path asd_exp/data/ffw_origin \
  --classicGNN_type GCN \
  --batch_size 128 \
  --num_epochs 1600
```

Supported model types in `asd_exp/exp2/classical_gnn.py`:

```text
GCN
GAT
EdgeGAT
GINE
GatedGCN
```

The trained weights are saved to:

```text
asd_exp/exp2/model_weights/best_model_<MODEL_TYPE>.pth
```

Example:

```text
asd_exp/exp2/model_weights/best_model_GCN.pth
```

## Evaluating A Trained Model

Use:

```text
asd_exp/exp2/exp2.py
```

Example:

```bash
cd asd_exp/exp2
python exp2.py \
  --dataset_path ../data/ffw_origin \
  --classicGNN_type GCN \
  --verify_model model_weights/best_model_GCN.pth
```

The script reports MAE on the selected dataset.

## GraphSVX-Style DGL Explanations

The adapted explanation script is:

```text
asd_exp/explain_graphsvx_dgl.py
```

It ports the core GraphSVX idea to the DGL WCD regression setting:

1. Select a graph and one target flow WCD output.
2. Treat target flow-node features and graph nodes as explanation players.
3. Sample coalitions using GraphSVX-style samplers.
4. Perturb the DGL graph by masking features/nodes.
5. Query the trained ClassicGNN on each perturbed graph.
6. Fit a Shapley-kernel-weighted linear surrogate.
7. Report feature and node contributions.

Example:

```bash
cd asd_exp
python explain_graphsvx_dgl.py \
  --dataset_path data/ffw_origin \
  --model_path exp2/model_weights/best_model_GCN.pth \
  --classicGNN_type GCN \
  --graph_index 0 \
  --target_index 0 \
  --num_samples 800 \
  --coal Smarter \
  --feat All \
  --surrogate WLS \
  --top_k 10
```

### Common arguments

| Argument | Meaning |
| --- | --- |
| `--dataset_path` | Dataset directory, e.g. `data/ffw_origin` or `data/nffw_origin`. |
| `--model_path` | Path to trained ClassicGNN weights. |
| `--classicGNN_type` | Must match the trained model type, e.g. `GCN` or `EdgeGAT`. |
| `--graph_index` | Which graph/network snapshot to explain. |
| `--target_index` | Which flow WCD output to explain. |
| `--num_samples` | Number of perturbed coalitions. Larger values are slower but more stable. |
| `--coal` | Coalition sampler: `Smarter`, `Smart`, `Random`, `All`, `SmarterSeparate`. |
| `--feat` | Which target flow-node features to include: `All`, `Null`, `Expectation`, `None`. |
| `--mask_nodes` | `isolate` removes incident edges of masked nodes, close to original GraphSVX `compute_pred` behavior. `neutral` keeps edges but replaces masked node features. |
