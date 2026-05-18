DistillationSVX
===============

DistillationSVX is a DGL-based experiment workspace for worst-case delay
(WCD) prediction on feedforward and non-feedforward industrial network graphs.

The project contains:

  - ClassicGNN training and evaluation code for WCD regression.
  - DGL CSV datasets for FFW / NFFW network snapshots.
  - A GraphSVX-inspired explainer adapted to DGL graph regression models.


Quick Start
-----------

Create the environment:

    conda create -n dis_svx_env python=3.10
    conda activate dis_svx_env
    pip install -r requirements.txt

Train a GCN on feedforward original data:

    python asd_exp/exp2/exp1.py \
      --dataset_path asd_exp/data/ffw_origin \
      --classicGNN_type GCN \
      --batch_size 128 \
      --num_epochs 1600

Evaluate the trained model:

    cd asd_exp/exp2
    python exp2.py \
      --dataset_path ../data/ffw_origin \
      --classicGNN_type GCN \
      --verify_model model_weights/best_model_GCN.pth

Run a GraphSVX-style explanation:

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


Repository Layout
-----------------

    requirements.txt
        macOS arm64 / CPU environment used during development.

    requirements_windows.txt
        Windows environment notes and dependencies, including CUDA 12.8 PyTorch
        wheels for newer NVIDIA GPUs.

    asd_exp/
        Main experiment code and datasets.

    asd_exp/exp2/classical_gnn.py
        Model definitions: GCN, GAT, EdgeGAT, GINE, GatedGCN, and the shared
        prediction head.

    asd_exp/exp2/exp1.py
        Training script.

    asd_exp/exp2/exp2.py
        Evaluation script.

    asd_exp/explain_graphsvx_dgl.py
        GraphSVX-inspired DGL explanation module for WCD regression models.

    asd_exp/data/
        DGL CSV graph datasets.


Environment Notes
-----------------

macOS / CPU:

    conda create -n dis_svx_env python=3.10
    conda activate dis_svx_env
    pip install -r requirements.txt

Windows, for example Ryzen 9700X + RTX 5070:

    conda create -n dis_svx_env_win python=3.10
    conda activate dis_svx_env_win
    python -m pip install --upgrade pip
    pip install -r requirements_windows.txt

Important Windows note:

    requirements_windows.txt installs CUDA 12.8 PyTorch wheels for newer NVIDIA
    GPUs. Native Windows DGL CUDA support may lag behind PyTorch/CUDA support,
    so DGL may run on CPU. For reliable DGL GPU training, WSL2/Linux is usually
    the safer route.


Dataset Format
--------------

The datasets live under:

    asd_exp/data/ffw_origin
    asd_exp/data/nffw_origin
    asd_exp/data/ffw_distill
    asd_exp/data/nffw_distill

Naming:

    FFW   = feedforward traffic pattern
    NFFW  = non-feedforward traffic pattern
    WCD   = worst-case delay

Each DGL CSV dataset contains:

    meta.yaml
    nodes.csv
    edges.csv
    graphs.csv

nodes.csv:

    graph_id,node_id,label,feat

Meaning:

    graph_id  - network snapshot id
    node_id   - node id inside the graph
    label     - -1 for server/non-target nodes; 0,1,2,... for flow nodes
    feat      - node feature vector

In this project, each graph contains both server nodes and flow nodes. Server
node features encode service-curve-related parameters; flow node features
encode arrival-curve-related parameters.

edges.csv:

    graph_id,src_id,dst_id,feat

Meaning:

    graph_id    - network snapshot id
    src_id      - directed edge source node
    dst_id      - directed edge target node
    feat        - positional/path-order encoding for the flow-server relation

The edge features are important because WCD depends not only on which servers a
flow traverses, but also on the traversal order and flow interactions.

graphs.csv:

    graph_id,label

The graph label is a padded vector of flow-specific WCD values. Example:

    0,"452.84, 134.55, 514.94, -1, -1, -1, -1"

This means graph 0 has valid WCD labels for flow 0, flow 1, and flow 2. The
values -1 are padding and are ignored during training and evaluation.


Models
------

Supported ClassicGNN model types:

    GCN
    GAT
    EdgeGAT
    GINE
    GatedGCN

All models share the same high-level interface:

    model(graph, num_nodes_per_graph) -> [batch_size, num_flows]

The model pipeline is:

    DGL graph
      -> node/edge encoding
      -> GNN layers
      -> flow-node embeddings
      -> MLP prediction head
      -> flow-specific WCD predictions

Saved checkpoints are written to:

    asd_exp/exp2/model_weights/best_model_<MODEL_TYPE>.pth

Example:

    asd_exp/exp2/model_weights/best_model_GCN.pth


Training
--------

Run from the repository root:

    python asd_exp/exp2/exp1.py \
      --dataset_path asd_exp/data/ffw_origin \
      --classicGNN_type GCN \
      --batch_size 128 \
      --num_epochs 1600

Use another model:

    python asd_exp/exp2/exp1.py \
      --dataset_path asd_exp/data/nffw_origin \
      --classicGNN_type EdgeGAT \
      --batch_size 128 \
      --num_epochs 1600

The training loss is L1 loss over valid WCD entries. Padding labels with value
-1 are ignored.


Evaluation
----------

Run from asd_exp/exp2:

    cd asd_exp/exp2
    python exp2.py \
      --dataset_path ../data/ffw_origin \
      --classicGNN_type GCN \
      --verify_model model_weights/best_model_GCN.pth

The script reports MAE on the selected dataset.


GraphSVX-Style Explanations
---------------------------

The adapted explanation script is:

    asd_exp/explain_graphsvx_dgl.py

It is a DGL/regression adaptation of the GraphSVX idea. It does not import the
original PyTorch Geometric GraphSVX class. Instead, it follows the same core
logic:

    1. Select one graph and one target flow WCD output.
    2. Treat target flow-node features and graph nodes as explanation players.
    3. Sample binary coalitions.
    4. Perturb the DGL graph according to each coalition.
    5. Query the trained ClassicGNN on each perturbed graph.
    6. Fit a Shapley-kernel-weighted linear surrogate.
    7. Report feature and node contribution scores.

Example:

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

Useful options:

    --graph_index
        Selects the network snapshot.

    --target_index
        Selects which flow WCD output to explain.

    --num_samples
        Number of perturbed coalitions. Larger values are slower but more
        stable.

    --coal
        Coalition sampler. Common choices: Smarter, Smart, Random, All.

    --feat
        Target flow-node feature selection. Common choices: All, Null,
        Expectation, None.

    --mask_nodes
        isolate: remove incident edges of masked nodes.
        neutral: keep edges but replace masked node features with baseline.

    --surrogate
        Weighted explanation model. Common choices: WLS, WLR_sklearn, Ridge.


Reading Explanation Output
--------------------------

Example output:

    prediction(full graph): 846.511353
    empty_value(all players off): 522.998962
    surrogate_base_value: 522.989937
    surrogate_base + sum(phi): 846.522526
    weighted_r2: 0.999946
    r2: 0.930380

Meaning:

    prediction(full graph)
        The trained model's WCD prediction on the original graph.

    empty_value(all players off)
        Prediction when all selected explanation players are masked.

    surrogate_base_value
        Intercept of the weighted linear explanation model.

    surrogate_base + sum(phi)
        Reconstruction of the prediction from all contribution scores.

    weighted_r2
        Fit quality of the explanation surrogate under GraphSVX/Shapley kernel
        weights. This is usually the more important R2.

    r2
        Ordinary unweighted fit quality over all sampled perturbations.

Node contribution example:

    node_id=6 local_index=6 contribution=+123.049280

Meaning:

    For the selected graph and target flow, node 6 pushes the model's predicted
    WCD upward by about 123.05 relative to the surrogate baseline.

Positive contribution:

    pushes the target WCD prediction upward.

Negative contribution:

    pushes the target WCD prediction downward.

local_index is the internal DGL node position. node_id is the id from the CSV
dataset. In these datasets they often match, but conceptually they are
different.


Graphormer Notes
----------------

A strict Graphormer teacher model is not just another DGL message-passing layer.
It usually needs additional graph-level tensors:

    shortest-path / spatial position
    attention bias
    degree encodings
    edge/path encodings
    padding masks

To use a Graphormer teacher with this project, the recommended approach is to
wrap it with the same interface as ClassicGNN:

    model(graph, num_nodes_per_graph) -> [batch_size, num_flows]

If that wrapper rebuilds the required Graphormer inputs from the current
DGLGraph, then explain_graphsvx_dgl.py can also explain the Graphormer model.


Recommended Checks
------------------

Before interpreting explanations:

    1. Verify the trained checkpoint exists.
    2. Evaluate the checkpoint and confirm MAE is reasonable.
    3. Run explanations with several seeds.
    4. Increase --num_samples and check whether top nodes remain stable.
    5. Prefer explanations with high weighted_r2.

For paper writing, describe the explainer as:

    GraphSVX-inspired
    GraphSVX-style
    DGL adaptation of the GraphSVX/Shapley idea

Do not claim it is a direct import of the original PyG GraphSVX implementation.


Git / Checkpoint Notes
----------------------

Large model checkpoints should usually not be committed to GitHub. The current
recommended .gitignore excludes:

    *.pth
    *.pt
    asd_exp/exp2/model_weights/
    asd_exp/method/model_weights/

If checkpoints must be published, consider Git LFS or GitHub Releases.
