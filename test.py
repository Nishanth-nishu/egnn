import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =========================
# IMPORT PROJECT MODULES
# =========================
from models.egnn_clean.egnn_clean import EGNN
from data.qm9_text3d_dataset import QM9Text3DDataset, collate_fn
from torch.utils.data import DataLoader


# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# BATCH UTILITIES (COPIED FROM TRAINING)
# =========================
def flatten_batch_with_batch_idx(x, pos, mask, device):
    B, N = mask.shape
    x_list, pos_list, batch_list = [], [], []

    for b in range(B):
        mol_mask = mask[b].bool()
        n_atoms = mol_mask.sum().item()
        if n_atoms == 0:
            continue

        x_list.append(x[b, mol_mask])
        pos_list.append(pos[b, mol_mask])
        batch_list.append(torch.full((n_atoms,), b, device=device, dtype=torch.long))

    if len(x_list) == 0:
        return (
            torch.empty((0, x.shape[-1]), device=device),
            torch.empty((0, 3), device=device),
            torch.empty((0,), device=device, dtype=torch.long),
        )

    return (
        torch.cat(x_list, dim=0),
        torch.cat(pos_list, dim=0),
        torch.cat(batch_list, dim=0),
    )


def build_edge_index_batch(mask, device):
    batch_size, _ = mask.shape
    edge_list = []
    node_offset = 0

    for b in range(batch_size):
        n_atoms = mask[b].sum().item()
        if n_atoms == 0:
            continue

        idx = torch.arange(node_offset, node_offset + n_atoms, device=device)
        row = idx.repeat(n_atoms)
        col = idx.repeat_interleave(n_atoms)

        edge_mask = row != col
        edge_list.append(torch.stack([row[edge_mask], col[edge_mask]]))
        node_offset += n_atoms

    if len(edge_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    return torch.cat(edge_list, dim=1)


# =========================
# MAIN VISUALIZATION
# =========================
def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # LOAD DATASET
    # -------------------------
    dataset = QM9Text3DDataset(
        "/scratch/nishanth.r/egnn/data/qm9_100k.jsonl",
        atom_feature_dim=10,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,      # IMPORTANT: 1 molecule for clean visualization
        shuffle=True,
        collate_fn=collate_fn,
    )

    # -------------------------
    # REBUILD MODEL (MUST MATCH TRAINING)
    # -------------------------
    model = EGNN(
        in_node_nf=10,
        hidden_nf=128,
        out_node_nf=128,
        in_edge_nf=0,
        n_layers=4,
        attention=True,
        normalize=True,
        device=device,
    ).to(device)

    # -------------------------
    # LOAD CHECKPOINT
    # -------------------------
    ckpt = torch.load("best_egnn_geometry.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Loaded model from epoch:", ckpt["epoch"])

    # -------------------------
    # GET ONE BATCH
    # -------------------------
    batch = next(iter(val_loader))

    x = batch["x"].to(device)
    pos = batch["pos"].to(device)
    mask = batch["mask"].to(device)

    x_flat, pos_flat, batch_idx = flatten_batch_with_batch_idx(x, pos, mask, device)
    edge_index = build_edge_index_batch(mask, device)

    # -------------------------
    # INFERENCE
    # -------------------------
    with torch.no_grad():
        pos_pred = model(x_flat, pos_flat, edge_index)

    # -------------------------
    # SELECT FIRST MOLECULE
    # -------------------------
    mol_id = batch_idx[0]
    mol_mask = batch_idx == mol_id

    P = pos_pred[mol_mask].cpu().numpy()
    Q = pos_flat[mol_mask].cpu().numpy()

    # -------------------------
    # VISUALIZATION
    # -------------------------
# -------------------------
# VISUALIZATION (CLUSTER-SAFE)
# -------------------------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c="green", s=50, label="True")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c="red", s=50, label="Predicted")

    ax.set_title("EGNN Geometry Prediction")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("egnn_prediction.png", dpi=300)
    print("Saved visualization to egnn_prediction.png")



if __name__ == "__main__":
    main()
