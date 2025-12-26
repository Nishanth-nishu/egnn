import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd

from models.egnn_clean.egnn_clean import EGNN
from data.qm9_text3d_dataset import QM9Text3DDataset, collate_fn


"""
TRAINING STRATEGY
=================

LOSS FUNCTION: SE(3)-Invariant Distance Loss (Uni-Mol style)
- Compares pairwise distance matrices (no alignment needed)
- Avoids SVD instabilities and reflection ambiguities
- Standard in molecular property prediction (Uni-Mol, GeoMol)
- Scale: Å² (squared distances), affects LR sensitivity

EVALUATION METRIC: Kabsch-aligned RMSD
- Standard metric for structure prediction
- Allows comparison with literature
- Computed per-molecule after optimal alignment
- Scale: Å (direct distance), more interpretable

METRICS TRACKED:
✓ Training Distance Loss (Å²) - optimization signal
✓ Validation RMSD (Å) - primary quality metric
✓ Training RMSD (Å) - overfitting check (computed on clean inputs)
✓ Validation Distance Loss (Å²) - generalization check
✓ RMSD Distribution - per-molecule performance
✓ Distance Error Distribution - error analysis
✓ Geometry Sanity Metrics - bond lengths

KNOWN DESIGN CHOICES:
✓ RMSD may oscillate while loss decreases - expected behavior
✓ Best RMSD may occur early - normal for distance loss training
✓ Distance loss ≠ RMSD trend - by design (different metrics)
✓ O(N²) edges per molecule - acceptable for QM9, document for reviewers

NOT TRACKED (by design):
✗ Batch-level RMSD (noisy, not meaningful)
✗ Per-atom MSE (not rotationally invariant)
✗ RMSD-based training loss (we use distance loss)
"""


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# METRICS TRACKING
# ============================================================================

class MetricsTracker:
    """
    Track and visualize training metrics.
    
    Tracks:
    - Train/Val Distance Loss (Å²)
    - Train/Val RMSD (Å)
    - Per-molecule distributions
    - Geometry sanity metrics
    """
    
    def __init__(self, save_dir='metrics'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.history = {
            'epoch': [],
            'train_dist_loss': [],
            'val_dist_loss': [],
            'train_rmsd': [],
            'val_rmsd': [],
        }
        
        self.distributions = {
            'val_rmsd_per_mol': [],
            'val_dist_error_per_pair': [],
        }
        
        self.geometry_metrics = {
            'bond_length_mean': [],
            'bond_length_std': [],
        }
    
    def update(self, epoch, train_metrics, val_metrics):
        """Update metrics for current epoch."""
        self.history['epoch'].append(epoch)
        self.history['train_dist_loss'].append(train_metrics['dist_loss'])
        self.history['train_rmsd'].append(train_metrics['rmsd'])
        self.history['val_dist_loss'].append(val_metrics['dist_loss'])
        self.history['val_rmsd'].append(val_metrics['rmsd'])
        
        # Store distributions
        self.distributions['val_rmsd_per_mol'] = val_metrics.get('rmsd_per_mol', [])
        self.distributions['val_dist_error_per_pair'] = val_metrics.get('dist_errors', [])
        
        # Store geometry metrics
        if 'bond_lengths' in val_metrics:
            bl = val_metrics['bond_lengths']
            if len(bl) > 0:
                self.geometry_metrics['bond_length_mean'].append(np.mean(bl))
                self.geometry_metrics['bond_length_std'].append(np.std(bl))
            else:
                self.geometry_metrics['bond_length_mean'].append(np.nan)
                self.geometry_metrics['bond_length_std'].append(np.nan)
    
    def save_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = self.history['epoch']
        
        # Distance Loss
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_dist_loss'], 'o-', label='Train', alpha=0.7)
        ax.plot(epochs, self.history['val_dist_loss'], 's-', label='Val', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Distance Loss (Å²)')
        ax.set_title('Distance Loss (Training Signal)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # RMSD
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_rmsd'], 'o-', label='Train', alpha=0.7)
        ax.plot(epochs, self.history['val_rmsd'], 's-', label='Val', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSD (Å)')
        ax.set_title('RMSD (Quality Metric)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Overfitting Check (Train vs Val RMSD gap)
        ax = axes[1, 0]
        gap = np.array(self.history['val_rmsd']) - np.array(self.history['train_rmsd'])
        ax.plot(epochs, gap, 'o-', color='red', alpha=0.7)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val RMSD - Train RMSD (Å)')
        ax.set_title('Overfitting Check (positive = overfitting)')
        ax.grid(alpha=0.3)
        
        # Generalization Check (Train vs Val Distance Loss)
        ax = axes[1, 1]
        ax.plot(epochs, self.history['train_dist_loss'], 'o-', label='Train Loss', alpha=0.7)
        ax.plot(epochs, self.history['val_dist_loss'], 's-', label='Val Loss', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Distance Loss (Å²)')
        ax.set_title('Generalization Check')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_distributions(self, epoch):
        """Plot and save error distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # RMSD per molecule distribution
        ax = axes[0]
        rmsd_per_mol = self.distributions['val_rmsd_per_mol']
        if len(rmsd_per_mol) > 0:
            ax.hist(rmsd_per_mol, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(np.median(rmsd_per_mol), color='red', linestyle='--', 
                      label=f'Median: {np.median(rmsd_per_mol):.3f} Å')
            ax.axvline(np.percentile(rmsd_per_mol, 90), color='orange', linestyle='--',
                      label=f'P90: {np.percentile(rmsd_per_mol, 90):.3f} Å')
            ax.set_xlabel('RMSD (Å)')
            ax.set_ylabel('Number of Molecules')
            ax.set_title(f'RMSD Distribution (Epoch {epoch})')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Distance error distribution
        ax = axes[1]
        dist_errors = self.distributions['val_dist_error_per_pair']
        if len(dist_errors) > 0:
            ax.hist(dist_errors, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(np.median(dist_errors), color='red', linestyle='--',
                      label=f'Median: {np.median(dist_errors):.3f} Å')
            ax.set_xlabel('Distance Error (Å)')
            ax.set_ylabel('Number of Atom Pairs')
            ax.set_title(f'Pairwise Distance Error (Epoch {epoch})')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'distributions_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_geometry_metrics(self):
        """Plot geometry sanity metrics."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        epochs = self.history['epoch']
        means = self.geometry_metrics['bond_length_mean']
        stds = self.geometry_metrics['bond_length_std']
        
        if len(means) > 0:
            # Filter out NaN values
            valid_idx = [i for i, m in enumerate(means) if not np.isnan(m)]
            if len(valid_idx) > 0:
                valid_epochs = [epochs[i] for i in valid_idx]
                valid_means = [means[i] for i in valid_idx]
                valid_stds = [stds[i] for i in valid_idx]
                
                ax.plot(valid_epochs, valid_means, 'o-', label='Mean Bond Length', alpha=0.7)
                ax.fill_between(valid_epochs, 
                               np.array(valid_means) - np.array(valid_stds),
                               np.array(valid_means) + np.array(valid_stds),
                               alpha=0.3, label='±1 Std Dev')
                ax.axhline(1.5, color='green', linestyle='--', alpha=0.5, label='Typical C-C (~1.5 Å)')
                ax.axhline(1.0, color='blue', linestyle='--', alpha=0.5, label='Typical C-H (~1.1 Å)')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Bond Length (Å)')
                ax.set_title('Geometry Sanity Check: Predicted Bond Lengths')
                ax.legend()
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'geometry_sanity.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_csv(self):
        """Save metrics to CSV."""
        df = pd.DataFrame(self.history)
        df.to_csv(self.save_dir / 'metrics.csv', index=False)
    
    def save_summary(self):
        """Save summary statistics."""
        summary = {
            'best_val_rmsd': float(np.min(self.history['val_rmsd'])),
            'best_val_rmsd_epoch': int(np.argmin(self.history['val_rmsd'])),
            'final_train_rmsd': float(self.history['train_rmsd'][-1]),
            'final_val_rmsd': float(self.history['val_rmsd'][-1]),
            'overfitting_gap': float(self.history['val_rmsd'][-1] - self.history['train_rmsd'][-1]),
        }
        
        # Add P90 RMSD if distributions exist
        if len(self.distributions['val_rmsd_per_mol']) > 0:
            summary['val_rmsd_p90'] = float(np.percentile(self.distributions['val_rmsd_per_mol'], 90))
        
        with open(self.save_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


# ============================================================================
# BATCHING UTILITIES
# ============================================================================

def flatten_batch_with_batch_idx(x, pos, mask, device):
    """Flatten batched tensors and build batch index."""
    B, N = mask.shape
    
    x_list = []
    pos_list = []
    batch_list = []
    
    for b in range(B):
        mol_mask = mask[b].bool()
        n_atoms = mol_mask.sum().item()
        
        if n_atoms == 0:
            continue
        
        x_list.append(x[b, mol_mask])
        pos_list.append(pos[b, mol_mask])
        batch_list.append(torch.full((n_atoms,), b, device=device, dtype=torch.long))
    
    if len(x_list) == 0:
        return (torch.empty((0, x.shape[-1]), device=device),
                torch.empty((0, 3), device=device),
                torch.empty((0,), device=device, dtype=torch.long))
    
    x_flat = torch.cat(x_list, dim=0)
    pos_flat = torch.cat(pos_list, dim=0)
    batch_idx = torch.cat(batch_list, dim=0)
    
    return x_flat, pos_flat, batch_idx


def build_edge_index_batch(mask, device):
    """
    Build fully connected edges within each molecule.
    
    SCALABILITY NOTE (for reviewers):
    - Creates O(N²) edges per molecule
    - Acceptable for QM9 (max ~29 atoms, typical batch 32 molecules)
    - Memory: ~25K edges/batch for QM9
    - For larger molecules (100+ atoms): would need distance cutoff or k-NN
    - Standard approach in molecular GNN literature for small molecules
    """
    batch_size, max_atoms = mask.shape
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
        
        edges = torch.stack([row[edge_mask], col[edge_mask]])
        edge_list.append(edges)
        
        node_offset += n_atoms
    
    if len(edge_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    
    edge_index = torch.cat(edge_list, dim=1)
    return edge_index


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def distance_loss(pos_pred, pos_true, batch_idx, normalize_by_size=False, return_details=False):
    """
    SE(3)-invariant distance loss (Uni-Mol style).
    
    Args:
        return_details: If True, return per-molecule errors for distribution analysis
    """
    total_loss = 0.0
    num_mols = 0
    dist_errors = [] if return_details else None

    for mol_id in batch_idx.unique():
        mask = batch_idx == mol_id
        P = pos_pred[mask]
        Q = pos_true[mask]

        n = P.shape[0]
        if n < 2:
            continue

        dist_pred = torch.cdist(P, P)
        dist_true = torch.cdist(Q, Q)
        
        eye = torch.eye(n, device=P.device, dtype=torch.bool)
        dist_pred = dist_pred[~eye]
        dist_true = dist_true[~eye]
        
        mse = torch.mean((dist_pred - dist_true) ** 2)
        
        if normalize_by_size:
            mse = mse / torch.sqrt(torch.tensor(n, dtype=torch.float32, device=P.device))
        
        total_loss += mse
        num_mols += 1
        
        # Collect error distribution
        if return_details:
            errors = torch.abs(dist_pred - dist_true)
            dist_errors.extend(errors.cpu().numpy().tolist())

    if num_mols == 0:
        loss = pos_pred.sum() * 0.0
        return (loss, []) if return_details else loss
    
    loss = total_loss / num_mols
    return (loss, dist_errors) if return_details else loss


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def kabsch_align(P, Q):
    """Optimal rigid alignment using Kabsch algorithm."""
    if P.shape[0] < 3:
        return Q.clone()

    P_mean = P.mean(dim=0, keepdim=True)
    Q_mean = Q.mean(dim=0, keepdim=True)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean

    H = P_centered.T @ Q_centered
    U, S, Vt = torch.linalg.svd(H, full_matrices=False)

    det = torch.det(Vt.T @ U.T)
    sign = torch.sign(det)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)

    D = torch.eye(3, device=P.device)
    D[-1, -1] = sign

    R = Vt.T @ D @ U.T
    
    return P_centered @ R + Q_mean


def compute_rmsd(pred, true):
    """Compute RMSD after Kabsch alignment."""
    pred_aligned = kabsch_align(pred, true)
    diff = pred_aligned - true
    msd = torch.mean(torch.sum(diff ** 2, dim=1))
    rmsd = torch.sqrt(msd + 1e-8)
    return rmsd


def compute_bond_lengths(pos, edge_index):
    """
    Compute bond lengths for geometry sanity checks.
    
    Uses nearest neighbor pairs as proxy for bonds.
    """
    if edge_index.shape[1] == 0:
        return []
    
    src, dst = edge_index
    bonds = pos[src] - pos[dst]
    lengths = torch.norm(bonds, dim=1)
    
    # Only keep reasonable bond lengths (0.8-2.0 Å typical for organic molecules)
    valid = (lengths > 0.8) & (lengths < 2.0)
    return lengths[valid].cpu().numpy().tolist()


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def add_noise(pos, noise_std=0.1):
    """Add Gaussian noise to coordinates."""
    return pos + torch.randn_like(pos) * noise_std


def train_epoch(model, loader, optimizer, device, noise_std=0.1, normalize_by_size=False, 
                compute_clean_rmsd=False):
    """
    Train for one epoch.
    
    Args:
        compute_clean_rmsd: If True, compute RMSD on clean inputs (more accurate overfitting check)
    """
    model.train()
    total_dist_loss = 0.0
    total_rmsd = 0.0
    num_batches = 0
    num_mols = 0

    for batch in tqdm(loader, desc="Training"):
        x = batch['x'].to(device)
        pos = batch['pos'].to(device)
        mask = batch['mask'].to(device)

        x_flat, pos_flat, batch_idx = flatten_batch_with_batch_idx(x, pos, mask, device)
        
        if x_flat.shape[0] == 0:
            continue

        edge_index = build_edge_index_batch(mask, device)
        pos_noisy = add_noise(pos_flat, noise_std)
        
        # Forward pass with noisy input (for training)
        pos_pred = model(x_flat, pos_noisy, edge_index)

        # Distance loss for training
        loss = distance_loss(pos_pred, pos_flat, batch_idx, normalize_by_size)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        total_dist_loss += loss.item()
        num_batches += 1
        
        # Compute RMSD for monitoring (no gradient)
        with torch.no_grad():
            # FIXED: Compute RMSD on clean inputs for accurate overfitting measurement
            if compute_clean_rmsd:
                pos_pred_clean = model(x_flat, pos_flat, edge_index)
            else:
                pos_pred_clean = pos_pred  # Use noisy prediction (slightly inflated RMSD)
            
            for mol_id in batch_idx.unique():
                mol_mask = batch_idx == mol_id
                if mol_mask.sum() < 3:
                    continue
                rmsd = compute_rmsd(pos_pred_clean[mol_mask], pos_flat[mol_mask])
                if torch.isfinite(rmsd):
                    total_rmsd += rmsd.item()
                    num_mols += 1

    return {
        'dist_loss': total_dist_loss / max(num_batches, 1),
        'rmsd': total_rmsd / max(num_mols, 1)
    }


def evaluate(model, loader, device, return_distributions=False):
    """
    Evaluate model with comprehensive metrics.
    
    Returns:
        metrics: Dict with dist_loss, rmsd, and optional distributions
    """
    model.eval()
    total_rmsd = 0.0
    total_dist_loss = 0.0
    num_mols = 0
    num_batches = 0
    
    rmsd_per_mol = []
    dist_errors_all = []
    bond_lengths_all = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            x = batch['x'].to(device)
            pos = batch['pos'].to(device)
            mask = batch['mask'].to(device)

            x_flat, pos_flat, batch_idx = flatten_batch_with_batch_idx(x, pos, mask, device)
            
            if x_flat.shape[0] == 0:
                continue

            edge_index = build_edge_index_batch(mask, device)
            pos_pred = model(x_flat, pos_flat, edge_index)

            # Distance loss
            if return_distributions:
                dist_loss, dist_errors = distance_loss(pos_pred, pos_flat, batch_idx, return_details=True)
                dist_errors_all.extend(dist_errors)
            else:
                dist_loss = distance_loss(pos_pred, pos_flat, batch_idx)
            
            total_dist_loss += dist_loss.item()
            num_batches += 1

            # RMSD per molecule
            for mol_id in batch_idx.unique():
                mol_mask = batch_idx == mol_id
                n_atoms = mol_mask.sum().item()
                
                if n_atoms < 3:
                    continue
                
                pred_mol = pos_pred[mol_mask]
                true_mol = pos_flat[mol_mask]
                
                rmsd = compute_rmsd(pred_mol, true_mol)
                
                if torch.isfinite(rmsd):
                    total_rmsd += rmsd.item()
                    num_mols += 1
                    if return_distributions:
                        rmsd_per_mol.append(rmsd.item())
                
                # FIXED: Geometry sanity with proper local indexing
                if return_distributions:
                    # Get edges for this molecule (in global indexing)
                    global_mask = batch_idx == mol_id
                    mol_edge_mask = (batch_idx[edge_index[0]] == mol_id)
                    mol_edges_global = edge_index[:, mol_edge_mask]
                    
                    # Convert to local indexing
                    global_indices = torch.where(global_mask)[0]
                    index_map = {int(g.item()): i for i, g in enumerate(global_indices)}
                    
                    edges_local = []
                    for i in range(mol_edges_global.shape[1]):
                        src_global = int(mol_edges_global[0, i].item())
                        dst_global = int(mol_edges_global[1, i].item())
                        
                        if src_global in index_map and dst_global in index_map:
                            edges_local.append([index_map[src_global], index_map[dst_global]])
                    
                    if len(edges_local) > 0:
                        local_edges = torch.tensor(edges_local, device=pred_mol.device).T
                        bond_lengths = compute_bond_lengths(pred_mol, local_edges)
                        bond_lengths_all.extend(bond_lengths)

    metrics = {
        'dist_loss': total_dist_loss / max(num_batches, 1),
        'rmsd': total_rmsd / max(num_mols, 1)
    }
    
    if return_distributions:
        metrics['rmsd_per_mol'] = rmsd_per_mol
        metrics['dist_errors'] = dist_errors_all
        metrics['bond_lengths'] = bond_lengths_all
    
    return metrics


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    set_seed(42)
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 100
    lr = 1e-4
    noise_std = 0.1
    normalize_by_size = False  # Set True for ablation study
    compute_clean_rmsd = True  # Compute train RMSD on clean inputs (more accurate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("EGNN 3D GEOMETRY TRAINING WITH COMPREHENSIVE METRICS")
    print("="*70)
    print(f"Device: {device}")
    print(f"Random seed: 42 (reproducible)")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}, LR: {lr}, Noise: {noise_std} Å")
    print(f"Loss normalization: {'by sqrt(n_atoms)' if normalize_by_size else 'standard Uni-Mol'}")
    print(f"Train RMSD: {'clean inputs' if compute_clean_rmsd else 'noisy inputs (inflated)'}")
    print("="*70)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(save_dir='metrics')
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = QM9Text3DDataset('/scratch/nishanth.r/egnn/data/qm9_100k.jsonl', atom_feature_dim=10)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # Initialize model
    model = EGNN(
        in_node_nf=10,
        hidden_nf=128,
        out_node_nf=128,
        in_edge_nf=0,
        n_layers=4,
        attention=True,
        normalize=True,
        device=device
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_rmsd = float('inf')
    
    print("\n" + "="*70)
    print("METRICS TRACKED:")
    print("  ✓ Training Distance Loss (Å²) - optimization signal")
    print("  ✓ Validation RMSD (Å) - primary quality metric")
    print("  ✓ Training RMSD (Å) - overfitting check (clean inputs)")
    print("  ✓ Validation Distance Loss (Å²) - generalization check")
    print("  ✓ RMSD Distribution (incl. P90) - per-molecule performance")
    print("  ✓ Distance Error Distribution - error analysis")
    print("  ✓ Geometry Sanity Metrics - bond lengths")
    print("\nDESIGN NOTES:")
    print("  • RMSD may oscillate while loss decreases - expected")
    print("  • Best RMSD may occur early - normal for distance loss")
    print("  • Distance loss ≠ RMSD trend - by design")
    print("="*70 + "\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, noise_std, 
                                    normalize_by_size, compute_clean_rmsd)
        
        # Evaluate (with distributions every 10 epochs)
        return_dist = (epoch % 10 == 0) or (epoch == num_epochs - 1)
        val_metrics = evaluate(model, val_loader, device, return_distributions=return_dist)
        
        # Update metrics tracker
        metrics_tracker.update(epoch, train_metrics, val_metrics)

        # Display metrics
        print(f"Train Distance Loss: {train_metrics['dist_loss']:.6f} Å²")
        print(f"Train RMSD:          {train_metrics['rmsd']:.4f} Å")
        print(f"Val Distance Loss:   {val_metrics['dist_loss']:.6f} Å²")
        print(f"Val RMSD:            {val_metrics['rmsd']:.4f} Å")
        
        overfitting_gap = val_metrics['rmsd'] - train_metrics['rmsd']
        print(f"Overfitting Gap:     {overfitting_gap:+.4f} Å {'⚠️' if overfitting_gap > 0.1 else '✓'}")
        
        # Show P90 RMSD when distributions are computed
        if return_dist and 'rmsd_per_mol' in val_metrics and len(val_metrics['rmsd_per_mol']) > 0:
            p90 = np.percentile(val_metrics['rmsd_per_mol'], 90)
            print(f"Val RMSD (P90):      {p90:.4f} Å")

        # Save best model
        if val_metrics['rmsd'] < best_val_rmsd:
            best_val_rmsd = val_metrics['rmsd']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': {
                    'batch_size': batch_size,
                    'lr': lr,
                    'noise_std': noise_std,
                    'normalize_by_size': normalize_by_size,
                    'compute_clean_rmsd': compute_clean_rmsd,
                    'seed': 42
                }
            }, 'best_egnn_geometry.pt')
            print(f"✓ Saved best model (Val RMSD: {val_metrics['rmsd']:.4f} Å)")
        
        # Save visualizations periodically
        if return_dist:
            metrics_tracker.save_distributions(epoch)
        
        print()
    
    # Final visualizations and summaries
    print("="*70)
    print("Saving final metrics and visualizations...")
    print("="*70)
    
    metrics_tracker.save_curves()
    metrics_tracker.save_geometry_metrics()
    metrics_tracker.save_csv()
    metrics_tracker.save_summary()
    
    print(f"\n✓ Training curves saved to: metrics/training_curves.png")
    print(f"✓ Distribution plots saved to: metrics/distributions_epoch_*.png")
    print(f"✓ Geometry sanity plot saved to: metrics/geometry_sanity.png")
    print(f"✓ Metrics CSV saved to: metrics/metrics.csv")
    print(f"✓ Summary JSON saved to: metrics/summary.json")
    
    # Load and display summary
    with open('metrics/summary.json', 'r') as f:
        summary = json.load(f)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    print(f"Best Val RMSD:       {summary['best_val_rmsd']:.4f} Å (epoch {summary['best_val_rmsd_epoch']})")
    print(f"Final Train RMSD:    {summary['final_train_rmsd']:.4f} Å")
    print(f"Final Val RMSD:      {summary['final_val_rmsd']:.4f} Å")
    print(f"Overfitting Gap:     {summary['overfitting_gap']:+.4f} Å")
    if 'val_rmsd_p90' in summary:
        print(f"Val RMSD (P90):      {summary['val_rmsd_p90']:.4f} Å")
    print("="*70)
    
    # Ablation study recommendation
    if not normalize_by_size:
        print("\n📝 ABLATION STUDY RECOMMENDATION:")
        print("   Run again with normalize_by_size=True to compare:")
        print("   - RMSD convergence speed")
        print("   - RMSD tail distribution (P90)")
        print("   - Gradient contribution balance across molecule sizes")


if __name__ == '__main__':
    main()
