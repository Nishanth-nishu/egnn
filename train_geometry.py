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

DATA SPLIT: 80:10:10 (train:val:test)
- Train: Optimization
- Val: Model selection (best val distance loss)
- Test: Final generalization assessment

LOSS FUNCTION: SE(3)-Invariant Distance Loss (Uni-Mol style)
- Scale: Å² (squared distances)

EVALUATION METRICS:
- RMSD (Å) - structure quality
- Pearson/Spearman correlation - ranking quality
- MAE, MSE, R² - regression quality

MODEL SELECTION: Best validation distance loss (primary signal)
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
    """Track and visualize training metrics."""
    
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
        
        self.distributions['val_rmsd_per_mol'] = val_metrics.get('rmsd_per_mol', [])
        self.distributions['val_dist_error_per_pair'] = val_metrics.get('dist_errors', [])
        
        if 'bond_lengths' in val_metrics:
            bl = val_metrics['bond_lengths']
            if len(bl) > 0:
                self.geometry_metrics['bond_length_mean'].append(np.mean(bl))
                self.geometry_metrics['bond_length_std'].append(np.std(bl))
            else:
                self.geometry_metrics['bond_length_mean'].append(np.nan)
                self.geometry_metrics['bond_length_std'].append(np.nan)
    
    def save_loss_curves(self):
        """Plot train vs val distance loss."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = self.history['epoch']
        
        ax.plot(epochs, self.history['train_dist_loss'], 'o-', label='Train', alpha=0.7, linewidth=2)
        ax.plot(epochs, self.history['val_dist_loss'], 's-', label='Val', alpha=0.7, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Distance Loss (Å²)', fontsize=12)
        ax.set_title('Training vs Validation Distance Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.save_dir / 'loss_curves.png'}")
    
    def save_rmsd_curves(self):
        """Plot train vs val RMSD."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = self.history['epoch']
        
        ax.plot(epochs, self.history['train_rmsd'], 'o-', label='Train', alpha=0.7, linewidth=2)
        ax.plot(epochs, self.history['val_rmsd'], 's-', label='Val', alpha=0.7, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('RMSD (Å)', fontsize=12)
        ax.set_title('Training vs Validation RMSD', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'rmsd_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.save_dir / 'rmsd_curves.png'}")
    
    def save_combined_curves(self):
        """Plot all metrics in one figure."""
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
        
        # Overfitting Check
        ax = axes[1, 0]
        gap = np.array(self.history['val_rmsd']) - np.array(self.history['train_rmsd'])
        ax.plot(epochs, gap, 'o-', color='red', alpha=0.7)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val RMSD - Train RMSD (Å)')
        ax.set_title('Overfitting Check')
        ax.grid(alpha=0.3)
        
        # Loss Generalization
        ax = axes[1, 1]
        loss_gap = np.array(self.history['val_dist_loss']) - np.array(self.history['train_dist_loss'])
        ax.plot(epochs, loss_gap, 'o-', color='orange', alpha=0.7)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val Loss - Train Loss (Å²)')
        ax.set_title('Loss Generalization Gap')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves_combined.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.save_dir / 'training_curves_combined.png'}")
    
    def save_distributions(self, epoch):
        """Plot and save error distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # RMSD distribution
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
        print(f"✓ Saved: {self.save_dir / f'distributions_epoch_{epoch}.png'}")
    
    def save_geometry_metrics(self):
        """Plot geometry sanity metrics."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        epochs = self.history['epoch']
        means = self.geometry_metrics['bond_length_mean']
        stds = self.geometry_metrics['bond_length_std']
        
        if len(means) > 0:
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
        print(f"✓ Saved: {self.save_dir / 'geometry_sanity.png'}")
    
    def save_csv(self):
        """Save metrics to CSV."""
        df = pd.DataFrame(self.history)
        df.to_csv(self.save_dir / 'metrics.csv', index=False)
        print(f"✓ Saved: {self.save_dir / 'metrics.csv'}")
    
    def save_summary(self):
        """Save summary statistics."""
        summary = {
            'best_val_dist_loss': float(np.min(self.history['val_dist_loss'])),
            'best_val_dist_loss_epoch': int(np.argmin(self.history['val_dist_loss'])),
            'best_val_rmsd': float(np.min(self.history['val_rmsd'])),
            'best_val_rmsd_epoch': int(np.argmin(self.history['val_rmsd'])),
            'final_train_rmsd': float(self.history['train_rmsd'][-1]),
            'final_val_rmsd': float(self.history['val_rmsd'][-1]),
            'final_train_loss': float(self.history['train_dist_loss'][-1]),
            'final_val_loss': float(self.history['val_dist_loss'][-1]),
        }
        
        if len(self.distributions['val_rmsd_per_mol']) > 0:
            summary['val_rmsd_p90'] = float(np.percentile(self.distributions['val_rmsd_per_mol'], 90))
        
        with open(self.save_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved: {self.save_dir / 'summary.json'}")


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
    """Build fully connected edges within each molecule."""
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
    """SE(3)-invariant distance loss (Uni-Mol style)."""
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
    """Compute bond lengths for geometry sanity checks."""
    if edge_index.shape[1] == 0:
        return []
    
    src, dst = edge_index
    bonds = pos[src] - pos[dst]
    lengths = torch.norm(bonds, dim=1)
    
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
    """Train for one epoch."""
    model.train()
    total_dist_loss = 0.0
    total_rmsd = 0.0
    num_batches = 0
    num_mols = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        x = batch['x'].to(device)
        pos = batch['pos'].to(device)
        mask = batch['mask'].to(device)

        x_flat, pos_flat, batch_idx = flatten_batch_with_batch_idx(x, pos, mask, device)
        
        if x_flat.shape[0] == 0:
            continue

        edge_index = build_edge_index_batch(mask, device)
        pos_noisy = add_noise(pos_flat, noise_std)
        
        pos_pred = model(x_flat, pos_noisy, edge_index)
        loss = distance_loss(pos_pred, pos_flat, batch_idx, normalize_by_size)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        total_dist_loss += loss.item()
        num_batches += 1
        
        with torch.no_grad():
            if compute_clean_rmsd:
                pos_pred_clean = model(x_flat, pos_flat, edge_index)
            else:
                pos_pred_clean = pos_pred
            
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
    """Evaluate model with comprehensive metrics."""
    model.eval()
    total_rmsd = 0.0
    total_dist_loss = 0.0
    num_mols = 0
    num_batches = 0
    
    rmsd_per_mol = []
    dist_errors_all = []
    bond_lengths_all = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            x = batch['x'].to(device)
            pos = batch['pos'].to(device)
            mask = batch['mask'].to(device)

            x_flat, pos_flat, batch_idx = flatten_batch_with_batch_idx(x, pos, mask, device)
            
            if x_flat.shape[0] == 0:
                continue

            edge_index = build_edge_index_batch(mask, device)
            pos_pred = model(x_flat, pos_flat, edge_index)

            if return_distributions:
                dist_loss, dist_errors = distance_loss(pos_pred, pos_flat, batch_idx, return_details=True)
                dist_errors_all.extend(dist_errors)
            else:
                dist_loss = distance_loss(pos_pred, pos_flat, batch_idx)
            
            total_dist_loss += dist_loss.item()
            num_batches += 1

            for mol_id in batch_idx.unique():
                mol_mask = batch_idx == mol_id
                n_atoms = mol_mask.sum().item()
                
                if n_atoms < 3:
                    continue
                
                pred_mol = pos_pred[mol_mask]
                true_mol = pos_flat[mol_mask]
                
                rmsd = compute_rmsd(pred_mol, true_mol)
                
                if torch.isfinite(rmsd):
                    rmsd_val = rmsd.item()
                    total_rmsd += rmsd_val
                    num_mols += 1
                    if return_distributions:
                        rmsd_per_mol.append(rmsd_val)
                
                if return_distributions:
                    global_mask = batch_idx == mol_id
                    mol_edge_mask = (batch_idx[edge_index[0]] == mol_id)
                    mol_edges_global = edge_index[:, mol_edge_mask]
                    
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
# TEST SET EVALUATION
# ============================================================================

def evaluate_test_set(model, test_loader, device, save_dir='metrics'):
    """Comprehensive test set evaluation with visualizations."""
    print("\n" + "="*70)
    print("TEST SET EVALUATION - GEOMETRY RECONSTRUCTION")
    print("="*70)
    
    save_dir = Path(save_dir)
    
    # Get detailed metrics
    test_metrics = evaluate(model, test_loader, device, return_distributions=True)
    
    rmsd_per_mol = test_metrics['rmsd_per_mol']
    
    # Compute geometry reconstruction statistics
    test_stats = {
        'test_dist_loss': test_metrics['dist_loss'],
        'test_rmsd_mean': test_metrics['rmsd'],
        'test_rmsd_median': float(np.median(rmsd_per_mol)),
        'test_rmsd_std': float(np.std(rmsd_per_mol)),
        'test_rmsd_mae': test_metrics['rmsd'],  # Mean absolute error is same as mean RMSD
        'test_rmsd_p10': float(np.percentile(rmsd_per_mol, 10)),
        'test_rmsd_p25': float(np.percentile(rmsd_per_mol, 25)),
        'test_rmsd_p50': float(np.percentile(rmsd_per_mol, 50)),
        'test_rmsd_p75': float(np.percentile(rmsd_per_mol, 75)),
        'test_rmsd_p90': float(np.percentile(rmsd_per_mol, 90)),
        'test_rmsd_p95': float(np.percentile(rmsd_per_mol, 95)),
        'test_rmsd_p99': float(np.percentile(rmsd_per_mol, 99)),
        'test_rmsd_min': float(np.min(rmsd_per_mol)),
        'test_rmsd_max': float(np.max(rmsd_per_mol)),
        'num_molecules': len(rmsd_per_mol),
    }
    
    # Calculate percentage of molecules below certain thresholds
    below_0_5 = (np.array(rmsd_per_mol) < 0.5).sum() / len(rmsd_per_mol) * 100
    below_1_0 = (np.array(rmsd_per_mol) < 1.0).sum() / len(rmsd_per_mol) * 100
    below_2_0 = (np.array(rmsd_per_mol) < 2.0).sum() / len(rmsd_per_mol) * 100
    
    test_stats['pct_below_0.5A'] = float(below_0_5)
    test_stats['pct_below_1.0A'] = float(below_1_0)
    test_stats['pct_below_2.0A'] = float(below_2_0)
    
    # Save test statistics
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(test_stats, f, indent=2)
    print(f"✓ Saved: {save_dir / 'test_results.json'}")
    
    # Visualizations - 2x3 grid focused on RMSD distribution
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. RMSD Distribution Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(rmsd_per_mol, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax1.axvline(np.median(rmsd_per_mol), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(rmsd_per_mol):.3f} Å')
    ax1.axvline(np.mean(rmsd_per_mol), color='green', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(rmsd_per_mol):.3f} Å')
    ax1.axvline(test_stats['test_rmsd_p90'], color='orange', linestyle='--', linewidth=2,
               label=f'P90: {test_stats["test_rmsd_p90"]:.3f} Å')
    ax1.set_xlabel('RMSD (Å)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Test Set RMSD Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # 2. Box Plot with statistics
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot(rmsd_per_mol, vert=True, patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['means'][0].set_marker('D')
    bp['means'][0].set_markerfacecolor('red')
    bp['means'][0].set_markersize(8)
    ax2.set_ylabel('RMSD (Å)', fontsize=11)
    ax2.set_title('RMSD Box Plot', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(['Test Set'])
    ax2.grid(alpha=0.3, axis='y')
    ax2.text(1.15, np.median(rmsd_per_mol), f'Median\n{np.median(rmsd_per_mol):.3f}Å', 
             fontsize=9, va='center')
    
    # 3. Percentiles Bar Chart
    ax3 = fig.add_subplot(gs[0, 2])
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    values = [test_stats[f'test_rmsd_p{p}'] for p in percentiles]
    colors = ['#1f77b4'] * 3 + ['#ff7f0e'] * 2 + ['#d62728'] * 2
    bars = ax3.bar([f'P{p}' for p in percentiles], values, alpha=0.7, color=colors, edgecolor='black')
    ax3.set_xlabel('Percentile', fontsize=11)
    ax3.set_ylabel('RMSD (Å)', fontsize=11)
    ax3.set_title('RMSD Percentiles', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Cumulative Distribution Function
    ax4 = fig.add_subplot(gs[1, 0])
    sorted_rmsd = np.sort(rmsd_per_mol)
    cumulative = np.arange(1, len(sorted_rmsd) + 1) / len(sorted_rmsd) * 100
    ax4.plot(sorted_rmsd, cumulative, linewidth=2, color='purple')
    ax4.axhline(50, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    ax4.axhline(90, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    ax4.axhline(95, color='darkred', linestyle='--', alpha=0.5, label='95th percentile')
    ax4.set_xlabel('RMSD (Å)', fontsize=11)
    ax4.set_ylabel('Cumulative Percentage (%)', fontsize=11)
    ax4.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    
    # 5. Key Metrics Summary
    ax5 = fig.add_subplot(gs[1, 1])
    metrics_names = ['Mean', 'Median', 'P90', 'P95']
    metrics_values = [test_stats['test_rmsd_mean'], test_stats['test_rmsd_median'],
                     test_stats['test_rmsd_p90'], test_stats['test_rmsd_p95']]
    colors_metrics = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    bars = ax5.bar(metrics_names, metrics_values, alpha=0.7, color=colors_metrics, edgecolor='black')
    ax5.set_ylabel('RMSD (Å)', fontsize=11)
    ax5.set_title('Key Reconstruction Metrics', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 6. Success Rate at Different Thresholds
    ax6 = fig.add_subplot(gs[1, 2])
    thresholds = [0.5, 1.0, 1.5, 2.0]
    success_rates = [
        (np.array(rmsd_per_mol) < t).sum() / len(rmsd_per_mol) * 100 
        for t in thresholds
    ]
    bars = ax6.bar([f'<{t}Å' for t in thresholds], success_rates, alpha=0.7, 
                   color='seagreen', edgecolor='black')
    ax6.set_ylabel('Success Rate (%)', fontsize=11)
    ax6.set_xlabel('RMSD Threshold', fontsize=11)
    ax6.set_title('Reconstruction Success Rate', fontsize=12, fontweight='bold')
    ax6.set_ylim([0, 100])
    ax6.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, success_rates):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.savefig(save_dir / 'test_evaluation_comprehensive.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir / 'test_evaluation_comprehensive.png'}")
    
    # Print results
    print("\nTest Set Results - Geometry Reconstruction:")
    print("-" * 70)
    print(f"Number of Molecules: {test_stats['num_molecules']}")
    print(f"\nDistance Loss:      {test_stats['test_dist_loss']:.6f} Å²")
    print(f"\nRMSD Statistics:")
    print(f"  Mean (MAE):       {test_stats['test_rmsd_mean']:.4f} Å")
    print(f"  Median:           {test_stats['test_rmsd_median']:.4f} Å")
    print(f"  Std Dev:          {test_stats['test_rmsd_std']:.4f} Å")
    print(f"  Min:              {test_stats['test_rmsd_min']:.4f} Å")
    print(f"  Max:              {test_stats['test_rmsd_max']:.4f} Å")
    print(f"\nPercentiles:")
    print(f"  P10:              {test_stats['test_rmsd_p10']:.4f} Å")
    print(f"  P25:              {test_stats['test_rmsd_p25']:.4f} Å")
    print(f"  P50 (Median):     {test_stats['test_rmsd_p50']:.4f} Å")
    print(f"  P75:              {test_stats['test_rmsd_p75']:.4f} Å")
    print(f"  P90:              {test_stats['test_rmsd_p90']:.4f} Å")
    print(f"  P95:              {test_stats['test_rmsd_p95']:.4f} Å")
    print(f"  P99:              {test_stats['test_rmsd_p99']:.4f} Å")
    print(f"\nSuccess Rates:")
    print(f"  RMSD < 0.5 Å:     {test_stats['pct_below_0.5A']:.1f}%")
    print(f"  RMSD < 1.0 Å:     {test_stats['pct_below_1.0A']:.1f}%")
    print(f"  RMSD < 2.0 Å:     {test_stats['pct_below_2.0A']:.1f}%")
    print("="*70)
    
    return test_stats


# ============================================================================
# CHECKPOINT STRATEGIES
# ============================================================================

class CheckpointManager:
    """
    Manage multiple checkpoint strategies:
    1. Best validation distance loss (primary training signal)
    2. EMA-smoothed validation loss (more stable)
    3. Best validation RMSD (quality metric)
    """
    
    def __init__(self, ema_alpha=0.9):
        self.ema_alpha = ema_alpha
        self.ema_val_loss = None
        
        self.best_val_dist_loss = float('inf')
        self.best_ema_val_loss = float('inf')
        self.best_val_rmsd = float('inf')
        
        self.best_epochs = {
            'val_dist_loss': -1,
            'ema_val_loss': -1,
            'val_rmsd': -1
        }
    
    def update_ema(self, current_val_loss):
        """Update exponential moving average of validation loss."""
        if self.ema_val_loss is None:
            self.ema_val_loss = current_val_loss
        else:
            self.ema_val_loss = (self.ema_alpha * current_val_loss + 
                                (1 - self.ema_alpha) * self.ema_val_loss)
        return self.ema_val_loss
    
    def should_save(self, epoch, val_dist_loss, val_rmsd):
        """
        Check which checkpoints should be saved.
        
        Returns:
            dict: Keys are checkpoint types that should be saved
        """
        save_checkpoints = {}
        
        # 1. Best validation distance loss
        if val_dist_loss < self.best_val_dist_loss:
            self.best_val_dist_loss = val_dist_loss
            self.best_epochs['val_dist_loss'] = epoch
            save_checkpoints['val_dist_loss'] = True
        
        # 2. Best EMA validation loss
        ema_loss = self.update_ema(val_dist_loss)
        if ema_loss < self.best_ema_val_loss:
            self.best_ema_val_loss = ema_loss
            self.best_epochs['ema_val_loss'] = epoch
            save_checkpoints['ema_val_loss'] = True
        
        # 3. Best validation RMSD
        if val_rmsd < self.best_val_rmsd:
            self.best_val_rmsd = val_rmsd
            self.best_epochs['val_rmsd'] = epoch
            save_checkpoints['val_rmsd'] = True
        
        return save_checkpoints, ema_loss
    
    def save_checkpoint(self, checkpoint_type, epoch, model, optimizer, train_metrics, val_metrics, config):
        """Save checkpoint for specific strategy."""
        filenames = {
            'val_dist_loss': 'best_by_val_distance_loss.pt',
            'ema_val_loss': 'best_by_EMA_based.pt',
            'val_rmsd': 'best_val_rmsd.pt'
        }
        
        checkpoint = {
            'epoch': epoch,
            'checkpoint_type': checkpoint_type,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config,
            'ema_val_loss': self.ema_val_loss if checkpoint_type == 'ema_val_loss' else None,
        }
        
        torch.save(checkpoint, filenames[checkpoint_type])
        return filenames[checkpoint_type]
    
    def get_summary(self):
        """Get summary of all checkpoints."""
        return {
            'best_val_dist_loss': {
                'value': self.best_val_dist_loss,
                'epoch': self.best_epochs['val_dist_loss']
            },
            'best_ema_val_loss': {
                'value': self.best_ema_val_loss,
                'epoch': self.best_epochs['ema_val_loss']
            },
            'best_val_rmsd': {
                'value': self.best_val_rmsd,
                'epoch': self.best_epochs['val_rmsd']
            }
        }


def compare_checkpoints_on_test(test_loader, device, save_dir='metrics'):
    """
    Load all three checkpoints and compare on test set.
    """
    print("\n" + "="*70)
    print("COMPARING ALL CHECKPOINT STRATEGIES ON TEST SET")
    print("="*70)
    
    save_dir = Path(save_dir)
    
    checkpoint_files = {
        'Val Dist Loss': 'best_by_val_distance_loss.pt',
        'EMA Val Loss': 'best_by_EMA_based.pt',
        'Val RMSD': 'best_val_rmsd.pt'
    }
    
    results = {}
    
    for strategy_name, filename in checkpoint_files.items():
        if not Path(filename).exists():
            print(f"⚠️  {filename} not found, skipping...")
            continue
        
        print(f"\n{'='*70}")
        print(f"Evaluating: {strategy_name}")
        print(f"File: {filename}")
        print(f"{'='*70}")
        
        # Load checkpoint
        checkpoint = torch.load(filename, map_location=device)
        
        # Create model
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
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded from epoch {checkpoint['epoch']}")
        if 'ema_val_loss' in checkpoint and checkpoint['ema_val_loss'] is not None:
            print(f"EMA val loss at save: {checkpoint['ema_val_loss']:.6f} Å²")
        
        # Evaluate on test set
        test_metrics = evaluate(model, test_loader, device, return_distributions=True)
        
        results[strategy_name] = {
            'epoch': checkpoint['epoch'],
            'test_dist_loss': test_metrics['dist_loss'],
            'test_rmsd_mean': test_metrics['rmsd'],
            'test_rmsd_median': float(np.median(test_metrics['rmsd_per_mol'])),
            'test_rmsd_p90': float(np.percentile(test_metrics['rmsd_per_mol'], 90)),
            'test_rmsd_p95': float(np.percentile(test_metrics['rmsd_per_mol'], 95)),
            'rmsd_distribution': test_metrics['rmsd_per_mol']
        }
        
        print(f"Test Distance Loss: {results[strategy_name]['test_dist_loss']:.6f} Å²")
        print(f"Test RMSD Mean:     {results[strategy_name]['test_rmsd_mean']:.4f} Å")
        print(f"Test RMSD Median:   {results[strategy_name]['test_rmsd_median']:.4f} Å")
        print(f"Test RMSD P90:      {results[strategy_name]['test_rmsd_p90']:.4f} Å")
    
    # Save comparison
    with open(save_dir / 'checkpoint_comparison.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for k, v in results.items():
            json_results[k] = {key: (val if not isinstance(val, list) else len(val)) 
                              for key, val in v.items() if key != 'rmsd_distribution'}
        json.dump(json_results, f, indent=2)
    print(f"\n✓ Saved: {save_dir / 'checkpoint_comparison.json'}")
    
    # Visualization comparing all three
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    strategies = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. Test Metrics Comparison - Bar Chart
    ax = axes[0, 0]
    metrics_to_compare = ['test_dist_loss', 'test_rmsd_mean', 'test_rmsd_p90']
    x = np.arange(len(strategies))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_compare):
        values = [results[s][metric] for s in strategies]
        ax.bar(x + i*width, values, width, label=metric.replace('test_', '').replace('_', ' ').title(),
               alpha=0.8, color=colors[i])
    
    ax.set_xlabel('Checkpoint Strategy', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Test Metrics Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # 2. RMSD Distribution Comparison
    ax = axes[0, 1]
    for i, strategy in enumerate(strategies):
        rmsd_dist = results[strategy]['rmsd_distribution']
        ax.hist(rmsd_dist, bins=50, alpha=0.5, label=strategy, color=colors[i], edgecolor='black')
    ax.set_xlabel('RMSD (Å)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('RMSD Distribution Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. CDF Comparison
    ax = axes[1, 0]
    for i, strategy in enumerate(strategies):
        rmsd_dist = np.sort(results[strategy]['rmsd_distribution'])
        cumulative = np.arange(1, len(rmsd_dist) + 1) / len(rmsd_dist) * 100
        ax.plot(rmsd_dist, cumulative, linewidth=2, label=strategy, color=colors[i])
    ax.set_xlabel('RMSD (Å)', fontsize=11)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=11)
    ax.set_title('Cumulative Distribution Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Percentile Comparison
    ax = axes[1, 1]
    percentiles_labels = ['Mean', 'Median', 'P90', 'P95']
    x = np.arange(len(percentiles_labels))
    width = 0.25
    
    for i, strategy in enumerate(strategies):
        values = [
            results[strategy]['test_rmsd_mean'],
            results[strategy]['test_rmsd_median'],
            results[strategy]['test_rmsd_p90'],
            results[strategy]['test_rmsd_p95']
        ]
        ax.bar(x + i*width, values, width, label=strategy, alpha=0.8, color=colors[i])
    
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('RMSD (Å)', fontsize=11)
    ax.set_title('Percentile Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(percentiles_labels)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'checkpoint_strategies_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir / 'checkpoint_strategies_comparison.png'}")
    
    # Print comparison table
    print("\n" + "="*70)
    print("CHECKPOINT STRATEGY COMPARISON - SUMMARY")
    print("="*70)
    print(f"{'Strategy':<20} {'Epoch':<8} {'Test Loss':<12} {'RMSD Mean':<12} {'RMSD P90':<12}")
    print("-"*70)
    for strategy in strategies:
        r = results[strategy]
        print(f"{strategy:<20} {r['epoch']:<8} {r['test_dist_loss']:<12.6f} {r['test_rmsd_mean']:<12.4f} {r['test_rmsd_p90']:<12.4f}")
    print("="*70)
    
    # Recommend best strategy
    best_by_loss = min(strategies, key=lambda s: results[s]['test_dist_loss'])
    best_by_rmsd = min(strategies, key=lambda s: results[s]['test_rmsd_mean'])
    best_by_p90 = min(strategies, key=lambda s: results[s]['test_rmsd_p90'])
    
    print("\nRECOMMENDATIONS:")
    print(f"  Best Test Distance Loss:  {best_by_loss}")
    print(f"  Best Test RMSD Mean:      {best_by_rmsd}")
    print(f"  Best Test RMSD P90:       {best_by_p90}")
    print("="*70)
    
    return results


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
    normalize_by_size = False
    compute_clean_rmsd = True
    ema_alpha = 0.9  # EMA smoothing factor (higher = more weight to recent)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("EGNN 3D GEOMETRY TRAINING - MULTI-CHECKPOINT STRATEGY")
    print("="*70)
    print(f"Device: {device}")
    print(f"Data Split: 80% train, 10% val, 10% test")
    print(f"Checkpoint Strategies:")
    print(f"  1. Best Validation Distance Loss (primary)")
    print(f"  2. Best EMA Validation Loss (α={ema_alpha}, stable)")
    print(f"  3. Best Validation RMSD (quality)")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}, LR: {lr}, Noise: {noise_std} Å")
    print("="*70)
    
    # Initialize metrics tracker and checkpoint manager
    metrics_tracker = MetricsTracker(save_dir='metrics')
    checkpoint_manager = CheckpointManager(ema_alpha=ema_alpha)
    
    # Load dataset with 80:10:10 split
    print("\nLoading dataset...")
    dataset = QM9Text3DDataset('/scratch/nishanth.r/egnn/data/qm9_100k.jsonl', atom_feature_dim=10)
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
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
    
    # Training loop configuration
    config = {
        'batch_size': batch_size,
        'lr': lr,
        'noise_std': noise_std,
        'normalize_by_size': normalize_by_size,
        'ema_alpha': ema_alpha,
        'seed': 42
    }
    
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, noise_std, 
                                    normalize_by_size, compute_clean_rmsd)
        
        # Validate
        return_dist = (epoch % 10 == 0) or (epoch == num_epochs - 1)
        val_metrics = evaluate(model, val_loader, device, return_distributions=return_dist)
        
        # Update metrics
        metrics_tracker.update(epoch, train_metrics, val_metrics)

        # Display
        print(f"Train Loss: {train_metrics['dist_loss']:.6f} Å² | Train RMSD: {train_metrics['rmsd']:.4f} Å")
        print(f"Val Loss:   {val_metrics['dist_loss']:.6f} Å² | Val RMSD:   {val_metrics['rmsd']:.4f} Å")
        
        # Check which checkpoints should be saved
        save_checkpoints, ema_loss = checkpoint_manager.should_save(
            epoch, val_metrics['dist_loss'], val_metrics['rmsd']
        )
        
        print(f"EMA Val Loss: {ema_loss:.6f} Å²")
        
        # Save checkpoints
        saved_files = []
        if 'val_dist_loss' in save_checkpoints:
            filename = checkpoint_manager.save_checkpoint(
                'val_dist_loss', epoch, model, optimizer, train_metrics, val_metrics, config
            )
            saved_files.append(f"Val Dist Loss → {filename}")
        
        if 'ema_val_loss' in save_checkpoints:
            filename = checkpoint_manager.save_checkpoint(
                'ema_val_loss', epoch, model, optimizer, train_metrics, val_metrics, config
            )
            saved_files.append(f"EMA Val Loss → {filename}")
        
        if 'val_rmsd' in save_checkpoints:
            filename = checkpoint_manager.save_checkpoint(
                'val_rmsd', epoch, model, optimizer, train_metrics, val_metrics, config
            )
            saved_files.append(f"Val RMSD → {filename}")
        
        if saved_files:
            print("✓ SAVED CHECKPOINTS:")
            for f in saved_files:
                print(f"  • {f}")
        
        # Save distributions
        if return_dist:
            metrics_tracker.save_distributions(epoch)
    
    # Save all training visualizations
    print("\n" + "="*70)
    print("SAVING TRAINING VISUALIZATIONS")
    print("="*70)
    
    metrics_tracker.save_loss_curves()
    metrics_tracker.save_rmsd_curves()
    metrics_tracker.save_combined_curves()
    metrics_tracker.save_geometry_metrics()
    metrics_tracker.save_csv()
    metrics_tracker.save_summary()
    
    # Save checkpoint summary
    checkpoint_summary = checkpoint_manager.get_summary()
    with open('metrics/checkpoint_summary.json', 'w') as f:
        json.dump(checkpoint_summary, f, indent=2)
    print(f"✓ Saved: metrics/checkpoint_summary.json")
    
    print("\n" + "="*70)
    print("CHECKPOINT SUMMARY")
    print("="*70)
    print(f"Best Val Dist Loss:  {checkpoint_summary['best_val_dist_loss']['value']:.6f} Å² (epoch {checkpoint_summary['best_val_dist_loss']['epoch']})")
    print(f"Best EMA Val Loss:   {checkpoint_summary['best_ema_val_loss']['value']:.6f} Å² (epoch {checkpoint_summary['best_ema_val_loss']['epoch']})")
    print(f"Best Val RMSD:       {checkpoint_summary['best_val_rmsd']['value']:.4f} Å (epoch {checkpoint_summary['best_val_rmsd']['epoch']})")
    print("="*70)
    
    # COMPARE ALL THREE CHECKPOINTS ON TEST SET
    comparison_results = compare_checkpoints_on_test(test_loader, device, save_dir='metrics')
    
    # Load best overall model (by EMA, most stable) for detailed test evaluation
    print("\n" + "="*70)
    print("DETAILED TEST EVALUATION - EMA CHECKPOINT (MOST STABLE)")
    print("="*70)
    
    checkpoint = torch.load('best_by_EMA_based.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_stats = evaluate_test_set(model, test_loader, device, save_dir='metrics')
    
    print("\n" + "="*70)
    print("ALL COMPLETE!")
    print("="*70)
    print("\nSaved files:")
    print("\nCheckpoints (3 strategies):")
    print("  • best_by_val_distance_loss.pt - Primary training signal")
    print("  • best_by_EMA_based.pt - Most stable (recommended)")
    print("  • best_val_rmsd.pt - Best quality metric")
    print("\nTraining visualizations:")
    print("  • metrics/loss_curves.png - Train vs Val distance loss")
    print("  • metrics/rmsd_curves.png - Train vs Val RMSD")
    print("  • metrics/training_curves_combined.png - All training metrics")
    print("\nTest evaluation:")
    print("  • metrics/test_evaluation_comprehensive.png - Detailed test analysis")
    print("  • metrics/checkpoint_strategies_comparison.png - Compare all 3 models")
    print("\nData files:")
    print("  • metrics/metrics.csv - Training history")
    print("  • metrics/summary.json - Training summary")
    print("  • metrics/checkpoint_summary.json - Checkpoint epochs")
    print("  • metrics/checkpoint_comparison.json - Test comparison")
    print("  • metrics/test_results.json - Test set results")
    print("="*70)


if __name__ == '__main__':
    main()