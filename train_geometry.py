import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Clean imports - no redefinitions
from models.egnn_clean.egnn_clean import EGNN
from data.qm9_text3d_dataset import QM9Text3DDataset, collate_fn


"""
BATCHING STRATEGY OVERVIEW
===========================

Problem: DataLoader outputs batched data [B, N, F] but EGNN expects [num_nodes, F]

Solution: Flatten batch + build batch-aware edges in train_geometry.py

Flow:
    DataLoader → [B, N, F] with padding
         ↓
    flatten_batch_with_batch_idx() → [num_real_atoms, F] + batch_idx (CONSISTENT)
         ↓
    build_edge_index_batch() → [2, E] (edges only within molecules)
         ↓
    EGNN forward → [num_real_atoms, 3]

Key: All tensors (x, pos, batch_idx) are built from SAME loop - guaranteed alignment

RESEARCH DESIGN DECISIONS (for paper/reviewers)
===============================================

1. KABSCH ALIGNMENT
   - Uses sign-based reflection correction (no Python branching)
   - Differentiable everywhere except true degeneracy
   - Standard formulation from GeoDiff/Uni-Mol

2. FULLY CONNECTED EDGES (O(N²))
   - Acceptable for QM9 (max ~29 atoms)
   - Common practice in molecular GNN literature
   - For larger systems: would need distance cutoff or k-NN
   - MUST acknowledge as limitation in paper

3. PER-MOLECULE RMSD LOSS
   - Each molecule aligned independently (SE(3) invariance)
   - Prevents cross-molecule gradient contamination
   - Critical for proper learning of 3D geometry
"""
def unimol_distance_loss(pos_pred, pos_true, batch_idx):
    """
    Uni-Mol style SE(3)-invariant distance loss.
    No alignment. No SVD. No NaNs.
    """
    loss = 0.0
    num_mols = 0

    for mol_id in batch_idx.unique():
        mask = batch_idx == mol_id
        P = pos_pred[mask]
        Q = pos_true[mask]

        n = P.shape[0]
        if n < 2:
            continue

        # Pairwise distances
        d_pred = torch.cdist(P, P)
        d_true = torch.cdist(Q, Q)

        loss = loss + torch.mean((d_pred - d_true) ** 2)
        num_mols += 1

    if num_mols == 0:
        # keeps graph alive, zero gradient
        return pos_pred.sum() * 0.0

    return loss / num_mols


def kabsch_align(P, Q):
    # ---- SAFETY CHECK (CRITICAL) ----
    if P.shape[0] < 3:
        return Q.clone()

    P_mean = P.mean(dim=0, keepdim=True)
    Q_mean = Q.mean(dim=0, keepdim=True)

    P_centered = P - P_mean
    Q_centered = Q - Q_mean

    H = P_centered.T @ Q_centered

    # ---- SVD ----
    U, S, Vt = torch.linalg.svd(H, full_matrices=False)

    # ---- REFLECTION SAFE CORRECTION ----
    det = torch.det(Vt.T @ U.T)
    sign = torch.sign(det)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)

    D = torch.eye(3, device=P.device)
    D[-1, -1] = sign

    R = Vt.T @ D @ U.T
    return P_centered @ R + Q_mean


def rmsd_after_kabsch(pred, true):
    """
    Compute RMSD after Kabsch alignment.
    
    Args:
        pred: [N, 3] predicted coordinates
        true: [N, 3] true coordinates
    
    Returns:
        rmsd: Root mean squared deviation after alignment
    """
    pred_aligned = kabsch_align(pred, true)
    diff = pred_aligned - true
    msd = torch.mean(torch.sum(diff ** 2, dim=1))
    rmsd = torch.sqrt(msd + 1e-8)

    return rmsd


def flatten_batch_with_batch_idx(x, pos, mask, device):
    """
    CRITICAL FIX: Flatten batch and build batch_idx CONSISTENTLY.
    
    This ensures x_flat, pos_flat, and batch_idx are PERFECTLY ALIGNED
    by building all three from the SAME loop over molecules.
    
    Args:
        x: [B, N, F] node features
        pos: [B, N, 3] coordinates
        mask: [B, N] boolean mask
        device: torch device
    
    Returns:
        x_flat: [num_real_atoms, F]
        pos_flat: [num_real_atoms, 3]
        batch_idx: [num_real_atoms] molecule ID for each atom
    """
    B, N = mask.shape
    
    x_list = []
    pos_list = []
    batch_list = []
    
    for b in range(B):
        # Get mask for this molecule
        mol_mask = mask[b].bool()
        n_atoms = mol_mask.sum().item()
        
        if n_atoms == 0:
            continue
        
        # Extract atoms for this molecule
        x_list.append(x[b, mol_mask])
        pos_list.append(pos[b, mol_mask])
        batch_list.append(torch.full((n_atoms,), b, device=device, dtype=torch.long))
    
    # Concatenate all molecules
    x_flat = torch.cat(x_list, dim=0)
    pos_flat = torch.cat(pos_list, dim=0)
    batch_idx = torch.cat(batch_list, dim=0)
    
    return x_flat, pos_flat, batch_idx


def build_edge_index_batch(mask):
    """
    VECTORIZED edge construction for batched molecules.
    
    Builds fully connected edges within each molecule.
    Ensures edges ONLY exist within each molecule, NOT across batch boundaries.
    
    IMPORTANT NOTE FOR PAPER:
    - This creates O(N²) edges per molecule (fully connected)
    - For QM9 (max ~29 atoms): this is standard and acceptable
    - For larger molecules: consider distance cutoff or k-NN edges
    - Reviewers will ask about scalability - acknowledge this limitation
    
    Input:  mask [B, N] - boolean mask where True = real atom
    Output: edge_index [2, E] - edge indices in FLATTENED space
    """
    batch_size, max_atoms = mask.shape
    edge_list = []
    node_offset = 0
    
    for b in range(batch_size):
        n_atoms = mask[b].sum().item()
        if n_atoms == 0:
            continue
        
        # Vectorized fully connected edges for this molecule
        idx = torch.arange(node_offset, node_offset + n_atoms)
        row = idx.repeat(n_atoms)
        col = idx.repeat_interleave(n_atoms)
        edge_mask = row != col
        
        edges = torch.stack([row[edge_mask], col[edge_mask]])
        edge_list.append(edges)
        
        node_offset += n_atoms
    
    if len(edge_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    
    edge_index = torch.cat(edge_list, dim=1)
    return edge_index


def add_noise(pos, noise_std=0.1):
    """Add Gaussian noise to coordinates."""
    noise = torch.randn_like(pos) * noise_std
    return pos + noise


def molecule_aware_rmsd_loss(pos_pred, pos_true, batch_idx):
    """
    Compute RMSD loss with Kabsch alignment, averaged over molecules.
    
    CRITICAL: This computes per-molecule RMSD with proper alignment.
    Different molecules are unrelated coordinate frames - we must
    separate them and align each independently.
    
    Args:
        pos_pred: [num_atoms, 3] predicted coordinates
        pos_true: [num_atoms, 3] true coordinates
        batch_idx: [num_atoms] molecule ID for each atom
    
    Returns:
        avg_rmsd: Average RMSD across all molecules in batch
    """
    # CRITICAL ASSERTION: Check alignment
    assert pos_pred.shape == pos_true.shape, f"Shape mismatch: pred={pos_pred.shape}, true={pos_true.shape}"
    assert batch_idx.shape[0] == pos_pred.shape[0], f"Batch idx mismatch: {batch_idx.shape[0]} vs {pos_pred.shape[0]}"
    
    total_loss = 0.0
    num_mols = 0
    
    for mol_id in batch_idx.unique():
        mol_mask = batch_idx == mol_id
        n_atoms = mol_mask.sum().item()
        
        # Skip degenerate molecules
        if n_atoms < 2:
            continue
        
        pred_m = pos_pred[mol_mask]
        true_m = pos_true[mol_mask]
        
        # CRITICAL ASSERTION: Check per-molecule alignment
        assert pred_m.shape == true_m.shape, f"Mol {mol_id}: pred={pred_m.shape}, true={true_m.shape}"
        
        # Compute Kabsch-aligned RMSD for this molecule
        rmsd = rmsd_after_kabsch(pred_m, true_m)
        if not torch.isfinite(rmsd):
            continue
        total_loss += rmsd
        num_mols += 1
    
    if num_mols == 0:
        return pos_pred.sum() * 0.0
    
    return total_loss / num_mols

'''
def train_epoch(model, loader, optimizer, device, noise_std=0.1, epoch=None):
    """
    Train for one epoch with proper batch handling and Kabsch-aligned loss.
    
    BATCHING FLOW (FIXED):
    1. DataLoader outputs: x[B,N,F], pos[B,N,3], mask[B,N]
    2. CONSISTENT flatten: x_flat, pos_flat, batch_idx from SAME loop
    3. Build edges: edge_index[2, E] (respects molecule boundaries)
    4. EGNN forward: (x_flat, pos_noisy, edge_index) → pos_pred[num_atoms, 3]
    5. ASSERTIONS: Check all tensors are aligned
    6. Loss: Per-molecule Kabsch-aligned RMSD
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training"):
        # Step 1: Get batched data [B, N, F]
        x = batch['x'].to(device)
        pos = batch['pos'].to(device)
        mask = batch['mask'].to(device)
        skipped_batches = 0
        skipped_molecules = 0

        
        # Step 2: CONSISTENT flatten - all from same loop
        x_flat, pos_flat, batch_idx = flatten_batch_with_batch_idx(x, pos, mask, device)
        
        # Step 3: Build edge_index (molecule-aware)
        edge_index = build_edge_index_batch(mask)
        edge_index = edge_index.to(device)
        
        # Step 4: Add noise to coordinates
        pos_noisy = add_noise(pos_flat, noise_std)
        
        # Step 5: Forward pass
        pos_pred = model(x_flat, pos_noisy, edge_index)
        
        # Step 6: CRITICAL ASSERTIONS before loss
        assert pos_pred.shape == pos_flat.shape, f"Forward output mismatch: {pos_pred.shape} vs {pos_flat.shape}"
        assert batch_idx.shape[0] == pos_flat.shape[0], f"Batch idx mismatch: {batch_idx.shape[0]} vs {pos_flat.shape[0]}"
        
        # Step 7: Compute per-molecule Kabsch-aligned RMSD
        if epoch is not None and epoch < 2:
            loss = torch.mean(torch.norm(pos_pred - pos_flat, dim=1))
        else:
            loss = molecule_aware_rmsd_loss(pos_pred, pos_flat, batch_idx)
        
        if not torch.isfinite(loss):
            skipped_molecules += 1
            continue
        if not loss.requires_grad:
            skipped_batches += 1
            continue


        # Step 8: Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return (total_loss / num_batches, skipped_molecules, skipped_batches) 
'''

def train_epoch(model, loader, optimizer, device, noise_std=0.1,epoch=None):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Training"):
        x = batch['x'].to(device)
        pos = batch['pos'].to(device)
        mask = batch['mask'].to(device)

        x_flat, pos_flat, batch_idx = flatten_batch_with_batch_idx(
            x, pos, mask, device
        )

        edge_index = build_edge_index_batch(mask).to(device)

        pos_noisy = add_noise(pos_flat, noise_std)

        pos_pred = model(x_flat, pos_noisy, edge_index)

        loss = unimol_distance_loss(pos_pred, pos_flat, batch_idx)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


'''
def evaluate(model, loader, device, noise_std=0.1, clean_eval=False):
    """
    Evaluate model with proper batch handling.
    
    Args:
        clean_eval: If True, evaluate without noise (for final metrics)
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            x = batch['x'].to(device)
            pos = batch['pos'].to(device)
            mask = batch['mask'].to(device)
            
            # CONSISTENT flatten
            x_flat, pos_flat, batch_idx = flatten_batch_with_batch_idx(x, pos, mask, device)
            
            edge_index = build_edge_index_batch(mask)
            edge_index = edge_index.to(device)
            
            # Add noise (or not, for clean evaluation)
            if clean_eval:
                pos_input = pos_flat
            else:
                pos_input = add_noise(pos_flat, noise_std)
            
            pos_pred = model(x_flat, pos_input, edge_index)
            
            # Assertions
            assert pos_pred.shape == pos_flat.shape
            assert batch_idx.shape[0] == pos_flat.shape[0]
            
            loss = molecule_aware_rmsd_loss(pos_pred, pos_flat, batch_idx)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches
'''
def evaluate(model, loader, device):
    model.eval()
    total_rmsd = 0.0
    num_mols = 0

    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            pos = batch['pos'].to(device)
            mask = batch['mask'].to(device)

            x_flat, pos_flat, batch_idx = flatten_batch_with_batch_idx(
                x, pos, mask, device
            )

            edge_index = build_edge_index_batch(mask).to(device)
            pos_pred = model(x_flat, pos_flat, edge_index)

            for mol_id in batch_idx.unique():
                m = batch_idx == mol_id
                if m.sum() < 3:
                    continue
                rmsd = rmsd_after_kabsch(pos_pred[m], pos_flat[m])
                total_rmsd += rmsd.item()
                num_mols += 1

    return total_rmsd / max(num_mols, 1)


def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 100
    lr = 1e-4
    noise_std = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}, LR: {lr}, Noise: {noise_std} Å")
    
    # Load dataset
    print("Loading dataset...")
    dataset = QM9Text3DDataset('/scratch/nishanth.r/egnn/data/qm9_100k.jsonl', atom_feature_dim=10)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
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
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train with noisy inputs
        train_loss= train_epoch(model, train_loader, optimizer, device, noise_std)
        
        # Validate with noisy inputs
        #val_loss_noisy = evaluate(model, val_loader, device, noise_std, clean_eval=False)
        
        # Validate without noise (clean performance)
        #val_loss_clean = evaluate(model, val_loader, device, clean_eval=True)
        val_rmsd = evaluate(model, val_loader, device)


        print(f"Train RMSD (noisy): {train_loss:.4f} Å")
        print(f"Val RMSD (noisy):   {val_loss_noisy:.4f} Å")
        print(f"Val RMSD (clean):   {val_loss_clean:.4f} Å")
        #print(f"Skipped batches: {skipped_batches}, skipped molecules: {skipped_molecules}")

        # Save best model (based on noisy validation)
        if val_rmsd < best_val_loss:

            best_val_loss = val_rmsd
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss_noisy,
            }, 'best_egnn_geometry.pt')
            print(f"✓ Saved best model (Val RMSD: {val_loss_noisy:.4f} Å)")
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best Val RMSD: {best_val_loss:.4f} Å")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()