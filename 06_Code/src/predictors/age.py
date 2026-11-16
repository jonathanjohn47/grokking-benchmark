"""AGE predictor (Predictor 4 of 8) — Adaptive Grokking Epoch via Neural Collapse."""
import torch
import numpy as np

_EPS = 1e-12

def compute_age_metrics_for_checkpoint(model, train_loader, device):
    """Compute NC1 for one frozen checkpoint.
    
    Returns dict with keys: nc1, fn, N
    """
    # 1. Extract features from penultimate layer
    features = []
    labels = []
    
    def hook_fn(module, inputs, outputs):
        # inputs[0] is the input tensor
        # inputs[0][:, 2, :] is the representation at position 2
        h = inputs[0]
        if h.ndim == 3:
            h = h[:, 2, :]
        features.append(h.detach())
    
    hook = model.output_head.register_forward_hook(hook_fn)
    
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            _ = model(x)
            labels.append(y)
    
    hook.remove()
    
    # 2. Concatenate and move to CPU float64
    Phi = torch.cat(features, dim=0).cpu().double()  # [N, 128]
    Y = torch.cat(labels, dim=0).cpu().long()  # [N]
    
    N = Phi.shape[0]
    C = model.output_head.out_features - 1  # Dynamically infer number of classes
    
    # 3. Compute class means and global mean
    mu_G = Phi.mean(dim=0)  # [128]
    
    Tr_W = 0.0
    Tr_B = 0.0
    n_classes_used = 0
    
    for c in range(C):
        mask = (Y == c)
        n_c = mask.sum().item()
        if n_c == 0:
            continue
        
        Phi_c = Phi[mask]  # [n_c, 128]
        mu_c = Phi_c.mean(dim=0)  # [128]
        
        # Within-class scatter for this class
        Tr_W += ((Phi_c - mu_c) ** 2).sum().item() / n_c
        
        # Between-class scatter for this class
        Tr_B += ((mu_c - mu_G) ** 2).sum().item()
        
        n_classes_used += 1
    
    # 4. Average over classes
    Tr_W = Tr_W / n_classes_used
    Tr_B = Tr_B / n_classes_used
    
    # 5. Compute NC1
    nc1 = Tr_W / (Tr_B + _EPS)
    
    # 6. Compute mean feature norm
    fn = torch.norm(Phi, dim=1).mean().item()
    
    return {
        "nc1": float(nc1),
        "fn": float(fn),
        "N": int(N)
    }
