import torch

def compute_naswot(model, data_batch, device):
    model.eval()
    
    activations = []
    hooks = []
    
    def hook_fn(module, input, output):
        # Binarisieren: 1 wenn Aktivierung > 0, sonst 0
        binary = (output > 0).float().view(output.size(0), -1)
        activations.append(binary.detach().cpu())
        
    # Hook auf alle GELU/ReLU Aktivierung (ViT nutzt GELU im MLP)
    for module in model.modules():
        if isinstance(module, (torch.nn.GELU, torch.nn.ReLU)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        model(data_batch.to(device))
        
    for h in hooks:
        h.remove()
        
    if not activations:
        return 0.0
    
    # [batch_size, alle_features_concat]
    K_matrix = torch.cat(activations, dim=1)
    
    # Kernel Matrix [B, B]: dot-product der Binärmuster
    K = K_matrix @ K_matrix.T
    
    # Regularisierung für numerische Stabilität
    score = torch.logdet(K + 1e-6 * torch.eye(K.size(0))).item()
    
    return score