import torch

def count_model_parameters(pth_path):
    checkpoint = torch.load(pth_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        print("Could not find valid state_dict key in checkpoint.")
        return

    total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
    print(f"Total number of parameters: {total_params:,}")

# Použití
count_model_parameters("model_8-1343.pth")