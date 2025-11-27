import torch
import importlib
from config import Config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    config = Config()
    print(f"Model Architecture: {config.model_arch}")
    
    if config.model_arch == "Styx":
        from models.styx import MinimalT2I
    elif config.model_arch == "ZImage":
        from models.zimage import MinimalT2I
    else:
        try:
            module = importlib.import_module(f"models.{config.model_arch.lower()}")
            MinimalT2I = getattr(module, "MinimalT2I")
        except ImportError:
            print(f"Could not load model {config.model_arch}")
            return

    # Mock Context Dim (e.g. from Text Encoder)
    # Qwen hidden size is usually larger, but let's assume 768 for compatibility check or reading from Config
    # Config doesn't have context_dim explicitly, it's derived in train.py.
    # We'll use 768 as a placeholder or calculate it if possible.
    context_dim = 768 * 4 # Default Qwen-like expansion used in train.py (text_encoder.config.hidden_size * 4)
    
    print(f"Initializing model with hidden_size={config.hidden_size}, depth={config.depth}, heads={config.num_heads}")

    model = MinimalT2I(
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        context_dim=context_dim, 
    )
    
    total_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    print(f"In Millions: {total_params/1e6:.2f}M")

if __name__ == "__main__":
    main()