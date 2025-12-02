import torch
import torch.nn as nn
from diffusers import Transformer2DModel

class MinimalT2I(nn.Module):
    """
    Wrapper around diffusers.Transformer2DModel to match the MinimalT2I interface.
    Replaces the custom Z-Image implementation with a standard Diffusers component.
    """
    def __init__(
        self,
        patch_size=2,
        in_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=12,
        context_dim=768,
        **kwargs
    ):
        super().__init__()
        
        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        attention_head_dim = hidden_size // num_heads
        
        # Initialize Diffusers Transformer2DModel
        # This uses a standard transformer backbone suitable for diffusion (DiT-like)
        self.model = Transformer2DModel(
            num_attention_heads=num_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=in_channels,
            num_layers=depth,
            dropout=0.0,
            cross_attention_dim=context_dim,
            sample_size=kwargs.get("sample_size", 32), # Default sample size (latent), can adapt
            patch_size=patch_size,
            norm_type="layer_norm", # Standard LayerNorm + Time Embedding added to input/attn
            use_linear_projection=True, # Project input channels to hidden_size
        )
        
        # Capture any other kwargs to avoid errors (e.g. virtual_expansion)
        self.unused_kwargs = kwargs

    def forward(self, x, t, text_emb):
        """
        Forward pass adapter.
        Args:
            x: (Batch, Channels, Height, Width) - Noisy input
            t: (Batch,) - Timesteps
            text_emb: (Batch, SeqLen, Dim) - Text embeddings
        """
        # Diffusers Transformer2DModel expects:
        # hidden_states -> x
        # timestep -> t
        # encoder_hidden_states -> text_emb
        
        output = self.model(
            hidden_states=x,
            timestep=t,
            encoder_hidden_states=text_emb,
            return_dict=True
        )
        
        return output.sample