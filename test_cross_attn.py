"""
Quick test to verify cross-attention architecture shapes.
"""
import torch
from model import DiT

def test_cross_attention_shapes():
    """Test that all tensor shapes are correct in cross-attention mode."""
    
    # Model config
    batch_size = 2
    seq_len = 77  # Text sequence length
    latent_size = 32
    hidden_size = 768
    context_dim = 768
    
    # Create model
    model = DiT(
        input_size=latent_size,
        patch_size=2,
        in_channels=4,
        hidden_size=hidden_size,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        context_dim=context_dim,
        use_cross_attn=True,
        sprint_enabled=False,
    )
    
    # Create dummy inputs
    x = torch.randn(batch_size, 4, latent_size, latent_size)
    t = torch.randint(0, 1000, (batch_size,)).float()
    text_embeds = torch.randn(batch_size, seq_len, context_dim)
    text_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
    coords = torch.rand(batch_size, 4)
    
    print("Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  t: {t.shape}")
    print(f"  text_embeds: {text_embeds.shape}")
    print(f"  text_masks: {text_masks.shape}")
    print(f"  coords: {coords.shape}")
    
    # Forward pass
    try:
        v_head, x_backbone = model(x, t, text_embeds, coords, text_mask=text_masks)
        
        print("\nOutput shapes:")
        print(f"  v_head: {v_head.shape}")
        print(f"  x_backbone: {x_backbone.shape}")
        
        # Verify shapes
        assert v_head.shape == x.shape, f"v_head shape mismatch: {v_head.shape} vs {x.shape}"
        assert x_backbone.shape == x.shape, f"x_backbone shape mismatch: {x_backbone.shape} vs {x.shape}"
        
        print("\n✅ All shapes correct!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_cross_attention_shapes()
