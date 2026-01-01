import torch
from model import DiT
from train import sample_flow

def test_model():
    print("Testing Model Forward Pass...")
    model = DiT(
        input_size=64,
        patch_size=4,
        in_channels=3,
        hidden_size=64, # Small for test
        depth=2,
        num_heads=2,
        num_classes=10,
        class_dropout_prob=0.1
    )
    
    x = torch.randn(2, 3, 64, 64)
    # Testing with float t (scaled)
    t = torch.rand((2,)) * 1000
    c = torch.randint(0, 10, (2,))
    coords = torch.rand(2, 4)
    
    out = model(x, t, c, coords)
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape
    print("Forward pass successful.")
    
    print("Testing Backward Pass...")
    loss = out.mean()
    loss.backward()
    print("Backward pass successful.")

def test_flow_matching():
    print("Testing Flow Matching Logic...")
    model = DiT(
        input_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=32,
        depth=1,
        num_heads=1,
        num_classes=5
    )
    
    x1 = torch.randn(2, 3, 32, 32)
    classes = torch.randint(0, 5, (2,))
    coords = torch.rand(2, 4)
    
    # Simulate training step
    t = torch.rand((2,))
    x0 = torch.randn_like(x1)
    
    t_reshaped = t.view(-1, 1, 1, 1)
    xt = (1 - t_reshaped) * x0 + t_reshaped * x1
    ut = x1 - x0
    
    vt = model(xt, t * 1000, classes, coords)
    
    loss = torch.mean((vt - ut) ** 2)
    print(f"Loss: {loss.item()}")
    
    # Sample
    samples = sample_flow(model, 32, 1, classes[:1], coords[:1], "cpu", steps=5)
    print(f"Sample shape: {samples.shape}")
    assert samples.shape == (1, 3, 32, 32)

if __name__ == "__main__":
    test_model()
    test_flow_matching()