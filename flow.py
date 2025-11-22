import torch
import torch.nn as nn
import torch.nn.functional as F


class xFlowMatching(nn.Module):
    def __init__(self, net):
        """
        Args:
            net: A neural network module that takes (z, t) as input and outputs x_pred.
                 Signature: net(z, t) -> x_pred
        """
        super().__init__()
        self.net = net

    def sample_t(self, batch_size, device):
        """Samples time t from uniform distribution [0, 1]."""
        return torch.rand((batch_size,), device=device)

    def reshape_t(self, t, x):
        """
        Reshapes t to allow broadcasting with x.
        e.g., if x is (B, C, H, W), t becomes (B, 1, 1, 1)
        """
        while len(t.shape) < len(x.shape):
            t = t.unsqueeze(-1)
        return t

    def forward(self, x):
        """
        Algorithm 1: Training step
        Args:
            x: Training batch of real data (e.g., images)
        Returns:
            loss: Scalar loss value
        """
        b = x.shape[0]
        device = x.device

        # 1. Sample t
        t = self.sample_t(b, device)

        # 2. Sample noise e
        e = torch.randn_like(x)

        # Reshape t for broadcasting operations
        t_reshaped = self.reshape_t(t, x)

        # 3. Interpolate z = t * x + (1 - t) * e
        # Note: At t=0, z=noise. At t=1, z=data.
        z = t_reshaped * x + (1 - t_reshaped) * e

        # 4. Calculate Target Velocity v
        # Formula from image: v = (x - z) / (1 - t)
        # Mathematical simplification:
        # v = (x - (tx + e - te)) / (1-t) = ((1-t)x - (1-t)e) / (1-t) = x - e
        # We use the algebraic simplification for numerical stability of the target.
        target_v = x - e

        # 5. Model Prediction
        # The network predicts 'x' (clean data) given noisy 'z' and time 't'
        x_pred = self.net(z, t)

        # 6. Derived Velocity Prediction
        # Formula: v_pred = (x_pred - z) / (1 - t)
        # We add a small epsilon to (1-t) to prevent division by zero if t draws exactly 1.0
        epsilon = 1e-5
        v_pred = (x_pred - z) / (1 - t_reshaped + epsilon)

        # 7. Loss (L2 Loss / MSE)
        loss = F.mse_loss(v_pred, target_v)

        return loss

    @torch.no_grad()
    def sample(self, shape, steps=50, device='cpu'):
        """
        Algorithm 2: Sampling step (Euler)
        Args:
            shape: Tuple of output shape (e.g., (1, 3, 32, 32))
            steps: Number of integration steps (ODE solver steps)
            device: Torch device
        Returns:
            Generated sample x_pred
        """
        # Start from pure noise (t=0 corresponds to noise in this schedule)
        z = torch.randn(shape, device=device)

        # Linear time steps from 0 to 1
        times = torch.linspace(0, 1, steps + 1, device=device)

        # Euler Integration Loop
        for i in range(steps):
            t_curr = times[i]
            t_next = times[i + 1]

            # Expand t for batch processing
            t_curr_expanded = t_curr.repeat(shape[0])
            # (Usually network expects t to be broadcastable or handled internally)

            # 1. Predict x
            # Note: Depending on your net implementation, you might need to reshape t
            # inside the loop similar to training.
            x_pred = self.net(z, t_curr_expanded)

            # 2. Derive v_pred
            # v_pred = (x_pred - z) / (1 - t)
            # Handle division stability near t=1
            denom = 1 - t_curr
            if denom < 1e-5:
                # At the very end of sampling, v is ill-defined by this formula,
                # but z should already be close to x.
                # We can assume v_pred = x_pred - z (approx) or just skip the last tiny update.
                # Here we clamp for safety.
                denom = 1e-5

            v_pred = (x_pred - z) / denom

            # 3. Update z
            # z_next = z + (t_next - t) * v_pred
            dt = t_next - t_curr
            z = z + dt * v_pred

        return z