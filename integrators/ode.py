import torch as th
import sys
import os
# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import time_shift, get_lin_function


class ode:
    """ODE solver class"""

    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        do_shift=False,
        time_shifting_factor=None,
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        self.do_shift = do_shift
        self.t = th.linspace(t0, t1, num_steps)
        if time_shifting_factor:
            self.t = self.t / (self.t + time_shifting_factor - time_shifting_factor * self.t)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type
        self.num_steps = num_steps

    def sample(self, x, model, **model_kwargs):
        """Sample using ODE integration with RK4 or Euler method"""
        device = x[0].device if isinstance(x, tuple) else x.device
        
        t = self.t.to(device)
        if self.do_shift:
            mu = get_lin_function(y1=0.5, y2=1.15)(x.shape[1] if not isinstance(x, tuple) else x[0].shape[1])
            t = time_shift(mu, 1.0, t)
        
        samples = [x]
        
        # Simple RK4 or Euler integration
        for i in range(len(t) - 1):
            t_current = th.ones(x[0].size(0) if isinstance(x, tuple) else x.size(0)).to(device) * t[i]
            dt = t[i + 1] - t[i]
            
            if self.sampler_type.lower() in ['rk4', 'dopri5']:
                # RK4 integration
                k1 = self.drift(x, t_current, model, **model_kwargs)
                k2 = self.drift(x + 0.5 * dt * k1, t_current + 0.5 * dt, model, **model_kwargs)
                k3 = self.drift(x + 0.5 * dt * k2, t_current + 0.5 * dt, model, **model_kwargs)
                k4 = self.drift(x + dt * k3, t_current + dt, model, **model_kwargs)
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                # Euler integration
                drift_val = self.drift(x, t_current, model, **model_kwargs)
                x = x + drift_val * dt
            
            samples.append(x)
        
        return samples
