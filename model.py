import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# -----------------------------------------------------------------------------
# 1. Golden Gate RoPE Implementation
# -----------------------------------------------------------------------------

class GoldenGateRoPE2d(nn.Module):
    """
    Golden Gate Rotary Positional Embeddings for 2D Data.
    Reference: https://jerryxio.ng/posts/nd-rope/
    """

    def __init__(self, head_dim, min_freq=0.2, max_freq=20.0, p_zero_freqs=0.0):
        super().__init__()
        assert head_dim % 2 == 0, "Head dimension must be even."

        self.head_dim = head_dim
        n_freqs = head_dim // 2

        # 1. Generate Log-Space Frequency Magnitudes
        if p_zero_freqs > 0:
            n_zero = round(p_zero_freqs * n_freqs)
            freqs = torch.cat([
                torch.zeros(n_zero),
                torch.linspace(math.log(min_freq), math.log(max_freq), n_freqs - n_zero).exp()
            ])
        else:
            freqs = torch.linspace(math.log(min_freq), math.log(max_freq), n_freqs).exp()

        # 2. Generate Direction Vectors using the Golden Ratio
        phi = (1 + math.sqrt(5)) / 2
        direction_spacing = math.pi / phi

        indices = torch.arange(n_freqs, dtype=torch.float32)
        angles = indices * direction_spacing
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        # 3. Create the Final Frequency Tensor
        # Shape: (n_freqs, 2)
        self.register_buffer("freqs_vec", directions * freqs.unsqueeze(-1))

    def forward(self, x):
        """
        Args: x: Input tensor of shape (Batch, Heads, H, W, Dim)
        Returns: Tensor of same shape with RoPE applied.
        """
        B, Heads, H, W, Dim = x.shape

        # 1. Create Position Grid [-1, 1]
        # We use the input device to ensure compatibility
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
            indexing="ij"
        )

        # Stack grid: (H, W, 2) -> [x, y]
        positions = torch.stack([grid_x, grid_y], dim=-1)

        # 2. Calculate Rotation Angles (Theta)
        # positions: (H, W, 2) @ freqs_vec.T: (2, Dim/2) -> (H, W, Dim/2)
        theta = torch.einsum("hwc,fc->hwf", positions, self.freqs_vec.to(dtype=x.dtype))

        # 3. Apply Rotation
        # Reshape x: (B, Heads, H, W, Dim/2, 2)
        x_pairs = x.view(B, Heads, H, W, Dim // 2, 2)

        # Prepare sin/cos: (1, 1, H, W, Dim/2, 1)
        cost = theta.cos().view(1, 1, H, W, Dim // 2, 1)
        sint = theta.sin().view(1, 1, H, W, Dim // 2, 1)

        # Rotation: [x, y] * [[cos, -sin], [sin, cos]]
        x_rot = x_pairs[..., 0:1] * cost - x_pairs[..., 1:2] * sint
        y_rot = x_pairs[..., 0:1] * sint + x_pairs[..., 1:2] * cost

        x_out = torch.cat([x_rot, y_rot], dim=-1)

        # Flatten last two dims back to Dim
        return x_out.flatten(-2)


# -----------------------------------------------------------------------------
# 2. Model Components
# -----------------------------------------------------------------------------

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvAttention(nn.Module):
    def __init__(self, dim, num_heads=8, is_cross_attention=False):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.is_cross_attention = is_cross_attention

        self.q_proj = nn.Conv2d(dim, dim, 1)

        if is_cross_attention:
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
        else:
            self.k_proj = nn.Conv2d(dim, dim, 1)
            self.v_proj = nn.Conv2d(dim, dim, 1)

        self.out_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x, context=None, rope_func=None):
        B, C, H, W = x.shape

        # 1. Project Queries
        # (B, C, H, W) -> (B, nH, hD, H, W) -> (B, nH, H, W, hD)
        q = self.q_proj(x).view(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2)

        if self.is_cross_attention:
            # Cross Attn: Keys/Values from Context (Text)
            # (B, L, C) -> (B, L, nH, hD) -> (B, nH, L, hD)
            k = self.k_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # No RoPE for Cross Attention (Image Q vs Text K has no 2D geometric relation)
            # Flatten Q spatial dims: (B, nH, H, W, hD) -> (B, nH, HW, hD)
            q = q.flatten(2, 3)

        else:
            # Self Attn: Keys/Values from Image
            k = self.k_proj(x).view(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2)
            v = self.v_proj(x).view(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2)

            # --- Apply Golden Gate RoPE ---
            if rope_func is not None:
                # q, k shape: (B, nH, H, W, hD)
                q = rope_func(q)
                k = rope_func(k)

            # Flatten spatial dims for attention: (B, nH, HW, hD)
            q = q.flatten(2, 3)
            k = k.flatten(2, 3)
            v = v.flatten(2, 3).contiguous()  # Contiguous often helps compile

        # 2. Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        # 3. Reshape Back
        # (B, nH, HW, hD) -> (B, nH, H, W, hD) -> (B, C, H, W)
        out = out.view(B, self.num_heads, H, W, self.head_dim).permute(0, 1, 4, 2, 3).reshape(B, C, H, W)

        return self.out_proj(out)


class ConvFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.dw1 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.proj_up = nn.Conv2d(dim, hidden_dim * 2, 1)
        self.dw2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, groups=hidden_dim)
        self.proj_down = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        x = self.dw1(x)
        x = self.proj_up(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.silu(x1) * x2
        x = self.dw2(x)
        x = self.proj_down(x)
        return x


class MinimalBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = LayerNorm2d(dim, eps=1e-6)
        self.norm2 = LayerNorm2d(dim, eps=1e-6)
        self.norm3 = LayerNorm2d(dim, eps=1e-6)

        self.attn = ConvAttention(dim, num_heads, is_cross_attention=False)
        self.cross_attn = ConvAttention(dim, num_heads, is_cross_attention=True)
        self.ffn = ConvFeedForward(dim, int(dim * 4))

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim, bias=True))

    def forward(self, x, t_emb, context, rope_func):
        params = self.adaLN_modulation(t_emb).chunk(9, dim=1)
        (shift_msa, scale_msa, gate_msa,
         shift_cross, scale_cross, gate_cross,
         shift_ffn, scale_ffn, gate_ffn) = [p.unsqueeze(2).unsqueeze(3) for p in params]

        # Pass RoPE to Self Attention
        normed_msa = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(normed_msa, rope_func=rope_func)

        normed_cross = modulate(self.norm2(x), shift_cross, scale_cross)
        x = x + gate_cross * self.cross_attn(normed_cross, context=context)  # No RoPE here

        normed_ffn = modulate(self.norm3(x), shift_ffn, scale_ffn)
        x = x + gate_ffn * self.ffn(normed_ffn)
        return x


class GeneralizedHyperConnection2d(nn.Module):
    """
    Implements the VWN connection mechanism for 2D feature maps.
    Compresses Over-Width state -> Backbone, runs backbone, Expands -> Over-Width state.
    """

    def __init__(self, virtual_dim, backbone_dim):
        super().__init__()
        self.virtual_dim = virtual_dim
        self.backbone_dim = backbone_dim

        # 1x1 Convs act as the Projection Matrices (A and B from the paper)
        self.compressor = nn.Conv2d(virtual_dim, backbone_dim, kernel_size=1, bias=False)
        self.expander = nn.Conv2d(backbone_dim, virtual_dim, kernel_size=1, bias=False)

        self._init_weights()

    def _init_weights(self):
        # Initialize Compressor to partial Identity
        nn.init.zeros_(self.compressor.weight)
        with torch.no_grad():
            min_dim = min(self.virtual_dim, self.backbone_dim)
            # Set weights to make it an identity pass-through for the shared dimensions
            for i in range(min_dim):
                self.compressor.weight[i, i, 0, 0] = 1.0

        # Initialize Expander to Zero
        # This ensures the layer starts as a residual pass-through of the virtual state
        nn.init.zeros_(self.expander.weight)

    def forward(self, h_virtual, backbone_block, t_emb, txt_emb, rope_func):
        # 1. Compress (Virtual -> Backbone)
        h_backbone = self.compressor(h_virtual)

        # 2. Backbone Processing (Standard T2I Block)
        h_processed = backbone_block(h_backbone, t_emb, txt_emb, rope_func)

        # 3. Expand (Backbone -> Virtual) and Add Residual
        out = self.expander(h_processed) + h_virtual
        return out


# -----------------------------------------------------------------------------
# 3. Main Model (Minimal T2I)
# -----------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_embed_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_embed_size = freq_embed_size

    def forward(self, t):
        half = self.freq_embed_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_embed_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


class MinimalT2I(nn.Module):
    def __init__(
            self,
            patch_size=16,
            in_channels=4,
            hidden_size=768,
            depth=12,
            num_heads=12,
            context_dim=768,
            virtual_expansion=1,  # Configurable VWN expansion (1 = disabled/standard)
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.virtual_expansion = virtual_expansion
        self.use_vwn = virtual_expansion > 1

        # Virtual Dimension (D' = r * D)
        self.virtual_dim = hidden_size * virtual_expansion if self.use_vwn else hidden_size

        # 1. Input: Direct projection
        # Replaced bottleneck patch embed with direct projection as requested
        self.patch_embed = nn.Conv2d(
            in_channels,
            self.virtual_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Golden Gate RoPE
        head_dim = hidden_size // num_heads
        self.rope = GoldenGateRoPE2d(head_dim=head_dim)

        # 2. Conditioning
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.context_proj = nn.Linear(context_dim, hidden_size)

        # 3. Backbone (with optional VWN)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            backbone = MinimalBlock(hidden_size, num_heads)
            if self.use_vwn:
                ghc = GeneralizedHyperConnection2d(self.virtual_dim, hidden_size)
                self.layers.append(nn.ModuleDict({'backbone': backbone, 'ghc': ghc}))
            else:
                self.layers.append(backbone)

        # 4. Output
        # Merged final_reduce into final_proj logic by operating on virtual_dim directly
        self.final_norm = LayerNorm2d(self.virtual_dim)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * self.virtual_dim, bias=True))
        self.final_proj = nn.ConvTranspose2d(self.virtual_dim, in_channels, kernel_size=patch_size, stride=patch_size)

        self.initialize_weights()

    def initialize_weights(self):
        # Init patch embed
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        nn.init.xavier_uniform_(self.final_proj.weight)
        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.constant_(self.context_proj.bias, 0)

        # AdaLN Zero Init
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)

        for layer in self.layers:
            if self.use_vwn:
                block = layer['backbone']
            else:
                block = layer

            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t, text_emb):
        # x: (B, Cin, H, W) -> (B, Virtual_Dim, H, W)
        x = self.patch_embed(x)

        t_emb = self.t_embedder(t)
        txt_emb = self.context_proj(text_emb)

        for layer in self.layers:
            if self.use_vwn:
                # GHC handles routing: x(wide) -> compress -> backbone(narrow) -> expand -> add -> x(wide)
                x = layer['ghc'](x, layer['backbone'], t_emb, txt_emb, rope_func=self.rope)
            else:
                # Standard Backbone
                x = layer(x, t_emb, txt_emb, rope_func=self.rope)

        # Final output block operates directly on virtual_dim
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift.unsqueeze(2).unsqueeze(3), scale.unsqueeze(2).unsqueeze(3))
        x = self.final_proj(x)
        return x


# -----------------------------------------------------------------------------
# 4. Benchmark
# -----------------------------------------------------------------------------

def benchmark_rope():
    print("\n" + "=" * 60)
    print("Minimal T2I + Golden Gate RoPE + Torch.Compile + Channels Last")
    print("=" * 60)

    BATCH_SIZE = 4
    IMG_SIZE = 512
    PATCH_SIZE = 32
    HIDDEN_SIZE = 768
    DEPTH = 12

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize
    model = MinimalT2I(
        patch_size=PATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        depth=DEPTH
    ).to(DEVICE)

    print(f"RoPE Mode: Golden Gate (Non-axial)")
    print(f"Resolution: {IMG_SIZE}x{IMG_SIZE} (Flexible)")

    # Optimize
    model = model.to(memory_format=torch.channels_last)
    print("Compiling...")
    # Compile mode note: RoPE involves complex reshaping. 'reduce-overhead' is usually safe.
    # If 'max-autotune' fails, revert to 'reduce-overhead' or default.
    model = torch.compile(model, mode="reduce-overhead")
    model.train()

    # Inputs
    x = torch.randn(BATCH_SIZE, 4, IMG_SIZE, IMG_SIZE, device=DEVICE).to(memory_format=torch.channels_last)
    t = torch.randint(0, 1000, (BATCH_SIZE,), device=DEVICE)
    ctx = torch.randn(BATCH_SIZE, 77, 768, device=DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler()

    # Warmup
    print("Warming up...")
    start_warm = time.time()
    for _ in range(5):
        with torch.amp.autocast(device_type='cuda', enabled=(DEVICE == 'cuda')):
            out = model(x, t, ctx)
            loss = out.mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    print(f"Warmup done: {time.time() - start_warm:.2f}s")

    # Benchmark
    print("Running Benchmark (50 iters)...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(50):
        with torch.amp.autocast(device_type='cuda', enabled=(DEVICE == 'cuda')):
            out = model(x, t, ctx)
            loss = out.mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end)
    avg_ms = elapsed / 50
    print("-" * 30)
    print(f"Avg Iteration Time: {avg_ms:.2f} ms")
    print(f"Throughput: {(BATCH_SIZE * 1000) / avg_ms:.2f} imgs/sec")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_rope()