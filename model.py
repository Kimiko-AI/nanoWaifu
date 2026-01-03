import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm1 = self.norm1(x)
        x_norm1 = modulate(x_norm1, shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm2 = self.norm2(x)
        x_norm2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_out =  self.mlp(x_norm2)
        
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class DiTBackbone(nn.Module):
    """
    Modified DiT to act as a backbone, returning feature maps and its own prediction.
    """
    def __init__(
        self,
        input_size=64,
        patch_size=4,
        in_channels=3,
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=100,
        class_dropout_prob=0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)
        self.coord_embedder = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels=in_channels)
        
        self.gradient_checkpointing = False
        self.initialize_weights()

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        c = self.in_channels
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, class_labels, crop_coords):
        x = self.x_embedder(x) + self.pos_embed
        
        # Save initial embedding for skip connection (N, T, D)
        x_start = x

        t_emb = self.t_embedder(t)
        
        if self.training:
            mask = torch.rand(class_labels.shape[0], device=class_labels.device) < self.class_dropout_prob
            class_labels = torch.where(mask, torch.tensor(self.num_classes, device=class_labels.device), class_labels)
        
        y_emb = self.y_embedder(class_labels)
        coord_emb = self.coord_embedder(crop_coords)
        c = t_emb + y_emb + coord_emb

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)
        
        # Compute DiT prediction
        x_pred = self.final_layer(x, c)
        x_pred = self.unpatchify(x_pred)

        # Reshape both to feature maps: (N, D, H_grid, W_grid)
        H_grid = W_grid = int(x.shape[1] ** 0.5)
        x_start = x_start.transpose(1, 2).reshape(x_start.shape[0], self.hidden_size, H_grid, W_grid)
        x = x.transpose(1, 2).reshape(x.shape[0], self.hidden_size, H_grid, W_grid)
        
        return x_pred, x_start, x, t_emb

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if temb_channels:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        if temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
            
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class ResNetHead(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, hidden_size=1024, num_blocks=4):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        
        # Timestep embedding (re-created here to match interface, or reused)
        # We'll expect t_emb to be passed in, but we need to project it
        # Actually, DiTBackbone returns t_emb (size backbone_hidden), we need to project it to 4*hidden_size?
        # Or we can just project the raw t_emb to fit ResBlock. ResBlock expects `temb_channels` input.
        # Let's say we receive the raw t_emb from backbone (size `backbone_hidden`).
        # We'll project it to `hidden_size` for the ResBlocks.
        
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_channels, hidden_size), 
        )

        # Input projection: Combine start + end features (2 * in_channels) -> hidden_size
        self.input_proj = nn.Conv2d(in_channels * 2, hidden_size, 3, padding=1)
        
        self.blocks = nn.ModuleList([
            ResBlock(hidden_size, hidden_size, temb_channels=hidden_size) 
            for _ in range(num_blocks)
        ])
        
        # PixelShuffle Upscaling
        # Output dim needs to be out_channels * patch_size^2
        self.final_conv = nn.Conv2d(hidden_size, out_channels * patch_size**2, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(patch_size)

    def forward(self, x_start, x_end, t_emb):
        # x_start, x_end: (N, C, H, W)
        # t_emb: (N, C)
        
        x = torch.cat([x_start, x_end], dim=1)
        x = self.input_proj(x)
        
        t_emb = self.temb_proj(t_emb)
        
        for block in self.blocks:
            x = block(x, t_emb)
            
        x = self.final_conv(x)
        x = self.pixel_shuffle(x)
        return x

class DiT(nn.Module):
    """
    Composite model: DiT Backbone + ResNet Head
    """
    def __init__(
        self,
        input_size=64,
        patch_size=4,
        in_channels=3,
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=100,
        class_dropout_prob=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.backbone = DiTBackbone(
            input_size, patch_size, in_channels, hidden_size,
            depth, num_heads, mlp_ratio, num_classes, class_dropout_prob
        )
        
        self.head = ResNetHead(
            in_channels=hidden_size, # Backbone output dim
            out_channels=in_channels, # Image channels
            patch_size=patch_size,
            hidden_size=1024,
            num_blocks=3
        )

    def enable_gradient_checkpointing(self):
        self.backbone.enable_gradient_checkpointing()

    def forward(self, x, t, class_labels, crop_coords):
        # DiT Backbone forward
        x_pred, x_start, x_end, t_emb = self.backbone(x, t, class_labels, crop_coords)
        
        # Head Forward
        out = self.head(x_start, x_end, t_emb)
        return out, x_pred

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb