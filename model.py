import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
import numpy as np
import math

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

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
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
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm2 = self.norm2(x)
        x_norm2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
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
    Modified DiT to act as a backbone, returning feature maps.
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

    def forward(self, x, t, class_labels, crop_coords):
        x = self.x_embedder(x) + self.pos_embed
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
        
        # Output x is (N, T, D). Reshape to feature map.
        H_grid = W_grid = int(x.shape[1] ** 0.5)
        x = x.transpose(1, 2).reshape(x.shape[0], self.hidden_size, H_grid, W_grid)
        return x, t_emb # Return t_emb to reuse in UNet if desired, though UNet usually makes its own.

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

class SmallUNet(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels, hidden_size=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, 4 * hidden_size),
        )
        temb_dim = 4 * hidden_size

        # Encoder - Stage 1 (Aggressive Downsample)
        # 64x64 -> 32x32
        self.enc1 = nn.Conv2d(in_channels, hidden_size * 2, kernel_size=7, stride=2, padding=3)
        self.block1 = ResBlock(hidden_size * 2, hidden_size * 2, temb_dim)
        
        # 32x32 -> 16x16
        self.down2 = nn.Conv2d(hidden_size * 2, hidden_size * 4, 3, stride=2, padding=1)
        self.block3 = ResBlock(hidden_size * 4, hidden_size * 4, temb_dim)

        # 16x16 -> 8x8
        self.down3 = nn.Conv2d(hidden_size * 4, hidden_size * 4, 3, stride=2, padding=1)
        self.block4 = ResBlock(hidden_size * 4, hidden_size * 4, temb_dim)

        # 8x8 -> 4x4
        self.down4 = nn.Conv2d(hidden_size * 4, hidden_size * 4, 3, stride=2, padding=1)
        self.block5 = ResBlock(hidden_size * 4, hidden_size * 4, temb_dim)

        # Bottleneck with conditioning (4x4)
        self.mid_block1 = ResBlock(hidden_size * 4 + cond_channels, hidden_size * 4, temb_dim)
        self.mid_block2 = ResBlock(hidden_size * 4, hidden_size * 4, temb_dim)
        
        # Decoder
        # 4x4 -> 8x8
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_up1 = nn.Conv2d(hidden_size * 4, hidden_size * 4, 3, padding=1)
        self.block6 = ResBlock(hidden_size * 4 + hidden_size * 4, hidden_size * 4, temb_dim)
        
        # 8x8 -> 16x16
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_up2 = nn.Conv2d(hidden_size * 4, hidden_size * 4, 3, padding=1)
        self.block7 = ResBlock(hidden_size * 4 + hidden_size * 4, hidden_size * 4, temb_dim)
        
        # 16x16 -> 32x32
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_up3 = nn.Conv2d(hidden_size * 4, hidden_size * 2, 3, padding=1)
        self.block8 = ResBlock(hidden_size * 2 + hidden_size * 2, hidden_size * 2, temb_dim)
        
        # Final Upsample: 32x32 -> 64x64
        self.up_out = nn.ConvTranspose2d(hidden_size * 2, hidden_size, kernel_size=4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(hidden_size, out_channels, 3, padding=1)

    def forward(self, x, t, cond_features):
        temb = self.time_embed(TimestepEmbedder.timestep_embedding(t, self.hidden_size))
        
        # Encoder
        x1 = self.enc1(x) # 32x32
        x1 = self.block1(x1, temb)
        
        x2 = self.down2(x1) # 16x16
        x2 = self.block3(x2, temb)
        
        x3 = self.down3(x2) # 8x8
        x3 = self.block4(x3, temb)
        
        x4 = self.down4(x3) # 4x4
        x4 = self.block5(x4, temb)
        
        # Bottleneck - Inject conditioning
        x_mid = torch.cat([x4, cond_features], dim=1)
        x_mid = self.mid_block1(x_mid, temb)
        x_mid = self.mid_block2(x_mid, temb)
        
        # Decoder
        x_up = self.up1(x_mid) # 8x8
        x_up = self.conv_up1(x_up)
        x_up = torch.cat([x_up, x3], dim=1)
        x_up = self.block6(x_up, temb)
        
        x_up = self.up2(x_up) # 16x16
        x_up = self.conv_up2(x_up)
        x_up = torch.cat([x_up, x2], dim=1)
        x_up = self.block7(x_up, temb)
        
        x_up = self.up3(x_up) # 32x32
        x_up = self.conv_up3(x_up)
        x_up = torch.cat([x_up, x1], dim=1)
        x_up = self.block8(x_up, temb)
        
        # Final Upsample
        out = self.up_out(x_up) # 64x64
        out = self.out_conv(out)
        return out

class DiT(nn.Module):
    """
    Composite model: DiT Backbone + SmallUNet Head
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
        self.backbone = DiTBackbone(
            input_size, patch_size, in_channels, hidden_size,
            depth, num_heads, mlp_ratio, num_classes, class_dropout_prob
        )
        # UNet input channels = in_channels (image)
        # UNet output channels = in_channels (velocity)
        # UNet cond channels = hidden_size (DiT output)
        self.head = SmallUNet(
            in_channels=in_channels,
            out_channels=in_channels,
            cond_channels=hidden_size,
            hidden_size=64 
        )

    def enable_gradient_checkpointing(self):
        self.backbone.enable_gradient_checkpointing()

    def forward(self, x, t, class_labels, crop_coords):
        # DiT Backbone forward
        features, _ = self.backbone(x, t, class_labels, crop_coords)
        
        # UNet Forward
        out = self.head(x, t, features)
        return out

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