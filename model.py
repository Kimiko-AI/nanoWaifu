import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
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
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(dim // num_heads)
        self.k_norm = nn.LayerNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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
        attn_out = self.attn(x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm2 = self.norm2(x)
        x_norm2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class FinalLayer(nn.Module):
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
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.unshuffle = nn.PixelUnshuffle(patch_size)
        self.proj = nn.Conv2d(in_chans * patch_size ** 2, embed_dim, kernel_size=7, padding=3)

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class DiTBackbone(nn.Module):
    def __init__(
            self,
            input_size=64,
            patch_size=4,
            in_channels=3,
            hidden_size=384,
            depth=6,
            num_heads=6,
            mlp_ratio=4.0,
            text_embed_dim=768,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Image Embedder
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)

        # Text Projection
        self.text_proj = nn.Linear(text_embed_dim, hidden_size)

        # Timestep & Coord Embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.coord_embedder = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Fixed Positional Embeddings
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels=in_channels)

        # --- Token Dropping Components ---
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.drop_restore_proj = nn.Linear(hidden_size * 2, hidden_size)
        # ---------------------------------

        self.gradient_checkpointing = False
        self.initialize_weights()

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def initialize_weights(self):
        # [Same initialization logic as before...]
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

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Init mask token and projection
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.drop_restore_proj.weight)
        nn.init.constant_(self.drop_restore_proj.bias, 0)

    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        c = self.in_channels
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, text_tokens, crop_coords, token_drop_ratio=0.0):
        # 1. Prepare Image Tokens
        x = self.x_embedder(x) + self.pos_embed

        # --- Capture x_start for ResNet (Image tokens right after embedder) ---
        x_start_resnet = x.clone()
        # ----------------------------------------------------------------------

        # 2. Prepare Text Tokens
        text_emb = self.text_proj(text_tokens)

        # 3. Concatenate: [Text, Image] -> Treated equally
        # Shape: (B, N_total, D) where N_total = N_txt + N_img
        x_concat = torch.cat([text_emb, x], dim=1)
        num_text = text_emb.shape[1]

        # 4. Prepare Conditioning
        t_emb = self.t_embedder(t)
        coord_emb = self.coord_embedder(crop_coords)
        c = t_emb + coord_emb

        # 5. Run First 2 Blocks (Full Sequence)
        for i in range(2):
            x_concat = self.blocks[i](x_concat, c)

        # Save state before dropping for skip connection
        x_skip_fusion = x_concat.clone()

        # 6. Token Dropping Logic
        if token_drop_ratio > 0.0:
            B, N, D = x_concat.shape
            len_keep = int(N * (1 - token_drop_ratio))

            # Generate noise for random shuffling of the WHOLE sequence
            noise = torch.rand(B, N, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # Keep the subset
            ids_keep = ids_shuffle[:, :len_keep]
            x_concat = torch.gather(x_concat, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            # Run Middle Blocks
            for i in range(2, len(self.blocks) - 2):
                if self.gradient_checkpointing and self.training:
                    x_concat = torch.utils.checkpoint.checkpoint(self.blocks[i], x_concat, c, use_reentrant=False)
                else:
                    x_concat = self.blocks[i](x_concat, c)

            # Restore Sequence
            mask_tokens = self.mask_token.repeat(B, N - len_keep, 1)
            x_concat = torch.cat([x_concat, mask_tokens], dim=1)
            x_concat = torch.gather(x_concat, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

            # Fuse: Concatenate (Before Drop) + (After Middle Blocks) -> Project
            x_concat = torch.cat([x_skip_fusion, x_concat], dim=-1)
            x_concat = self.drop_restore_proj(x_concat)

        else:
            # If drop ratio is 0, just run the middle blocks normally without fancy logic
            for i in range(2, len(self.blocks) - 2):
                x_concat = self.blocks[i](x_concat, c)

        # 7. Run Final 2 Blocks
        for i in range(len(self.blocks) - 2, len(self.blocks)):
            x_concat = self.blocks[i](x_concat, c)

        # 8. Slice output to retrieve only Image tokens for prediction
        x_img_out = x_concat[:, num_text:, :]

        # 9. Final Projection
        x_pred = self.final_layer(x_img_out, c)
        x_pred = self.unpatchify(x_pred)

        # Reshape for ResNet Head
        H_grid = W_grid = int(x.shape[1] ** 0.5)

        # Reshape x_start (from embedder) and x_end (output of backbone)
        x_start_resnet = x_start_resnet.transpose(1, 2).reshape(x.shape[0], self.hidden_size, H_grid, W_grid)
        x_img_out = x_img_out.transpose(1, 2).reshape(x.shape[0], self.hidden_size, H_grid, W_grid)

        return x_pred, x_start_resnet, x_img_out, t_emb


class ResBlockAdaLN(nn.Module):
    def __init__(self, channels, temb_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(16, channels, eps=1e-6)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(16, channels, eps=1e-6)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        # AdaLN modulation: shift, scale
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_channels, 2 * channels)
        )

        # Zero-init second conv for stability
        nn.init.constant_(self.conv2.weight, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)

    def forward(self, x, temb):
        shift, scale = self.adaLN(temb).chunk(2, dim=1)

        h = self.norm1(x)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return x + h


class ResNetHead(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            patch_size,
            hidden_size=1024,
            num_blocks=4,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Normalize + project each stream separately
        self.start_norm = nn.GroupNorm(16, in_channels, eps=1e-6)
        self.end_norm = nn.GroupNorm(16, in_channels, eps=1e-6)

        self.start_proj = nn.Conv2d(in_channels, hidden_size, 3, 1, 1)
        self.end_proj = nn.Conv2d(in_channels, hidden_size, 3, 1, 1)

        # Timestep embedding projection
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_channels, hidden_size),
        )

        self.blocks = nn.ModuleList([
            ResBlockAdaLN(hidden_size, hidden_size)
            for _ in range(num_blocks)
        ])

        self.final_conv = nn.Conv2d(
            hidden_size,
            out_channels * patch_size ** 2,
            3,
            padding=1
        )

        self.pixel_shuffle = nn.PixelShuffle(patch_size)

        # Zero-init final conv (critical)
        nn.init.constant_(self.final_conv.weight, 0)
        nn.init.constant_(self.final_conv.bias, 0)

    def forward(self, x_start, x_end, t_emb):
        # Normalize + project
        x = (
                self.start_proj(self.start_norm(x_start)) +
                self.end_proj(self.end_norm(x_end))
        )

        temb = self.temb_proj(t_emb)

        for block in self.blocks:
            x = block(x, temb)

        x = self.final_conv(x)
        x = self.pixel_shuffle(x)
        return x


class DiT(nn.Module):
    def __init__(
            self,
            input_size=64,
            patch_size=4,
            in_channels=3,
            hidden_size=384,
            depth=6,
            num_heads=6,
            mlp_ratio=4.0,
            text_embed_dim=768,
    ):
        super().__init__()
        self.backbone = DiTBackbone(
            input_size, patch_size, in_channels, hidden_size,
            depth, num_heads, mlp_ratio, text_embed_dim
        )

        self.head = ResNetHead(
            in_channels=hidden_size,
            out_channels=in_channels,
            patch_size=patch_size,
            hidden_size=1024,
            num_blocks=4
        )

    def enable_gradient_checkpointing(self):
        self.backbone.enable_gradient_checkpointing()

    def forward(self, x, t, text_tokens, crop_coords, token_drop_ratio):
        """
        x: (B, C, H, W)
        t: (B,)
        text_tokens: (B, Seq_Len, Text_Dim) - from CLIP/T5
        crop_coords: (B, 4)
        """
        x_pred, x_start, x_end, t_emb = self.backbone(x, t, text_tokens, crop_coords, token_drop_ratio)
        out = self.head(x_start, x_end, t_emb)
        return out, x_pred


# Pos Embed Helpers (Unchanged)
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
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
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb