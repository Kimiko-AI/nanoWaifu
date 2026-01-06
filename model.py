import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
import numpy as np
import math


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TokenDropper(nn.Module):
    """
    Implements structured group-wise token subsampling for SPRINT.
    Maintains both global coverage and local detail by using structured sampling.
    """
    def __init__(self, drop_ratio=0.75):
        super().__init__()
        self.drop_ratio = drop_ratio

    def forward(self, x, drop_ratio=None):
        """
        Args:
            x: (B, N, D) token sequence
            drop_ratio: optional override for drop ratio
        Returns:
            x_sparse: (B, N_sparse, D) dropped tokens
            keep_indices: (B, N_sparse) indices of kept tokens
        """
        if drop_ratio is None:
            drop_ratio = self.drop_ratio

        if drop_ratio == 0.0 or not self.training:
            # No dropping during inference or when ratio is 0
            B, N, D = x.shape
            keep_indices = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            return x, keep_indices

        B, N, D = x.shape
        keep_ratio = 1.0 - drop_ratio
        num_keep = max(1, int(N * keep_ratio))

        # Structured group-wise subsampling
        # Use uniform sampling with slight randomization to maintain coverage
        if self.training:
            # Random sampling with structured pattern
            step = N / num_keep
            base_indices = torch.arange(num_keep, device=x.device).float() * step
            # Add small random offset for diversity
            offsets = torch.rand(B, num_keep, device=x.device) * (step * 0.5)
            indices = (base_indices.unsqueeze(0) + offsets).long()
            indices = torch.clamp(indices, 0, N - 1)
        else:
            # Deterministic uniform sampling for inference
            indices = torch.linspace(0, N - 1, num_keep, device=x.device).long()
            indices = indices.unsqueeze(0).expand(B, -1)

        # Gather tokens
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
        x_sparse = torch.gather(x, 1, indices_expanded)

        return x_sparse, indices


def scatter_tokens(x_sparse, indices, full_shape, fill_value=0.0):
    """
    Scatter sparse tokens back to full sequence.

    Args:
        x_sparse: (B, N_sparse, D) sparse tokens
        indices: (B, N_sparse) indices where tokens should be placed
        full_shape: (B, N_full, D) target shape
        fill_value: value to fill non-selected positions
    Returns:
        x_full: (B, N_full, D) full sequence with scattered tokens
    """
    B, N_full, D = full_shape
    x_full = torch.full((B, N_full, D), fill_value, device=x_sparse.device, dtype=x_sparse.dtype)

    # Scatter tokens back
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
    x_full.scatter_(1, indices_expanded, x_sparse)

    return x_full



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
    """
    Original DiTBlock with AdaLN modulation (kept for text refiner).
    """
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


class DiTBlockWithCrossAttention(nn.Module):
    """
    DiT Block with cross-attention to text embeddings instead of AdaLN.
    Supports attention masks for variable-length text sequences.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_timestep_cond=True):
        super().__init__()
        self.use_timestep_cond = use_timestep_cond

        # Self-attention for image tokens
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)

        # Cross-attention to text tokens
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)

        # MLP
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        # Optional timestep conditioning via addition
        if use_timestep_cond:
            self.timestep_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size)
            )

    def forward(self, x, text_embeds, text_mask=None, timestep_emb=None):
        """
        Args:
            x: (B, N_img, D) image tokens
            text_embeds: (B, N_text, D) text embeddings
            text_mask: (B, N_text) attention mask for text (True = attend, False = ignore)
            timestep_emb: (B, D) timestep embedding (optional)
        """
        # Timestep conditioning (additive)
        if self.use_timestep_cond and timestep_emb is not None:
            t_cond = self.timestep_proj(timestep_emb).unsqueeze(1)  # (B, 1, D)
            x = x + t_cond

        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Cross-attention to text
        x_norm = self.norm2(x)
        # Convert mask: True (attend) -> False (not masked), False (ignore) -> True (masked)
        if text_mask is not None:
            attn_mask = ~text_mask  # Invert for PyTorch convention
        else:
            attn_mask = None

        cross_out, _ = self.cross_attn(
            query=x_norm,
            key=text_embeds,
            value=text_embeds,
           # key_padding_mask=attn_mask
        )
        x = x + cross_out

        # MLP
        x_norm = self.norm3(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x


class TextRefiner(nn.Module):
    """
    Refines LLM text embeddings using 2 layers of self-attention DiTBlocks.
    Uses AdaLN conditioning with timestep and crop coordinates.
    """
    def __init__(self, text_dim, hidden_size, num_heads, num_layers=2, mlp_ratio=4.0):
        super().__init__()
        self.text_dim = text_dim
        self.hidden_size = hidden_size

        # Project text embeddings to hidden size
        self.text_proj = nn.Linear(text_dim, hidden_size)

        # Self-attention blocks to refine text (with AdaLN)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])

    def forward(self, text_embeds, t_coord_emb):
        """
        Args:
            text_embeds: (B, N_text, text_dim) LLM embeddings
            t_coord_emb: (B, hidden_size) combined timestep + coord embeddings
        Returns:
            refined_embeds: (B, N_text, hidden_size) refined embeddings
        """
        # Project to hidden size
        x = self.text_proj(text_embeds)  # (B, N_text, hidden_size)

        # Process through blocks with AdaLN conditioning
        for block in self.blocks:
            x = block(x, t_coord_emb)

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    Supports both AdaLN mode (original) and cross-attention mode.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_cross_attn=False):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

        if not use_cross_attn:
            # Original AdaLN mode
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
        else:
            # Cross-attention mode: just use simple scaling
            self.scale = nn.Parameter(torch.ones(hidden_size))
            self.shift = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, c=None):
        if not self.use_cross_attn and c is not None:
            # AdaLN mode
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
        else:
            # Cross-attention mode or no conditioning
            x = self.norm_final(x)
            x = x * self.scale + self.shift

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
    Now supports cross-attention to LLM text embeddings.
    """
    def __init__(
        self,
        input_size=32, # Latent size (e.g. 256/8)
        patch_size=2,
        in_channels=4, # Latent channels
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        context_dim=768, # LLM embedding dimension
        use_cross_attn=True, # Use cross-attention instead of AdaLN
        # SPRINT parameters
        sprint_enabled=False,
        token_drop_ratio=0.75,
        encoder_depth=None,
        middle_depth=None,
        decoder_depth=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.context_dim = context_dim
        self.use_cross_attn = use_cross_attn

        # SPRINT configuration
        self.sprint_enabled = sprint_enabled
        self.token_drop_ratio = token_drop_ratio

        # Partition blocks into encoder/middle/decoder
        if sprint_enabled and encoder_depth is not None:
            assert encoder_depth + middle_depth + decoder_depth == depth, \
                f"encoder_depth ({encoder_depth}) + middle_depth ({middle_depth}) + decoder_depth ({decoder_depth}) must equal depth ({depth})"
            self.encoder_depth = encoder_depth
            self.middle_depth = middle_depth
            self.decoder_depth = decoder_depth
        else:
            # Default: no partitioning, all blocks are "encoder"
            self.encoder_depth = depth
            self.middle_depth = 0
            self.decoder_depth = 0

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Text refiner: processes LLM embeddings with 2 DiTBlocks
        if use_cross_attn:
            self.text_refiner = TextRefiner(
                text_dim=context_dim,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=2,
                mlp_ratio=mlp_ratio
            )
        else:
            # Fallback: simple projection for AdaLN mode
            self.text_embedder = nn.Sequential(
                nn.Linear(context_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        self.coord_embedder = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Create blocks partitioned into encoder/middle/decoder
        if use_cross_attn:
            # Use cross-attention blocks
            self.encoder_blocks = nn.ModuleList([
                DiTBlockWithCrossAttention(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_timestep_cond=True)
                for _ in range(self.encoder_depth)
            ])

            self.middle_blocks = nn.ModuleList([
                DiTBlockWithCrossAttention(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_timestep_cond=True)
                for _ in range(self.middle_depth)
            ])

            self.decoder_blocks = nn.ModuleList([
                DiTBlockWithCrossAttention(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_timestep_cond=True)
                for _ in range(self.decoder_depth)
            ])
        else:
            # Use original AdaLN blocks
            self.encoder_blocks = nn.ModuleList([
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(self.encoder_depth)
            ])

            self.middle_blocks = nn.ModuleList([
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(self.middle_depth)
            ])

            self.decoder_blocks = nn.ModuleList([
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(self.decoder_depth)
            ])

        # Token dropper for SPRINT
        if sprint_enabled:
            self.token_dropper = TokenDropper(drop_ratio=token_drop_ratio)
            # Residual projection to combine encoder and decoder paths
            if self.decoder_depth > 0:
                self.residual_proj = nn.Linear(hidden_size, hidden_size)

        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels=in_channels, use_cross_attn=use_cross_attn)

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

        # Initialize text embedder/refiner
        if self.use_cross_attn:
            # Initialize text_refiner
            nn.init.normal_(self.text_refiner.text_proj.weight, std=0.02)
            nn.init.constant_(self.text_refiner.text_proj.bias, 0)
        else:
            # Initialize text_embedder for AdaLN mode
            for layer in self.text_embedder:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    nn.init.constant_(layer.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize all blocks (encoder, middle, decoder)
        all_blocks = list(self.encoder_blocks) + list(self.middle_blocks) + list(self.decoder_blocks)
        for block in all_blocks:
            if isinstance(block, DiTBlock):
                # AdaLN blocks
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            elif isinstance(block, DiTBlockWithCrossAttention):
                # Cross-attention blocks - initialize timestep projection
                if block.use_timestep_cond:
                    nn.init.normal_(block.timestep_proj[-1].weight, std=0.02)
                    nn.init.constant_(block.timestep_proj[-1].bias, 0)

        # Initialize SPRINT residual projection if exists
        if self.sprint_enabled and hasattr(self, 'residual_proj'):
            nn.init.xavier_uniform_(self.residual_proj.weight)
            if self.residual_proj.bias is not None:
                nn.init.constant_(self.residual_proj.bias, 0)

        # Zero-out output layers:
        if not self.use_cross_attn:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        c = self.in_channels

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, text_embed, crop_coords, text_mask=None, token_drop_ratio=None):
        """
        x: (N, C, H, W) latents
        t: (N,) timesteps
        text_embed: (N, N_text, context_dim) LLM sequence embeddings OR (N, context_dim) pooled embeddings
        crop_coords: (N, 4) relative coordinates
        text_mask: (N, N_text) attention mask for text (optional, for cross-attention mode)
        """
        x = self.x_embedder(x) + self.pos_embed

        # Save initial embedding for skip connection (N, T, D)
        x_start = x

        t_emb = self.t_embedder(t)
        coord_emb = self.coord_embedder(crop_coords)

        if self.use_cross_attn:
            # Cross-attention mode: refine text embeddings
            # text_embed should be (N, N_text, context_dim)
            if text_embed.dim() == 2:
                # If pooled, unsqueeze to (N, 1, context_dim)
                text_embed = text_embed.unsqueeze(1)

            # Combine timestep and coord embeddings for conditioning
            t_coord_emb = t_emb + coord_emb  # (N, hidden_size)

            # Refine text through text_refiner with timestep/coord conditioning
            text_refined = self.text_refiner(text_embed, t_coord_emb)  # (N, N_text, hidden_size)
        else:
            # AdaLN mode: pool text embeddings and combine with timestep
            if text_embed.dim() == 3:
                # If sequence, pool it
                text_embed = text_embed.mean(dim=1)

            y_emb = self.text_embedder(text_embed)
            c = t_emb + y_emb + coord_emb

        # SPRINT: Encoder (Dense Shallow Path)
        for block in self.encoder_blocks:
            if self.use_cross_attn:
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        block, x, text_refined, text_mask, t_coord_emb, use_reentrant=False
                    )
                else:
                    x = block(x, text_refined, text_mask, t_coord_emb)
            else:
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(block, x, c, use_reentrant=False)
                else:
                    x = block(x, c)

        # Save encoder output for residual connection
        x_encoder = x

        # SPRINT: Middle (Sparse Deep Path) with token dropping
        if self.sprint_enabled and self.middle_depth > 0:
            # Apply token dropping
            if token_drop_ratio is None:
                token_drop_ratio = self.token_drop_ratio
            x_sparse, keep_indices = self.token_dropper(x, drop_ratio=token_drop_ratio)

            # Process sparse tokens through middle blocks
            for block in self.middle_blocks:
                if self.use_cross_attn:
                    if self.gradient_checkpointing and self.training:
                        x_sparse = torch.utils.checkpoint.checkpoint(
                            block, x_sparse, text_refined, text_mask, t_coord_emb, use_reentrant=False
                        )
                    else:
                        x_sparse = block(x_sparse, text_refined, text_mask, t_coord_emb)
                else:
                    if self.gradient_checkpointing and self.training:
                        x_sparse = torch.utils.checkpoint.checkpoint(block, x_sparse, c, use_reentrant=False)
                    else:
                        x_sparse = block(x_sparse, c)

            # Scatter sparse tokens back to full sequence
            x_middle = scatter_tokens(x_sparse, keep_indices, x.shape, fill_value=0.0)
        else:
            x_middle = x

        # SPRINT: Decoder (Dense Path with Residual Fusion)
        if self.sprint_enabled and self.decoder_depth > 0:
            # Fuse encoder and middle outputs via residual connection
            x = x_middle + self.residual_proj(x_encoder)
        else:
            x = x_middle


        for block in self.decoder_blocks:
            if self.use_cross_attn:
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        block, x, text_refined, text_mask, t_coord_emb, use_reentrant=False
                    )
                else:
                    x = block(x, text_refined, text_mask, t_coord_emb)
            else:
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(block, x, c, use_reentrant=False)
                else:
                    x = block(x, c)

        # Compute DiT prediction
        if self.use_cross_attn:
            x_pred = self.final_layer(x)
        else:
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
    Now supports cross-attention to LLM text embeddings.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        context_dim=768,
        use_cross_attn=True,
        # SPRINT parameters
        sprint_enabled=False,
        token_drop_ratio=0.75,
        encoder_depth=None,
        middle_depth=None,
        decoder_depth=None,
    ):
        super().__init__()
        self.sprint_enabled = sprint_enabled
        self.token_drop_ratio = token_drop_ratio
        self.use_cross_attn = use_cross_attn

        self.backbone = DiTBackbone(
            input_size, patch_size, in_channels, hidden_size,
            depth, num_heads, mlp_ratio, context_dim,
            use_cross_attn=use_cross_attn,
            sprint_enabled=sprint_enabled,
            token_drop_ratio=token_drop_ratio,
            encoder_depth=encoder_depth,
            middle_depth=middle_depth,
            decoder_depth=decoder_depth,
        )

        self.head = ResNetHead(
            in_channels=hidden_size, # Backbone output dim
            out_channels=in_channels, # Latent channels
            patch_size=patch_size,
            hidden_size=1024,
            num_blocks=3
        )

    def enable_gradient_checkpointing(self):
        self.backbone.enable_gradient_checkpointing()

    def forward(self, x, t, text_embed, crop_coords, text_mask=None, token_drop_ratio=None):
        # DiT Backbone forward
        x_pred, x_start, x_end, t_emb = self.backbone(
            x, t, text_embed, crop_coords,
            text_mask=text_mask,
            token_drop_ratio=token_drop_ratio
        )

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