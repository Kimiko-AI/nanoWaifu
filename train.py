import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import yaml
import os
import argparse
import numpy as np
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import wandb
import glob
import builtins
from diffusers import AutoencoderKLFlux2 as AutoencoderKL2Flux
from transformers import AutoTokenizer, AutoModel

from model import DiT
from dataset import WDSLoader
from text_encoder import encode_prompt_with_llm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transport import Transport, Sampler, ModelType, PathType


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return True, rank, local_rank, world_size, device
    else:
        return False, 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def cleanup_checkpoints(output_dir, max_checkpoints, rank):
    if rank != 0: return
    # Find all checkpoints
    checkpoints = glob.glob(os.path.join(output_dir, "ckpt_step_*.pth"))
    # Sort by step number (extracted from filename)
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Remove older checkpoints if we have more than max_checkpoints
    if len(checkpoints) > max_checkpoints:
        checkpoints_to_remove = checkpoints[:-max_checkpoints]
        for ckpt in checkpoints_to_remove:
            try:
                os.remove(ckpt)
                print(f"Removed old checkpoint: {ckpt}")
            except OSError as e:
                print(f"Error removing {ckpt}: {e}")


def save_checkpoint(model, optimizers, rank, output_dir, step, config):
    if rank != 0: return
    print(f"\nSaving Checkpoint at step {step}...")
    model_to_save = model.module if hasattr(model, 'module') else model
    ckpt_path = os.path.join(output_dir, f'ckpt_step_{step}.pth')

    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_muon_state_dict": optimizers[0].state_dict(),
        "optimizer_adamw_state_dict": optimizers[1].state_dict(),
        "global_step": step,
        "config": config,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }

    torch.save(checkpoint, ckpt_path)
    cleanup_checkpoints(output_dir, config.get('max_checkpoints', 3), rank)
    print(f"Checkpoint saved to {ckpt_path}")


@torch.no_grad()
def sample_flow(model, vae, tokenizer, text_encoder, latent_size, batch_size, prompts, coords, device,
                    transport, steps=50, cfg_scale=1.4, sparse_drop_ratio=0.75):
    """
    Sample using Transport module's ODE sampler.
    Uses RK4 integration for accurate flow matching sampling.
    """
    from transport import Sampler
    
    # Initialize sampler
    sampler = Sampler(transport)
    
    # Pre-encode text
    cond_embed, cond_mask = encode_prompt_with_llm(tokenizer, text_encoder, prompts, device)
    
    # Create model wrapper with CFG (Classifier-Free Guidance)
    def model_with_cfg(x, t, **kwargs):
        """Model wrapper that applies CFG between dense and sparse predictions"""
        t_batch = t if t.dim() > 0 else t.unsqueeze(0).expand(x.shape[0])
        
        # Dense prediction (conditioned)
        v_cond = model(x, t_batch, cond_embed, coords, 
                      text_mask=cond_mask, token_drop_ratio=0.0)
        
        # Sparse prediction (SPRINT)
        v_sparse = model(x, t_batch, cond_embed, coords, 
                        text_mask=cond_mask, token_drop_ratio=sparse_drop_ratio)
        
        # Apply CFG: v = v_sparse + cfg_scale * (v_cond - v_sparse)
        return v_sparse + cfg_scale * (v_cond - v_sparse)
    
    # Initial noise
    x = torch.randn((batch_size, 32, latent_size, latent_size), device=device)
    
    # Sample using ODE solver from transport module
    sample_fn = sampler.sample_ode(
        sampling_method='rk4',  # Use RK4 for accuracy
        num_steps=steps,
        atol=1e-5,
        rtol=1e-5,
    )
    
    # Run sampling (returns list of intermediate states)
    samples_list = sample_fn(x, model_with_cfg)
    
    # Get final sample
    x_final = samples_list[-1]
    
    # Decode latents to images
    samples = vae.decode(x_final).sample
    return samples


def train(config_path):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()

    # Suppress printing on non-master ranks
    if rank != 0:
        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Using device: {device}, Rank: {rank}, World Size: {world_size}")

    # Initialize WandB only on rank 0
    if rank == 0:
        wandb.init(project=config.get('wandb_project', 'nanoWaifu-T2I'), config=config)

    # Load VAE and LLM Text Encoder
    vae = AutoencoderKL2Flux.from_pretrained(config['model']['vae_model']).to(device).eval()
    vae.requires_grad_(False)

    # Load LLM for sequence embeddings with cross-attention
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder_model'])
    text_encoder = AutoModel.from_pretrained(config['model']['text_encoder_model']).to(device).eval()
    text_encoder.requires_grad_(False)
    print(f"Using LLM text encoder: {config['model']['text_encoder_model']}")

    # Load Data
    wds_loader = WDSLoader(
        url=config['data']['webdataset_url'],
        csv_path=config['data'].get('csv_path'),
        image_size=config['training']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        use_advanced_captions=config['data'].get('use_advanced_captions', True)
    )
    dataloader = wds_loader.make_loader()

    # SPRINT configuration
    sprint_config = config.get('sprint', {})
    sprint_enabled = sprint_config.get('enabled', False)

    latent_size = config['training']['image_size'] // 8

    model = DiT(
        input_size=latent_size,
        patch_size=config['training']['patch_size'],
        in_channels=config['model']['in_channels'],
        hidden_size=config['model']['dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['heads'],
        mlp_ratio=config['model']['mlp_dim'] / config['model']['dim'],
        context_dim=config['model']['context_dim'],
        use_cross_attn=True,  # Always use cross-attention with LLM
        # SPRINT parameters
        sprint_enabled=sprint_enabled,
        token_drop_ratio=sprint_config.get('token_drop_ratio', 0.75),
        encoder_depth=sprint_config.get('encoder_depth'),
        middle_depth=sprint_config.get('middle_depth'),
        decoder_depth=sprint_config.get('decoder_depth'),
    ).to(device)

    # Initialize Transport for flow matching
    transport = Transport(
        model_type=ModelType.VELOCITY,
        path_type=PathType.LINEAR,
        loss_type=None,  # Not used in current implementation
        train_eps=1e-5,
        sample_eps=1e-5,
        snr_type="uniform",
        do_shift=False,  # Can be enabled for better sampling
        seq_len=latent_size * latent_size,  # Total number of latent tokens
    )
    print("Initialized Transport module for flow matching")

    if config['training'].get('gradient_checkpointing', False):
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled.")

    # Separate parameters by dimensionality for Muon + AdamW hybrid optimizer
    # Muon for >2D params (weight matrices), AdamW for 1D params (biases, norms)
    params_2d = []  # For Muon
    params_1d = []  # For AdamW
    names_2d = []
    names_1d = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Count dimensions that are greater than 1
            meaningful_dims = sum(1 for s in param.shape if s > 1)
            
            if meaningful_dims == 2:
                params_2d.append(param)
                names_2d.append(name)
            else:
                params_1d.append(param)
                names_1d.append(name)
        
    print(f"Muon optimizer: {len(params_2d)} parameters (2D)")
    for name in names_2d:
        print(f"  - {name}")
    
    print(f"\nAdamW optimizer: {len(params_1d)} parameters (1D)")
    for name in names_1d:
        print(f"  - {name}")
    
    # Muon for weight matrices (>2D parameters)
    optimizer_muon = torch.optim.Muon(
        params_2d,
        lr=config['training']['learning_rate'],
        momentum=0.95,
        nesterov=True, adjust_lr_fn = "match_rms_adamw"
    )
    
    # AdamW for biases and normalization parameters (1D parameters)
    optimizer_adamw = AdamW(
        params_1d,
        lr=config['training']['learning_rate'],
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )
    
    # Combine optimizers for easier management
    optimizers = [optimizer_muon, optimizer_adamw]

    # Resume logic (simplified for fresh start on T2I)
    start_epoch = 0
    global_step = 0
    resume_path = config.get('resume_from', "outputs/")

    if resume_path:
        if os.path.isdir(resume_path):
            # If a directory is provided, find the latest checkpoint
            ckpt_files = glob.glob(os.path.join(resume_path, "ckpt_step_*.pth"))
            if ckpt_files:
                resume_path = sorted(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            else:
                resume_path = None

        if resume_path and os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            
            # Get the state dict from the checkpoint
            ckpt_state_dict = checkpoint["model_state_dict"]
            
            # Get the current model's state dict (handle DDP)
            model_to_load = model.module if hasattr(model, 'module') else model
            current_model_dict = model_to_load.state_dict()
        
            # Filter out keys that don't match in shape
            new_state_dict = {}
            for k, v in ckpt_state_dict.items():
                if k in current_model_dict:
                    if v.shape == current_model_dict[k].shape:
                        new_state_dict[k] = v
                    else:
                        print(f"Skipping parameter {k} due to shape mismatch: "
                              f"CKPT {v.shape} vs MODEL {current_model_dict[k].shape}")
                else:
                    print(f"Skipping parameter {k}: Not found in current model.")
        
            # Load the filtered dict
            # We still use strict=False to allow for missing keys we just filtered out
            model_to_load.load_state_dict(new_state_dict, strict=False)
            
            # Load optimizer states if available
            if "optimizer_muon_state_dict" in checkpoint and "optimizer_adamw_state_dict" in checkpoint:
                try:
                   # optimizer_muon.load_state_dict(checkpoint["optimizer_muon_state_dict"])
                    #optimizer_adamw.load_state_dict(checkpoint["optimizer_adamw_state_dict"])
                    print("Loaded optimizer states")
                except Exception as e:
                    print(f"Warning: Could not load optimizer states: {e}")
                    print("Continuing with fresh optimizer states")
            elif "optimizer_state_dict" in checkpoint:
                # Legacy single optimizer checkpoint - skip loading
                print("Warning: Legacy single optimizer checkpoint detected, using fresh optimizer states")
            
            # Load step and RNG
            global_step = checkpoint["global_step"]
            if "rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["rng_state"].cpu())
            if "cuda_rng_state" in checkpoint:
                torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())
            
            print(f"Successfully resumed at step {global_step}")
        else:
            print(f"No checkpoint found at {resume_path}, starting from scratch.")

    # Ensure all ranks start at the same step before proceeding
    if is_ddp:
        dist.barrier()

    # Wrap model in DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    os.makedirs(config['training']['output_dir'], exist_ok=True)

    cfg_scale = config['training'].get('cfg_scale', 4.0)
    max_train_steps = config['training'].get('max_train_steps', 1000000)

    # SPRINT two-stage training schedule
    two_stage_training = sprint_config.get('two_stage_training', False)
    stage1_steps = sprint_config.get('stage1_steps', max_train_steps)
    base_token_drop_ratio = sprint_config.get('token_drop_ratio', 0.75)

    if rank == 0:
        pbar = tqdm(range(global_step, max_train_steps), desc="Steps", dynamic_ncols=True)
    else:
        pbar = None

    data_iter = iter(dataloader)

    # Training Loop
    while global_step < max_train_steps:
        model.train()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images, prompts, coords = batch
        images = images.to(device)
        coords = coords.to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Encode images to latents
                latents = vae.encode(images).latent_dist.sample()

                # Encode text with LLM
                # Randomly drop prompts for CFG
                if config['training'].get('class_dropout_prob', 0.1) > 0:
                    indices = torch.rand(len(prompts)) < config['training']['class_dropout_prob']
                    train_prompts = [p if not indices[i] else "" for i, p in enumerate(prompts)]
                else:
                    train_prompts = prompts

                # Encode with LLM
                text_embeds, text_masks = encode_prompt_with_llm(
                    tokenizer, text_encoder, train_prompts, device,
                    max_sequence_length=config['model'].get('llm_max_seq_length', 64)
                )

        # SPRINT token drop ratio with random no-drop
        if sprint_enabled and two_stage_training:
            current_token_drop_ratio = base_token_drop_ratio if global_step < stage1_steps else 0.0
        else:
            current_token_drop_ratio = base_token_drop_ratio if sprint_enabled else 0.0

        # 10% chance to not drop tokens (helps model learn both sparse and dense scenarios)
        # Use global_step as seed for multi-GPU synchronization
        torch.manual_seed(global_step)
        if torch.rand(1).item() < 0.1:
            current_token_drop_ratio = 0.0

        # Flow Matching Training using Transport module
        # Create a model wrapper that handles the SPRINT token dropping
        def model_fn(xt, t):
            """Wrapper to make model compatible with transport.training_losses"""
            v_head = model(
                xt, t, text_embeds, coords,
                text_mask=text_masks,
                token_drop_ratio=current_token_drop_ratio
            )
            return v_head

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Use transport module for flow matching loss computation
            loss_dict = transport.training_losses(model_fn, latents)
            loss = loss_dict["loss"]
        # Zero gradients for both optimizers
        optimizer_muon.zero_grad()
        optimizer_adamw.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Step both optimizers
        optimizer_muon.step()
        optimizer_adamw.step()


        global_step += 1

        if rank == 0:
            pbar.update(1)
            logs = {
                "loss": loss.item(),
                "task_loss": loss_dict["task_loss"].mean().item() if loss_dict["task_loss"].dim() > 0 else loss_dict["task_loss"].item(),
                "lr_muon": optimizer_muon.param_groups[0]['lr'],
                "lr_adamw": optimizer_adamw.param_groups[0]['lr'],
                "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            }
            if sprint_enabled:
                logs["token_drop_ratio"] = current_token_drop_ratio

            pbar.set_postfix(**logs)
            if global_step % config['training']['log_every_steps'] == 0:
                wandb.log({f"train/{k}": v for k, v in logs.items()}, step=global_step)

        if global_step % config['training']['save_image_every_steps'] == 0:
            if is_ddp: dist.barrier()
            if rank == 0:
                save_checkpoint(model, optimizers, rank, config['training']['output_dir'], global_step, config)

                print("\nSampling...")
                model.eval()
                with torch.no_grad():
                    # Sample with first few prompts from batch
                    sample_prompts = prompts[:16]
                    sample_coords = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * len(sample_prompts), device=device)
                    samples = sample_flow(
                        model, vae, tokenizer, text_encoder, latent_size, len(sample_prompts),
                        sample_prompts, sample_coords, device,
                        transport=transport,  # Pass transport object
                        cfg_scale=cfg_scale
                    )
                    samples = (samples + 1) / 2.0
                    samples = torch.clamp(samples, 0, 1)
                    grid = make_grid(samples, nrow=4)
                    wandb.log({"samples": wandb.Image(grid, caption=f"Step {global_step}: {sample_prompts[0]}")},
                              step=global_step)
                model.train()
            if is_ddp: dist.barrier()

    # Save final checkpoint
    if rank == 0:
        save_checkpoint(model, optimizers, rank, config['training']['output_dir'], global_step, config)
        pbar.close()
        wandb.finish()
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    train(args.config)