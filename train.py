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
from diffusers import AutoencoderKL
from transformers import AutoTokenizer, AutoModel

from model import DiT
from dataset import WDSLoader
from text_encoder import encode_prompt_with_llm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


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
    checkpoints = glob.glob(os.path.join(output_dir, "ckpt_step_*.pth"))
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(checkpoints) > max_checkpoints:
        checkpoints_to_remove = checkpoints[:-max_checkpoints]
        for ckpt in checkpoints_to_remove:
            try:
                os.remove(ckpt)
                print(f"Removed old checkpoint: {ckpt}")
            except OSError as e:
                print(f"Error removing {ckpt}: {e}")


def save_checkpoint(model, optimizers, rank, output_dir, step, config, fixed_prompts=None, fixed_noise=None):
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
        "fixed_prompts": fixed_prompts,
        "fixed_noise": fixed_noise
    }

    torch.save(checkpoint, ckpt_path)
    cleanup_checkpoints(output_dir, config.get('max_checkpoints', 3), rank)
    print(f"Checkpoint saved to {ckpt_path}")


@torch.no_grad()
def sample_flow(model, vae, tokenizer, text_encoder, latent_size, batch_size, prompts, coords, device,
                steps=50, cfg_scale=1.4, sparse_drop_ratio=1.0, noise=None):
    """
    Sample using RK4 integration.
    Args:
        noise: Optional (B, C, H, W) tensor. If provided, uses this exact noise instead of sampling random noise.
    """
    # 1. Initialize Latent Gaussian Noise
    if noise is not None:
        # Clone to ensure we don't modify the stored fixed noise in place during integration
        x = noise.clone().to(device)
        # Ensure batch size matches provided noise
        if x.shape[0] != batch_size:
            # If we have more noise than needed, slice it. If less, repeat it.
            if x.shape[0] > batch_size:
                x = x[:batch_size]
            else:
                x = x.repeat(int(batch_size / x.shape[0]) + 1, 1, 1, 1)[:batch_size]
    else:
        x = torch.randn((batch_size, 16, latent_size, latent_size), device=device)

    dt = 1.0 / steps

    # 2. Pre-encode text
    text_tokens, _ = encode_prompt_with_llm(tokenizer, text_encoder, prompts, device)

    if coords.shape[0] != batch_size:
        coords = coords.repeat(batch_size, 1)

    def get_v(x_current, t_current):
        t_scaled = torch.full((x_current.shape[0],), t_current, device=device)

        _, v_cond = model(x_current, t_scaled, text_tokens, coords, token_drop_ratio=0)
        _, v_sparse = model(x_current, t_scaled, text_tokens, coords, token_drop_ratio=0.75)

        return v_sparse + cfg_scale * (v_cond - v_sparse)

    # 3. RK4 Loop
    for i in tqdm(range(steps), desc='RK4 Sampling', leave=False):
        t = i / steps
        k1 = get_v(x, t)
        k2 = get_v(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = get_v(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = get_v(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    latents = x / vae.config.scaling_factor + vae.config.shift_factor
    images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    return images


def train(config_path):
    is_ddp, rank, local_rank, world_size, device = setup_ddp()
    # seed_everything(42)

    if rank != 0:
        def print_pass(*args, **kwargs): pass

        builtins.print = print_pass

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Using device: {device}, Rank: {rank}, World Size: {world_size}")

    if rank == 0:
        wandb.init(project=config.get('wandb_project', 'nanoWaifu-T2I'), config=config)

    # Load Models
    vae = AutoencoderKL.from_pretrained(config['model']['vae_model']).to(device).eval()
    vae.requires_grad_(False)

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
        text_embed_dim=config['model']['context_dim'],
    ).to(device)

    if config['training'].get('gradient_checkpointing', False):
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled.")

    # Optimizer setup
    params_2d = []
    params_1d = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            meaningful_dims = sum(1 for s in param.shape if s > 1)
            if meaningful_dims == 2:
                params_2d.append(param)
            else:
                params_1d.append(param)

    optimizer_muon = torch.optim.Muon(
        params_2d, lr=config['training']['learning_rate'],
        momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw"
    )

    optimizer_adamw = AdamW(
        params_1d, lr=config['training']['learning_rate'],
        weight_decay=0.1, betas=(0.9, 0.95),
    )

    optimizers = [optimizer_muon, optimizer_adamw]

    # --- RESUME LOGIC & FIXED DATA SETUP ---
    start_epoch = 0
    global_step = 0
    resume_path = config.get('resume_from', "outputs/")

    # Initialize containers for fixed data
    fixed_prompts = None
    fixed_noise = None

    if resume_path:
        if os.path.isdir(resume_path):
            ckpt_files = glob.glob(os.path.join(resume_path, "ckpt_step_*.pth"))
            if ckpt_files:
                resume_path = sorted(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            else:
                resume_path = None

        if resume_path and os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)

            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(checkpoint["model_state_dict"], strict=False)

            # Load optimizers
            if "optimizer_muon_state_dict" in checkpoint and "optimizer_adamw_state_dict" in checkpoint:
                try:
                    optimizer_muon.load_state_dict(checkpoint["optimizer_muon_state_dict"])
                    optimizer_adamw.load_state_dict(checkpoint["optimizer_adamw_state_dict"])
                except Exception as e:
                    print(f"Warning: Could not load optimizer states: {e}")

            global_step = checkpoint["global_step"] - 10
            if "rng_state" in checkpoint: torch.set_rng_state(checkpoint["rng_state"].cpu())
            if "cuda_rng_state" in checkpoint: torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())

            # Load fixed data if available
            # if "fixed_prompts" in checkpoint and checkpoint["fixed_prompts"] is not None:
            #    fixed_prompts = checkpoint["fixed_prompts"]
            #    print(f"Loaded {len(fixed_prompts)} fixed prompts from checkpoint.")

            if "fixed_noise" in checkpoint and checkpoint["fixed_noise"] is not None:
                fixed_noise = checkpoint["fixed_noise"].to(device)
                print("Loaded fixed noise from checkpoint.")

            print(f"Successfully resumed at step {global_step}")
        else:
            print(f"No checkpoint found at {resume_path}, starting from scratch.")

    if is_ddp:
        dist.barrier()
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    os.makedirs(config['training']['output_dir'], exist_ok=True)
    cfg_scale = config['training'].get('cfg_scale', 4.0)
    max_train_steps = config['training'].get('max_train_steps', 1000000)

    # SPRINT settings
    two_stage_training = sprint_config.get('two_stage_training', False)
    stage1_steps = sprint_config.get('stage1_steps', max_train_steps)
    base_token_drop_ratio = sprint_config.get('token_drop_ratio', 0.75)

    if rank == 0:
        pbar = tqdm(range(global_step, max_train_steps), desc="Steps", dynamic_ncols=True)
    else:
        pbar = None

    data_iter = iter(dataloader)

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
        if rank == 0 and fixed_prompts is None:
            # Look for prompts containing "general"
            general_candidates = [p for p in prompts if "general" in p]

            # If we found at least 16, lock them in
            if len(general_candidates) >= 16:
                fixed_prompts = general_candidates[:16]
                fixed_noise = torch.randn((16, 16, latent_size, latent_size), device=device)
                print(f"\n[Step {global_step}] Found and locked 16 'general' prompts for fixed validation.")
            # Optional: if you want to take fewer than 16 if that's all that exists, you can modify logic here.

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                latents = (vae.encode(
                    images).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
                text_embeds, text_mask = encode_prompt_with_llm(
                    tokenizer, text_encoder, prompts, device,
                    max_sequence_length=config['model'].get('llm_max_seq_length', 64)
                )
                dropout_prob = config['training'].get('class_dropout_prob', 0.1)
                if dropout_prob > 0:
                    batch_size_curr = text_embeds.shape[0]
                    drop_mask = torch.rand(batch_size_curr, device=device) < dropout_prob
                    text_embeds[drop_mask] = 0.0

        if sprint_enabled and two_stage_training:
            current_token_drop_ratio = base_token_drop_ratio if global_step < stage1_steps else 0.0
        else:
            current_token_drop_ratio = base_token_drop_ratio if sprint_enabled else 0.0

        torch.manual_seed(global_step)
        if torch.rand(1).item() < 0.1:
            current_token_drop_ratio = 0.0

        t = torch.rand((latents.shape[0],), device=device)
        x0 = torch.randn_like(latents)
        x1 = latents
        t_reshaped = t.view(-1, 1, 1, 1)
        xt = (1 - t_reshaped) * x0 + t_reshaped * x1
        ut = x1 - x0

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            v_head, v_base = model(
                xt, t, text_embeds, coords,
                token_drop_ratio=current_token_drop_ratio
            )
            loss_head = ((v_head - ut) ** 2).mean()
            loss_base = ((v_base - ut) ** 2).mean()
            loss = loss_head + loss_base * 1 / 4

        optimizer_muon.zero_grad()
        optimizer_adamw.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_muon.step()
        optimizer_adamw.step()

        global_step += 1

        if rank == 0:
            pbar.update(1)
            logs = {
                "loss": loss.item(),
                "loss_head": loss_head.item(),
                "loss_base": loss_base.item(),
                "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            }
            pbar.set_postfix(**logs)
            if global_step % config['training']['log_every_steps'] == 0:
                wandb.log({f"train/{k}": v for k, v in logs.items()}, step=global_step)

        if global_step % config['training']['save_image_every_steps'] == 0:
            if is_ddp: dist.barrier()
            if rank == 0:
                # Save fixed data (prompts/noise) to the checkpoint for future runs
                save_checkpoint(model, optimizers, rank, config['training']['output_dir'],
                                global_step, config, fixed_prompts, fixed_noise)

                print("\nSampling...")
                model.eval()
                with torch.no_grad():
                    # Determine what to use for sampling
                    if fixed_prompts is not None and fixed_noise is not None:
                        # Case A: We found our 'general' prompts and locked them
                        use_prompts = fixed_prompts
                        use_noise = fixed_noise
                        caption_prefix = "Fixed"
                    else:
                        # Case B: We haven't found 16 'general' prompts yet.
                        # Fallback to current batch + random noise just to see *something*
                        use_prompts = prompts[:16]
                        use_noise = None  # Will generate random inside sample_flow
                        caption_prefix = "Random (Waiting for General)"

                    sample_coords = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * len(use_prompts), device=device)

                    samples = sample_flow(
                        model, vae, tokenizer, text_encoder, latent_size, len(use_prompts),
                        use_prompts, sample_coords, device,
                        cfg_scale=cfg_scale,
                        noise=use_noise
                    )

                    # samples = (samples + 1) / 2.0
                    # samples = torch.clamp(samples, 0, 1)
                    grid = make_grid(samples, nrow=4)

                    wandb.log({"samples": wandb.Image(grid,
                                                      caption=f"Step {global_step} [{caption_prefix}]: {use_prompts[0]}...")},
                              step=global_step)
                model.train()
            if is_ddp: dist.barrier()

    if rank == 0:
        save_checkpoint(model, optimizers, rank, config['training']['output_dir'],
                        global_step, config, fixed_prompts, fixed_noise)
        pbar.close()
        wandb.finish()
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    train(args.config)