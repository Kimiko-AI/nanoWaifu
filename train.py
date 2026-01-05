import torch
import torch.nn as nn
from pytorch_optimizer.optimizer import ScheduleFreeAdamW
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
from transformers import CLIPTextModel, CLIPTokenizer

from model import DiT
from dataset import WDSLoader


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


@torch.no_grad()
def encode_text(tokenizer, text_encoder, prompts, device):
    inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
    outputs = text_encoder(**inputs)
    # Use pooled output for simplicity, or we could use sequence output with cross-attention
    return outputs.pooler_output


@torch.no_grad()
def sample_flow(model, vae, tokenizer, text_encoder, latent_size, batch_size, prompts, coords, device, steps=50, cfg_scale=4.0):
    """
    Sample using Euler integration of the flow ODE with Classifier-Free Guidance.
    """
    # Start from noise x_0 (latents)
    x = torch.randn((batch_size, 4, latent_size, latent_size), device=device)

    dt = 1.0 / steps
    indices = torch.linspace(0, 1, steps, device=device)

    # Encode prompts
    cond_embed = encode_text(tokenizer, text_encoder, prompts, device)
    uncond_embed = encode_text(tokenizer, text_encoder, [""] * batch_size, device)

    for i in tqdm(range(steps), desc='Sampling', leave=False):
        t = indices[i]

        # Prepare inputs for batch (cond + uncond)
        x_in = torch.cat([x, x])
        t_batch = torch.full((batch_size * 2,), t.item(), device=device, dtype=torch.float)
        c_in = torch.cat([cond_embed, uncond_embed])
        coords_in = torch.cat([coords, coords])

        # Predict velocity field v_t
        v_pred, _ = model(x_in, t_batch * 1000, c_in, coords_in)

        v_cond, v_uncond = v_pred.chunk(2)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        # Euler step: x_{t+dt} = x_t + v_t * dt
        x = x + v * dt

    # Decode latents back to pixels
    x = x / 0.18215
    samples = vae.decode(x).sample
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

    # Load VAE and Text Encoder
    vae = AutoencoderKL.from_pretrained(config['model']['vae_model']).to(device).eval()
    vae.requires_grad_(False)
    
    tokenizer = CLIPTokenizer.from_pretrained(config['model']['text_encoder_model'])
    text_encoder = CLIPTextModel.from_pretrained(config['model']['text_encoder_model']).to(device).eval()
    text_encoder.requires_grad_(False)

    # Load Data
    wds_loader = WDSLoader(
        url=config['data']['webdataset_url'],
        csv_path=config['data']['csv_path'],
        image_size=config['training']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
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
        # SPRINT parameters
        sprint_enabled=sprint_enabled,
        token_drop_ratio=sprint_config.get('token_drop_ratio', 0.75),
        encoder_depth=sprint_config.get('encoder_depth'),
        middle_depth=sprint_config.get('middle_depth'),
        decoder_depth=sprint_config.get('decoder_depth'),
    ).to(device)

    if config['training'].get('gradient_checkpointing', False):
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled.")

    optimizer = ScheduleFreeAdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['learning_rate'], weight_decay=1e-2)

    # Resume logic (simplified for fresh start on T2I)
    start_epoch = 0
    global_step = 0
    resume_path = config.get('resume_from', "")

    if resume_path and os.path.exists(resume_path):
        # ... existing resume logic could go here if needed ...
        pass

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
        optimizer.train()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images, prompts, coords = batch
        images = images.to(device)
        coords = coords.to(device)

        with torch.no_grad():
            # Encode images to latents
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
            
            # Encode text
            # Randomly drop prompts for CFG
            if config['training'].get('class_dropout_prob', 0.1) > 0:
                indices = torch.rand(len(prompts)) < config['training']['class_dropout_prob']
                train_prompts = [p if not indices[i] else "" for i, p in enumerate(prompts)]
            else:
                train_prompts = prompts
            
            text_embeds = encode_text(tokenizer, text_encoder, train_prompts, device)

        # SPRINT token drop ratio
        if sprint_enabled and two_stage_training:
            current_token_drop_ratio = base_token_drop_ratio if global_step < stage1_steps else 0.0
        else:
            current_token_drop_ratio = base_token_drop_ratio if sprint_enabled else 0.0
        
        # Flow Matching Training
        t = torch.rand((latents.shape[0],), device=device)
        x0 = torch.randn_like(latents)
        x1 = latents
        t_reshaped = t.view(-1, 1, 1, 1)
        xt = (1 - t_reshaped) * x0 + t_reshaped * x1
        ut = x1 - x0

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            v_head, x_backbone = model(xt, t * 1000, text_embeds, coords, token_drop_ratio=current_token_drop_ratio)
            loss_head = torch.mean((v_head - ut) ** 2)
            loss_backbone = torch.mean((x_backbone - x1) ** 2)
            loss = loss_head + loss_backbone

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        global_step += 1

        if rank == 0:
            pbar.update(1)
            logs = {
                "loss": loss.item(),
                "loss_head": loss_head.item(),
                "loss_backbone": loss_backbone.item(),
                "lr": optimizer.param_groups[0]['lr'],
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
                print("\nSampling and Saving Checkpoint...")
                model_to_save = model.module if is_ddp else model
                ckpt_path = os.path.join(config['training']['output_dir'], f'ckpt_step_{global_step}.pth')
                torch.save({"model_state_dict": model_to_save.state_dict(), "global_step": global_step}, ckpt_path)
                cleanup_checkpoints(config['training']['output_dir'], config.get('max_checkpoints', 3), rank)

                model.eval()
                with torch.no_grad():
                    # Sample with first few prompts from batch
                    sample_prompts = prompts[:4]
                    sample_coords = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * len(sample_prompts), device=device)
                    samples = sample_flow(model, vae, tokenizer, text_encoder, latent_size, len(sample_prompts),
                                          sample_prompts, sample_coords, device, cfg_scale=cfg_scale)
                    samples = (samples + 1) / 2.0
                    samples = torch.clamp(samples, 0, 1)
                    grid = make_grid(samples, nrow=2)
                    wandb.log({"samples": wandb.Image(grid, caption=f"Step {global_step}: {sample_prompts[0]}")}, step=global_step)
                model.train()
            if is_ddp: dist.barrier()

    if rank == 0:
        pbar.close()
        wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    train(args.config)