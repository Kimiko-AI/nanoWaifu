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
def sample_flow(model, image_size, batch_size, classes, coords, device, steps=50, cfg_scale=4.0):
    """
    Sample using Euler integration of the flow ODE with Classifier-Free Guidance.
    """
    # Handle DDP model wrapper
    model_engine = model.module if isinstance(model, DDP) else model

    # Start from noise x_0
    x = torch.randn((batch_size, 3, image_size, image_size), device=device)

    dt = 1.0 / steps
    indices = torch.linspace(0, 1, steps, device=device)

    # Pre-prepare null classes for unconditioned pass
    null_classes = torch.full_like(classes, model_engine.num_classes, device=device)

    for i in tqdm(range(steps), desc='Sampling', leave=False):
        t = indices[i]

        # Prepare inputs for batch (cond + uncond)
        x_in = torch.cat([x, x])
        t_batch = torch.full((batch_size * 2,), t.item(), device=device, dtype=torch.float)
        c_in = torch.cat([classes, null_classes])
        coords_in = torch.cat([coords, coords])

        # Predict velocity field v_t
        v_pred, _ = model(x_in, t_batch * 1000, c_in, coords_in)

        v_cond, v_uncond = v_pred.chunk(2)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        # Euler step: x_{t+dt} = x_t + v_t * dt
        x = x + v * dt

    return x


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
        wandb.init(project=config.get('wandb_project', 'nanoWaifu-DiT'), config=config)

    # Load Data
    wds_loader = WDSLoader(
        url=config['data']['webdataset_url'],
        csv_path=config['data']['csv_path'],
        image_size=config['training']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    dataloader = wds_loader.make_loader()

    # Init Model
    num_classes = wds_loader.num_classes

    # SPRINT configuration
    sprint_config = config.get('sprint', {})
    sprint_enabled = sprint_config.get('enabled', False)
    
    model = DiT(
        input_size=config['training']['image_size'],
        patch_size=config['training']['patch_size'],
        in_channels=config['model']['in_channels'],
        hidden_size=config['model']['dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['heads'],
        mlp_ratio=config['model']['mlp_dim'] / config['model']['dim'],
        num_classes=num_classes,
        class_dropout_prob=config['training']['class_dropout_prob'],
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

    if config['training'].get('freeze_backbone', False):
        print("Freezing backbone parameters...")
        for param in model.backbone.parameters():
            param.requires_grad = False

    optimizer = ScheduleFreeAdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['learning_rate'], weight_decay=1e-2)

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    resume_path = config.get('resume_from', "/workspace/shinon/t2i/nanoWaifu/outputs/ckpt_step_65000.pth")

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=device)

            # Extract state dict whether it's wrapped or raw
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            model_state = model.state_dict()
            new_state_dict = {}

            # Remap keys
            for k, v in state_dict.items():
                target_key = None
                # Try direct match
                if k in model_state:
                    target_key = k
                # Try backbone prefix match (Old DiT -> New DiTBackbone)
                elif f"backbone.{k}" in model_state:
                    target_key = f"backbone.{k}"

                if target_key:
                    if model_state[target_key].shape == v.shape:
                        new_state_dict[target_key] = v
                    else:
                        print(f"Skipping key {k} -> {target_key} due to shape mismatch: {v.shape} vs {model_state[target_key].shape}")
                
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded checkpoint. Missing keys (expected for new head): {len(missing)}")

            if new_state_dict:
                print(f"Loaded keys: {len(new_state_dict)}")
                for key in sorted(new_state_dict.keys()):
                    print(f"  + {key}")

            if missing:
                print("Missing keys:")
                for key in sorted(missing):
                    print(f"  - {key}")
            
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
                for key in sorted(unexpected):
                    print(f"  - {key}")

            # Attempt to load optimizer state if available and compatible
            if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    print("Loaded optimizer state.")
                except Exception as e:
                    print(f"Could not load optimizer state (expected since architecture changed): {e}")

            # Load global step
            if isinstance(checkpoint, dict) and "global_step" in checkpoint:
                global_step = checkpoint["global_step"]
                print(f"Resuming from global step: {global_step}")

            print("Successfully loaded checkpoint.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    # Wrap model in DDP
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    # model.compile() # Optional, can enable if needed
    os.makedirs(config['training']['output_dir'], exist_ok=True)

    cfg_scale = config['training'].get('cfg_scale', 4.0)
    
    # SPRINT two-stage training schedule
    two_stage_training = sprint_config.get('two_stage_training', False)
    stage1_steps = sprint_config.get('stage1_steps', max_train_steps)
    stage2_steps = sprint_config.get('stage2_steps', 0)
    base_token_drop_ratio = sprint_config.get('token_drop_ratio', 0.75)

    # Training Loop
    # Calculate max_train_steps if not explicitly provided
    max_train_steps = config['training'].get('max_train_steps', config['training']['num_epochs'] * 1000)

    # Create progress bar only on rank 0
    if rank == 0:
        pbar = tqdm(range(global_step, max_train_steps), desc="Steps", dynamic_ncols=True)
    else:
        pbar = None

    # Create iterator
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

        x1, class_ids, coords = batch
        x1 = x1.to(device)
        class_ids = class_ids.to(device)
        coords = coords.to(device)

        # SPRINT: Determine current token drop ratio based on training stage
        if sprint_enabled and two_stage_training:
            if global_step < stage1_steps:
                # Stage 1: Use configured token drop ratio
                current_token_drop_ratio = base_token_drop_ratio
            else:
                # Stage 2: No token dropping (fine-tuning)
                current_token_drop_ratio = 0.0
        else:
            current_token_drop_ratio = base_token_drop_ratio if sprint_enabled else 0.0
        
        # Flow Matching Training
        t = torch.rand((x1.shape[0],), device=device)
        x0 = torch.randn_like(x1)
        t_reshaped = t.view(-1, 1, 1, 1)
        xt = (1 - t_reshaped) * x0 + t_reshaped * x1
        ut = x1 - x0

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            v_head, x_backbone = model(xt, t * 1000, class_ids, coords, token_drop_ratio=current_token_drop_ratio)
            loss_head = torch.mean((v_head - ut) ** 2)
            loss_backbone = torch.mean((x_backbone - x1) ** 2)
            loss = loss_head + loss_backbone

        optimizer.zero_grad()
        loss.backward()

        # Add gradient clipping for stability and get norm for logging
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        global_step += 1

        # Update progress bar and logs
        if rank == 0:
            pbar.update(1)

            current_lr = optimizer.param_groups[0]['lr']
            logs = {
                "loss": loss.item(),
                "loss_head_v": loss_head.item(),
                "loss_backbone_x": loss_backbone.item(),
                "lr": current_lr,
                "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            }
            
            # Add SPRINT-specific logs
            if sprint_enabled:
                logs["token_drop_ratio"] = current_token_drop_ratio
                if two_stage_training:
                    current_stage = 1 if global_step < stage1_steps else 2
                    logs["training_stage"] = current_stage
            
            pbar.set_postfix(**logs)

            # Log to W&B (only on rank 0)
            if global_step % config['training']['log_every_steps'] == 0:
                wandb_log = {f"train/{k}": v for k, v in logs.items()}
                wandb.log(wandb_log, step=global_step)

        # Sample and save checkpoint (only on rank 0)
        if global_step % config['training']['save_image_every_steps'] == 0:
            # Synchronize all processes before checkpoint
            if is_ddp:
                dist.barrier()

            if rank == 0:
                print("\nSampling and Saving Checkpoint...")

                # Save Checkpoint (Unwrap DDP)
                model_to_save = model.module if is_ddp else model
                ckpt_state = {
                    "model_state_dict": model_to_save.state_dict(),
                    "global_step": global_step,
                    "config": config
                }
                ckpt_path = os.path.join(config['training']['output_dir'], f'ckpt_step_{global_step}.pth')
                torch.save(ckpt_state, ckpt_path)
                cleanup_checkpoints(config['training']['output_dir'], config.get('max_checkpoints', 3), rank)

                # Sample
                model.eval()
                optimizer.eval()
                with torch.no_grad():
                    sample_classes = torch.randint(0, num_classes, (4,), device=device)
                    sample_coords = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 4, device=device)
                    samples = sample_flow(model, config['training']['image_size'], 4,
                                          sample_classes, sample_coords, device, cfg_scale=cfg_scale)
                    samples = (samples + 1) / 2.0
                    samples = torch.clamp(samples, 0, 1)
                    grid = make_grid(samples, nrow=2)
                    wandb_image = wandb.Image(grid, caption=f"Sample Step {global_step} (CFG={cfg_scale})")
                    wandb.log({"samples": wandb_image}, step=global_step)

                model.train()
                optimizer.train()
                print("Checkpoint and sampling complete.\n")

            # Synchronize again after checkpoint
            if is_ddp:
                dist.barrier()

    print("Training Complete.")
    if rank == 0:
        pbar.close()
        final_path = os.path.join(config['training']['output_dir'], 'dit_model_final.pth')
        model_to_save = model.module if is_ddp else model
        torch.save(model_to_save.state_dict(), final_path)
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    train(args.config)