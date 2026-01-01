import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import argparse
import numpy as np
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb
import glob

from model import DiT
from dataset import WDSLoader

def cleanup_checkpoints(output_dir, max_checkpoints):
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
    dx/dt = v(x, t)
    x_1 = x_0 + int_0^1 v(x_t, t) dt
    
    v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
    """
    # Start from noise x_0
    x = torch.randn((batch_size, 3, image_size, image_size), device=device)
    
    dt = 1.0 / steps
    indices = torch.linspace(0, 1, steps, device=device)
    
    # Pre-prepare null classes for unconditioned pass (index = num_classes)
    # We can assume model.num_classes is available or pass it. 
    # model.num_classes corresponds to the unconditioned token index.
    null_classes = torch.full_like(classes, model.num_classes, device=device)
    
    for i in tqdm(range(steps), desc='Sampling', leave=False):
        t = indices[i]
        
        # Prepare inputs for batch (cond + uncond)
        x_in = torch.cat([x, x])
        t_batch = torch.full((batch_size * 2,), t.item(), device=device, dtype=torch.float)
        c_in = torch.cat([classes, null_classes])
        coords_in = torch.cat([coords, coords])
        
        # Predict velocity field v_t
        # Scale t by 1000 to match the frequency range expected by the embedder
        v_pred = model(x_in, t_batch * 1000, c_in, coords_in)
        
        v_cond, v_uncond = v_pred.chunk(2)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        # Euler step: x_{t+dt} = x_t + v_t * dt
        x = x + v * dt
        
    return x

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize WandB
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
    
    model = DiT(
        input_size=config['training']['image_size'],
        patch_size=config['training']['patch_size'],
        in_channels=config['model']['in_channels'],
        hidden_size=config['model']['dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['heads'],
        mlp_ratio=config['model']['mlp_dim'] / config['model']['dim'],
        num_classes=num_classes,
        class_dropout_prob=config['training']['class_dropout_prob']
    ).to(device)
    
    # Resume from checkpoint if specified
    resume_path = config.get('resume_from', "")
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        try:
            state_dict = torch.load(resume_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Successfully loaded model weights.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    global_step = 0
    cfg_scale = config['training'].get('cfg_scale', 4.0)

    for epoch in range(config['training']['num_epochs']):
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        for batch in dataloader:
            x1, class_ids, coords = batch
            x1 = x1.to(device)
            class_ids = class_ids.to(device)
            coords = coords.to(device)
            
            # Flow Matching Training
            # 1. Sample t uniform [0, 1]
            t = torch.rand((x1.shape[0],), device=device)
            
            # 2. Sample noise x0
            x0 = torch.randn_like(x1)
            
            # 3. Compute x_t (Linear interpolation / Optimal Transport path)
            # x_t = (1 - t) * x0 + t * x1
            t_reshaped = t.view(-1, 1, 1, 1)
            xt = (1 - t_reshaped) * x0 + t_reshaped * x1
            
            # 4. Target vector field v_t = dx_t/dt = x1 - x0
            ut = x1 - x0
            
            # 5. Predict vector field
            # Scale t by 1000 for embedding frequency compatibility
            vt = model(xt, t * 1000, class_ids, coords)
            
            loss = torch.mean((vt - ut) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            if global_step % config['training']['log_every_steps'] == 0:
                print(f"Step {global_step}, Loss: {loss.item():.4f}")
                wandb.log({"train/loss": loss.item()}, step=global_step)
            
            if global_step % config['training']['save_image_every_steps'] == 0:
                print("Sampling and Saving Checkpoint...")
                
                # Save Checkpoint
                ckpt_path = os.path.join(config['training']['output_dir'], f'ckpt_step_{global_step}.pth')
                torch.save(model.state_dict(), ckpt_path)
                cleanup_checkpoints(config['training']['output_dir'], config.get('max_checkpoints', 3))
                
                # Sample
                model.eval()
                sample_classes = torch.randint(0, num_classes, (4,), device=device)
                sample_coords = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 4, device=device)
                
                samples = sample_flow(model, config['training']['image_size'], 4, sample_classes, sample_coords, device, cfg_scale=cfg_scale)
                
                samples = (samples + 1) / 2.0
                samples = torch.clamp(samples, 0, 1)
                
                grid = make_grid(samples, nrow=2)
                # Log image to wandb
                wandb_image = wandb.Image(grid, caption=f"Sample Step {global_step} (CFG={cfg_scale})")
                wandb.log({"samples": wandb_image}, step=global_step)
                
                model.train()

    print("Training Complete.")
    final_path = os.path.join(config['training']['output_dir'], 'dit_model_final.pth')
    torch.save(model.state_dict(), final_path)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train(args.config)