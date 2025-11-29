import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from diffusers import AutoencoderKL
import os
import numpy as np
import random
import wandb
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import sys
import importlib

# Import from local files
from data import get_wds_loader, get_food101_loader # Import get_food101_loader
from flow import xFlowMatching
from config import Config

# Optimize for Tensor Cores
torch.set_float32_matmul_precision('high')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        print(f"[DDP] Initialized Process {rank}/{world_size} (Local: {local_rank})")
        return rank, local_rank, world_size
    else:
        print("[DDP] Not detected. Running in single-process mode.")
        return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


class Trainer:
    def __init__(self, config: Config, rank=0, local_rank=0, world_size=1):
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size

        self.device = torch.device(f"cuda:{local_rank}")
        self.target_device = self.device
        self.saved_checkpoints = []  # List of (loss, path)
        
        self.vae = None
        self.tokenizer = None
        self.text_encoder = None
        self.class_embedding = None
        self.null_class_embedding = None
        self.context_dim = None

        # Create output directory (only main process)
        if is_main_process():
            os.makedirs(self.config.output_dir, exist_ok=True)
            # Initialize WandB
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__
            )

    def setup(self):
        # 0. Load VAE (Optional)
        if self.config.use_vae:
            if is_main_process():
                print(f"Loading VAE: {self.config.vae_path}...")
            
            self.vae = AutoencoderKL.from_pretrained(self.config.vae_path).to(self.device)
            self.vae.eval()
            self.vae.requires_grad_(False)
            
            # Adjust in_channels for Latent Space
            self.config.in_channels = self.vae.config.latent_channels
            if is_main_process():
                print(f"VAE Loaded. Input channels adjusted to: {self.config.in_channels}")
        
        # 1. Initialize Conditioning Components
        if self.config.conditioning_type == "text":
            if is_main_process():
                print(f"Loading Text Encoder: {self.config.text_encoder_path}...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_encoder_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.text_encoder = AutoModelForCausalLM.from_pretrained(
                self.config.text_encoder_path,
                dtype=torch.bfloat16
            ).to(self.device)
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)

            if is_main_process():
                print("Compiling Text Encoder...")
            self.text_encoder = torch.compile(self.text_encoder)
            self.context_dim = self.config.context_dim_text

        elif self.config.conditioning_type == "class":
            if is_main_process():
                print(f"Initializing Class Embeddings for {self.config.num_classes} classes...")
            # +1 for a learnable null embedding for classifier-free guidance
            self.class_embedding = nn.Embedding(self.config.num_classes + 1, self.config.context_dim_class).to(self.device)
            self.null_class_embedding = nn.Parameter(torch.randn(1, 1, self.config.context_dim_class)).to(self.device)
            self.context_dim = self.config.context_dim_class

        else:
            raise ValueError(f"Unknown conditioning_type: {self.config.conditioning_type}")

        # 2. Initialize Model
        if is_main_process():
            print(f"Initializing {self.config.model_arch} Model...")

        if self.config.model_arch == "Styx":
            from models.styx import MinimalT2I
            ModelClass = MinimalT2I
        elif self.config.model_arch == "ZImage":
            from models.zimage import MinimalT2I
            ModelClass = MinimalT2I
        else:
             # Fallback or dynamic loading for other models
            try:
                module = importlib.import_module(f"models.{self.config.model_arch.lower()}")
                ModelClass = getattr(module, "MinimalT2I") 
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Unknown or invalid model architecture: {self.config.model_arch}") from e

        model = ModelClass(
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            num_heads=self.config.num_heads,
            context_dim=self.context_dim, # Use the dynamically set context_dim
            virtual_expansion=self.config.virtual_expansion,
        ).to(self.device).to(memory_format=torch.channels_last)

        # Wrap in DDP
        if self.world_size > 1:
            self.model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
            self.raw_model = model  # Access to original model for saving
        else:
            self.model = model
            self.raw_model = model

        # 3. Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        # Disable scaler for bf16, enable only for fp16
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.config.mixed_precision == 'fp16'))

        # 4. Flow Matching Helper
        self.flow_helper = xFlowMatching(self.model)
        
    def encode_data(self, pixels):
        """Encodes pixels to latents if VAE is enabled, otherwise returns pixels."""
        if self.vae is None:
            return pixels
        
        with torch.no_grad():
            # Input pixels are [-1, 1]. VAE expects that.
            # VAE usually runs in float32 or same as weights. 
            # We cast to VAE dtype.
            latents = self.vae.encode(pixels.to(dtype=self.vae.dtype)).latent_dist.sample()
            
            # Scale and Shift
            latents = latents * self.config.vae_scale_factor + self.config.vae_shift_factor
            
            # Cast back to model precision if needed (usually bf16 for training)
            # We'll let mixed precision context handle it in training loop, 
            # but explicit cast can be safer if VAE is fp32.
            return latents
            
    def decode_latents(self, latents):
        """Decodes latents to pixels if VAE is enabled."""
        if self.vae is None:
            # Un-normalize [-1, 1] -> [0, 1]
            return ((latents / 2) + 0.5).clamp(0, 1)
        
        # Inverse Scale and Shift
        latents = (latents - self.config.vae_shift_factor) / self.config.vae_scale_factor
        
        with torch.no_grad():
            images = self.vae.decode(latents.to(dtype=self.vae.dtype)).sample
            
        # VAE outputs are [-1, 1] usually. Un-normalize to [0, 1]
        return ((images / 2) + 0.5).clamp(0, 1)

    def _encode_prompt(self, prompt_batch, proportion_empty_prompts, is_train=True):
        # Batch processing of captions
        if isinstance(prompt_batch[0], (list, np.ndarray)):
            if is_train:
                captions = [random.choice(p) for p in prompt_batch]
            else:
                captions = [p[0] for p in prompt_batch]
        else:
            captions = list(prompt_batch)

        # Whole batch dropout
        if is_train and random.random() < proportion_empty_prompts:
            # Create zeros directly - skipping text encoder
            B = len(captions)
            D = self.context_dim # Use the dynamically set context_dim

            # Robust dtype fetching
            try:
                dtype = self.raw_model.patch_embed.weight.dtype
            except:
                dtype = torch.bfloat16

            # Return (B, 1, D) - L=1 is sufficient for zero-attention
            return torch.zeros(B, 1, D, device=self.target_device, dtype=dtype)

        text_inputs = self.tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(self.target_device)
        prompt_masks = text_inputs.attention_mask.to(self.target_device)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(self.config.mixed_precision != 'no'),
                                                 dtype=torch.bfloat16):
            prompt_embeds = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=prompt_masks,
                output_hidden_states=True,
            ).hidden_states

            # Stack hidden states
            prompt_embeds = torch.stack(prompt_embeds, dim=0)

            # Select 4 layers
            indices = torch.linspace(0, len(prompt_embeds) - 1, 5, dtype=torch.long)[1:]
            prompt_embeds = prompt_embeds[indices]

            # Permute and Reshape
            prompt_embeds = prompt_embeds.permute(1, 2, 0, 3).reshape(prompt_embeds.size(1), prompt_embeds.size(2), -1)

            # Cast to model dtype
            prompt_embeds = prompt_embeds.to(dtype=self.raw_model.patch_embed.weight.dtype)

        return prompt_embeds


    def _get_class_conditioning(self, class_labels, proportion_empty_prompts, is_train=True):
        if is_train and random.random() < proportion_empty_prompts:
            # Use null class embedding
            class_emb = self.null_class_embedding.repeat(len(class_labels), 1, 1)
        else:
            # +1 to class_labels for the null embedding index (assuming class_labels are 0 to num_classes-1)
            # The last embedding (num_classes) will be reserved for null_class_embedding or special tokens
            class_emb = self.class_embedding(class_labels).unsqueeze(1) # Unsqueeze for sequence length of 1
        return class_emb

    def get_conditioning(self, conditioning_input, proportion_empty_prompts=0.1, is_train=True):
        if self.config.conditioning_type == "text":
            return self._encode_prompt(conditioning_input, proportion_empty_prompts, is_train)
        elif self.config.conditioning_type == "class":
            return self._get_class_conditioning(conditioning_input, proportion_empty_prompts, is_train)
        else:
            raise ValueError(f"Unknown conditioning_type: {self.config.conditioning_type}")

    def sample_images(self, conditioning_input, step):
        """
        Evaluation/Sampling - Only run on Main Process
        """
        if not is_main_process():
            return

        self.model.eval()
        print(f"\nGenerating evaluation images at step {step}...")

        cfg_scale = self.config.cfg_scale

        with torch.no_grad():
            # Conditional Embeddings
            # Ensure conditioning_input is a tensor for class conditioning
            if self.config.conditioning_type == "class":
                conditioning_input_tensor = torch.tensor(conditioning_input, device=self.device, dtype=torch.long)
            else: # text conditioning_input is already a list of strings
                conditioning_input_tensor = conditioning_input 

            cond_emb = self.get_conditioning(conditioning_input_tensor, proportion_empty_prompts=0.0, is_train=False)

            # Unconditional Embeddings
            if self.config.conditioning_type == "text":
                B = len(conditioning_input)
                D = self.context_dim
                uncond_emb = torch.zeros(B, 1, D, device=self.target_device, dtype=cond_emb.dtype) # Assuming L=1 for uncond
            elif self.config.conditioning_type == "class":
                uncond_emb = self.null_class_embedding.repeat(len(conditioning_input), 1, 1)
            else:
                raise ValueError(f"Unknown conditioning_type: {self.config.conditioning_type}")

            # Latent Shape
            if self.vae:
                # Standard VAE reduces dim by 8
                H = W = self.config.image_size // 8
            else:
                H = W = self.config.image_size
                
            shape = (len(conditioning_input), self.config.in_channels, H, W)

            z = torch.randn(shape, device=self.device).to(memory_format=torch.channels_last)

            steps = self.config.eval_steps
            times = torch.linspace(0, 1, steps + 1, device=self.device)

            for i in range(steps):
                t_curr = times[i]
                t_next = times[i + 1]

                # Expand t for batch
                t_curr_expanded = t_curr.repeat(shape[0])

                # --- CFG (No Batching) ---
                # Predict Unconditional
                x_pred_uncond = self.model(z, t_curr_expanded, uncond_emb)

                # Predict Conditional
                x_pred_cond = self.model(z, t_curr_expanded, cond_emb)

                # Apply Guidance
                # pred = uncond + scale * (cond - uncond)
                x_pred = x_pred_uncond + cfg_scale * (x_pred_cond - x_pred_uncond)

                # Derive v_pred
                denom = 1 - t_curr
                if denom < 1e-5: denom = 1e-5
                v_pred = (x_pred - z) / denom

                dt = t_next - t_curr
                z = z + dt * v_pred

            # Decode latents/pixels
            images = self.decode_latents(z).cpu()

            wandb_images = []
            for idx, img_tensor in enumerate(images):
                pil_img = transforms.ToPILImage()(img_tensor)
                caption_text = f"Class: {conditioning_input[idx]}" if self.config.conditioning_type == "class" else conditioning_input[idx][:100]
                wandb_images.append(wandb.Image(pil_img, caption=caption_text))

            wandb.log({"eval/images": wandb_images, "global_step": step})

        self.model.train()

    def train(self):
        # DDP: Each process gets its own slice via split_by_node internally
        if self.config.dataset_name == "webdataset":
            dataloader = get_wds_loader(
                url_pattern=self.config.data_path,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                is_train=True
            )
        elif self.config.dataset_name == "food101":
            dataloader = get_food101_loader( # Already imported, no need for `from data import`
                config=self.config,
                is_train=True
            )
        else:
            raise ValueError(f"Unknown dataset_name: {self.config.dataset_name}")

        if is_main_process():
            print("Starting Training Loop...")

        # Scheduler
        num_warmup_steps = int(0.01 * self.config.max_steps)
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.config.max_steps
        )

        # Resume Logic
        start_step = 0
        ema_loss = None
        if self.config.resume_from:
            if is_main_process():
                print(f"Resuming from {self.config.resume_from}...")
            ckpt = torch.load(self.config.resume_from, map_location=self.device)
            self.raw_model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_step = ckpt['step'] + 1
            if 'ema_loss' in ckpt:
                ema_loss = ckpt['ema_loss']

        self.model.train()
        step = start_step
        data_iter = iter(dataloader)

        # Tqdm only on main process
        if is_main_process():
            progress_bar = tqdm(total=self.config.max_steps)
        else:
            progress_bar = None

        # Loss averaging
        running_loss = 0.0
        running_steps = 0

        while step < self.config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            except Exception as e:
                if is_main_process():
                    print(f"Data loading error: {e}")
                continue

            # Handle batch differently based on dataset and conditioning type
            pixels = None
            conditioning_input = None

            if self.config.dataset_name == "webdataset":
                pixels = batch["pixels"].to(self.device, memory_format=torch.channels_last)
                if self.config.conditioning_type == "text":
                    conditioning_input = batch["prompts"]
                elif self.config.conditioning_type == "class":
                    # If webdataset is used with class conditioning, we need class labels.
                    # This case is currently not directly supported by webdataset loader for class_labels.
                    # For now, generate random class labels for demonstration/testing purposes.
                    # A proper solution would involve modifying webdataset to yield class labels.
                    print("Warning: Webdataset used with class conditioning. Generating random class labels.")
                    conditioning_input = torch.randint(0, self.config.num_classes, (pixels.shape[0],), device=self.device)
                else:
                    raise ValueError(f"Unknown conditioning_type for webdataset: {self.config.conditioning_type}")
                    
            elif self.config.dataset_name == "food101":
                pixels = batch[0].to(self.device, memory_format=torch.channels_last)
                if self.config.conditioning_type == "class":
                    conditioning_input = batch[1].to(self.device)
                elif self.config.conditioning_type == "text":
                    # If food101 is used with text conditioning, we need prompts.
                    # This case is not currently supported by food101 loader for text prompts.
                    # For now, generate dummy prompts for demonstration/testing purposes.
                    print("Warning: Food101 used with text conditioning. Generating dummy text prompts.")
                    conditioning_input = ["dummy prompt"] * pixels.shape[0]
                else:
                    raise ValueError(f"Unknown conditioning_type for food101: {self.config.conditioning_type}")
            else:
                raise ValueError(f"Unknown dataset_name: {self.config.dataset_name}")
            
            # Encode to Latents (if VAE)
            latents = self.encode_data(pixels)
            latents = latents.to(memory_format=torch.channels_last)

            # Get Conditioning
            cond_emb = self.get_conditioning(
                conditioning_input,
                proportion_empty_prompts=self.config.proportion_empty_prompts,
                is_train=True
            )

            # Mixed Precision
            with torch.amp.autocast('cuda', enabled=(self.config.mixed_precision != 'no'), dtype=torch.bfloat16):
                # Flow Matching works on 'latents' now
                t = self.flow_helper.sample_t(latents.shape[0], self.device)
                e = torch.randn_like(latents)

                t_reshaped = self.flow_helper.reshape_t(t, latents)
                z = t_reshaped * latents + (1 - t_reshaped) * e
                # target_v = latents - e # Not used in loss calc currently, x_pred is predicted
                
                z = z.to(memory_format=torch.channels_last)

                x_pred = self.model(z, t, cond_emb)
                snr_weights = 1.0 / (1.0 - t + 1e-4) ** 2
                snr_weights = torch.clamp(snr_weights, max=5.0)
                if hasattr(self.flow_helper, 'reshape_t'):
                    snr_weights = self.flow_helper.reshape_t(snr_weights, latents)
                else:
                    snr_weights = snr_weights.view(-1, 1, 1, 1)
                
                loss_elementwise = F.mse_loss(x_pred, latents, reduction='none')
                loss = (loss_elementwise * snr_weights).mean()

                loss = loss / self.config.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.config.grad_accum_steps == 0:
                # Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                scheduler.step()

                # EMA Loss
                current_loss = loss.item() * self.config.grad_accum_steps
                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = ema_loss * 0.99 + current_loss * 0.01

                # Accumulate for logging
                running_loss += current_loss
                running_steps += 1

                # Logging (Main process only)
                if is_main_process():
                    if step % self.config.log_every == 0:
                        avg_loss = running_loss / running_steps if running_steps > 0 else 0.0

                        wandb.log({
                            "train/loss": avg_loss,
                            "train/ema_loss": ema_loss,
                            "train/lr": self.optimizer.param_groups[0]['lr']
                        }, step=step)
                        progress_bar.set_description(f"Loss: {avg_loss:.4f} | EMA: {ema_loss:.4f}")

                        # Reset running stats
                        running_loss = 0.0
                        running_steps = 0

                    progress_bar.update(1)

                    # Evaluation
                    if step > 0 and step % self.config.eval_every == 0:
                        # For eval, sample num_eval_images from 0 to num_classes-1
                        eval_conditioning_input = list(range(min(self.config.num_eval_images, self.config.num_classes)))
                        # If more eval images requested than classes, repeat some classes
                        while len(eval_conditioning_input) < self.config.num_eval_images:
                            eval_conditioning_input.append(random.randint(0, self.config.num_classes - 1))
                        self.sample_images(eval_conditioning_input, step)
                        self.calculate_fid(step) # Call FID calculation here

                    # Saving (Use raw_model to avoid 'module.' prefix)
                    if step > 0 and step % self.config.save_every == 0:
                        save_path = os.path.join(self.config.output_dir, f"checkpoint_{step}.pt")
                        torch.save({
                            'step': step,
                            'model_state_dict': self.raw_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'config': self.config.__dict__,
                            'ema_loss': ema_loss
                        }, save_path)
                        print(f"Saved checkpoint: {save_path}")

                        # Top K Logic (Lowest EMA Loss)
                        self.saved_checkpoints.append((ema_loss, save_path))
                        self.saved_checkpoints.sort(key=lambda x: x[0])  # Ascending sort

                        while len(self.saved_checkpoints) > self.config.save_top_k:
                            to_remove = self.saved_checkpoints.pop(-1)  # Remove highest loss
                            try:
                                if os.path.exists(to_remove[1]):
                                    os.remove(to_remove[1])
                                    print(f"Removed old checkpoint: {to_remove[1]}")
                            except OSError as e:
                                print(f"Error removing checkpoint: {e}")

                step += 1

    def calculate_fid(self, step):
        if not is_main_process():
            return
        
        self.model.eval()
        print(f"\nCalculating FID at step {step}...")

        # Create a temporary directory for generated images
        gen_dir = os.path.join(self.config.output_dir, f"fid_gen_step_{step}")
        os.makedirs(gen_dir, exist_ok=True)

        num_generated_images = 0
        total_fid_samples = self.config.fid_num_samples
        fid_batch_size = self.config.fid_batch_size

        with torch.no_grad():
            while num_generated_images < total_fid_samples:
                batch_size = min(fid_batch_size, total_fid_samples - num_generated_images)
                
                # Generate random class labels for the batch
                # Assuming class conditioning is active for FID
                if self.config.conditioning_type == "class":
                    # Generate a diverse set of class labels for FID
                    # This ensures coverage across different classes
                    if self.config.num_classes > 0:
                        class_indices = list(range(self.config.num_classes))
                        random.shuffle(class_indices)
                        current_class_labels = class_indices[:batch_size]
                        # If batch_size > num_classes, repeat classes
                        while len(current_class_labels) < batch_size:
                            current_class_labels.append(random.choice(class_indices))
                    else:
                        raise ValueError("num_classes must be greater than 0 for class-conditional FID.")
                    
                    random_class_labels = torch.tensor(current_class_labels, device=self.device, dtype=torch.long)
                else:
                    # If text conditioning, we would need to generate diverse prompts.
                    # This is more complex and depends on the project's text generation capabilities.
                    # For now, raise an error or use dummy prompts if text conditioning is attempted with FID.
                    raise NotImplementedError("FID calculation with text conditioning requires generating diverse prompts, which is not implemented yet.")

                # Get conditional embeddings
                cond_emb = self.get_conditioning(random_class_labels, proportion_empty_prompts=0.0, is_train=False)
                
                # Latent Shape
                if self.vae:
                    H = W = self.config.image_size // 8
                else:
                    H = W = self.config.image_size
                    
                shape = (batch_size, self.config.in_channels, H, W)

                z = torch.randn(shape, device=self.device).to(memory_format=torch.channels_last)

                # Simplified sampling for FID (no CFG, fewer steps)
                # This could be made configurable
                steps = 10 # Fewer steps for faster generation, potentially lower FID but faster eval
                times = torch.linspace(0, 1, steps + 1, device=self.device)

                for i in range(steps):
                    t_curr = times[i]
                    t_next = times[i + 1]

                    t_curr_expanded = t_curr.repeat(shape[0])

                    # No CFG for FID generation for simplicity, use conditional only
                    x_pred = self.model(z, t_curr_expanded, cond_emb)

                    denom = 1 - t_curr
                    if denom < 1e-5: denom = 1e-5
                    v_pred = (x_pred - z) / denom

                    dt = t_next - t_curr
                    z = z + dt * v_pred

                images = self.decode_latents(z).cpu()
                
                # Save generated images
                for img_idx, img_tensor in enumerate(images):
                    pil_img = transforms.ToPILImage()(img_tensor)
                    pil_img.save(os.path.join(gen_dir, f"img_{num_generated_images + img_idx:05d}.png"))
                
                num_generated_images += batch_size
                print(f"Generated {num_generated_images}/{total_fid_samples} images for FID.")
            
            # Compute FID
            # torch-fidelity expects a path to a directory of images or pre-calculated stats
            # For reference, use the pre-calculated stats provided in config
            try:
                from torch_fidelity import calculate_metrics
                metrics = calculate_metrics(
                    input1=gen_dir,
                    input2=self.config.fid_reference_stats_path,
                    cuda=True,
                    batch_size=fid_batch_size,
                    fid=True,
                    verbose=False,
                    samples_shuffle=True, # Shuffle samples to avoid order bias
                    datasets_root=self.config.food101_dataset_path # Pass the dataset path for internal FID calculation if needed
                )
                fid_score = metrics['fidelity/fid_full']
                print(f"FID Score at step {step}: {fid_score:.2f}")
                wandb.log({"eval/fid": fid_score, "global_step": step})
            except Exception as e:
                print(f"Error calculating FID: {e}")
                fid_score = -1 # Indicate failure

        # Clean up generated images
        import shutil
        try:
            shutil.rmtree(gen_dir)
            print(f"Cleaned up temporary directory: {gen_dir}")
        except OSError as e:
            print(f"Error removing temporary directory {gen_dir}: {e}")

        self.model.train()
        return fid_score


if __name__ == "__main__":
    rank, local_rank, world_size = setup_ddp()
    config = Config()

    # Adjust learning rate for DDP (optional but recommended: scale with world size)
    # config.learning_rate *= world_size

    trainer = Trainer(config, rank=rank, local_rank=local_rank, world_size=world_size)
    trainer.setup()
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        cleanup_ddp()