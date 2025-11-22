import torch

class Config:
    def __init__(self):
        # Model Configuration
        self.model_name = "MinimalT2I-Pixel"
        self.text_encoder_path = "HuggingFaceTB/SmolLM-135M"
        self.hidden_size = 768
        self.depth = 12
        self.num_heads = 12
        self.patch_size = 2
        self.in_channels = 3  # Pixel space (RGB)
        
        # Data Configuration
        self.data_path = "C:/Users/nRuaif/ChatError/Dan_dataset/shards/{00000..00010}.tar"
        self.batch_size = 4
        self.num_workers = 4
        self.image_size = 256 # Base size for bucketing reference
        
        # Training Configuration
        self.learning_rate = 1e-4
        self.max_steps = 10000
        self.grad_accum_steps = 4
        self.mixed_precision = "bf16" # 'bf16', 'fp16', or 'no'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proportion_empty_prompts = 0.1 # Classifier-free guidance probability
        
        # Logging & Saving
        self.wandb_project = "nanoWaifu-T2I"
        self.wandb_run_name = "pixel-space-run-01"
        self.log_every = 10
        self.save_every = 1000
        self.eval_every = 500
        self.output_dir = "./outputs"
        self.save_top_k = 3
        self.resume_from = None # Path to checkpoint .pt file to resume from
        
        # Generation/Eval
        self.num_eval_images = 4
        self.eval_steps = 50 # Diffusion steps for sampling
