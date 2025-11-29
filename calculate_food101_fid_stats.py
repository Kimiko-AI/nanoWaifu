import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics
from config import Config # Assuming config.py is in the same directory

def generate_food101_fid_stats():
    config = Config()

    # Ensure the directory for FID stats exists
    os.makedirs(os.path.dirname(config.fid_reference_stats_path), exist_ok=True)

    # Transformations for the real images (must match the generated images)
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalize to [-1, 1]
    ])

    # Load Food-101 test dataset
    print(f"Loading Food-101 test dataset from {config.food101_dataset_path}...")
    dataset = datasets.Food101(
        root=config.food101_dataset_path,
        split='test',
        download=True, # Download if not available
        transform=transform
    )
    
    # DataLoader for the dataset
    dataloader = DataLoader(
        dataset,
        batch_size=config.fid_batch_size,
        shuffle=False, # Order doesn't matter for stats calculation
        num_workers=config.num_workers,
        pin_memory=True
    )

    print(f"Calculating FID statistics for Food-101 test set (input2) with {len(dataset)} images...")
    
    metrics = calculate_metrics(
        input1=dataloader, # Pass the DataLoader for the real images
        input2=None, # We want to calculate and save stats for input1
        cuda=True,
        batch_size=config.fid_batch_size,
        fid=True,
        save_cpu_stats=True, # Save CPU stats for reproducibility
        save_path=config.fid_reference_stats_path, # Path to save the stats
        samples_shuffle=True, # Shuffle samples for stats calculation
        verbose=True,
    )
    
    print(f"Food-101 FID statistics saved to: {config.fid_reference_stats_path}")
    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    generate_food101_fid_stats()