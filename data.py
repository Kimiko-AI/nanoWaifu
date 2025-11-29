import torch
import webdataset as wds
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
from PIL import Image
import random
import numpy as np
from functools import partial

# --- Configuration ---
# TODO: Make these configurable via arguments if needed
BUCKET_SIZES = [
    (256, 256),
    (224, 288),
    (288, 224),
    (320, 192),
    (192, 320),
]
# Pre-calculate ratios for faster lookups
BUCKET_RATIOS = np.array([w / h for w, h in BUCKET_SIZES])


def assign_bucket_index(width, height):
    ratio = width / height
    # Find index of closest ratio
    idx = (np.abs(BUCKET_RATIOS - ratio)).argmin()
    return idx


def resize_to_bucket(image, bucket_idx):
    target_w, target_h = BUCKET_SIZES[bucket_idx]
    return image.resize((target_w, target_h), Image.BICUBIC)


# --- 1. Preprocessing Function (for WebDataset) ---
def transform_sample(sample):
    """
    Extracts data from the raw WebDataset dictionary and prepares it.
    """
    # WebDataset decodes "json" into a dict and "webp" into a PIL Image automatically
    json_data = sample["json"]
    image = sample["webp"]

    # Tag processing
    rating = json_data.get("rating", [])
    character_tags = json_data.get("character_tags", [])
    general_tags = json_data.get("general_tags", [])
    all_tags = rating + character_tags + general_tags
    tag_str = " ".join(map(str, all_tags))[:512]

    return {
        "image": image,
        "prompts": tag_str,
        "key": sample["__key__"]
    }


# --- 2. The Bucket Batcher (The Core Logic for WebDataset) ---
def bucket_batcher(data_stream, batch_size=16):
    """
    A generator that consumes the stream, groups by bucket,
    and yields batches when a bucket is full.
    """
    # Initialize empty lists for each bucket
    buckets = [[] for _ in BUCKET_SIZES]
    # Normalize to [-1, 1]
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    for sample in data_stream:
        try:
            image = sample["image"]
            w, h = image.size

            # 1. Determine Bucket
            b_idx = assign_bucket_index(w, h)

            # 2. Resize & Transform
            image_resized = resize_to_bucket(image, b_idx)
            image_tensor = to_tensor(image_resized)

            # 3. Add to specific bucket buffer
            buckets[b_idx].append({
                "pixels": image_tensor,
                "prompts": sample["prompts"]
            })

            # 4. Yield if full
            if len(buckets[b_idx]) >= batch_size:
                batch = buckets[b_idx]

                # Collate manually
                yield {
                    "pixels": torch.stack([x["pixels"] for x in batch]),
                    "prompts": [x["prompts"] for x in batch]
                }

                # Reset this specific bucket
                buckets[b_idx] = []

        except Exception as e:
            # print(f"Error processing sample {sample.get('key', 'unknown')}: {e}")
            continue


# --- 3. The WebDataset Pipeline Builder ---
def get_wds_loader(url_pattern, batch_size, num_workers=4, is_train=True):
    # A. Basic Pipeline
    # handler=wds.warn_and_continue catches errors during decoding/reading and skips them
    # nodesplitter=wds.split_by_node ensures each GPU (node) gets a subset of shards (Crucial for DDP)
    dataset = wds.WebDataset(
        url_pattern, 
        resampled=True, 
        handler=wds.warn_and_continue,
        nodesplitter=wds.split_by_node,
        shardshuffle=True
    )

    # B. Sharding for Workers
    if is_train:
        dataset = dataset.shuffle(1000)
    
    # Split among workers on the same node
    dataset = dataset.compose(wds.split_by_worker)

    # C. Decoding and Mapping
    dataset = (
        dataset
        .decode("pil", handler=wds.warn_and_continue)  # handle decoding errors
        .map(transform_sample, handler=wds.warn_and_continue)  # handle transform errors
    )

    # D. Apply Bucket Batching
    dataset = dataset.compose(partial(bucket_batcher, batch_size=batch_size))

    # E. DataLoader
    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    return loader


# --- 4. Food-101 Dataset Loader ---
def get_food101_loader(config, is_train=True):
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.Food101(
        root=config.food101_dataset_path,
        split='train' if is_train else 'test',
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_train,
        num_workers=config.num_workers,
        pin_memory=True
    )
    return loader