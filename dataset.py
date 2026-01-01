import webdataset as wds
import torch
from torch.utils.data import DataLoader
import pandas as pd
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import random
import io
from PIL import Image
import json

def warn_and_continue(exn):
    print(f"Warning: {exn}")
    return True

class WDSLoader:
    def __init__(self, url, csv_path, image_size=64, batch_size=16, num_workers=4):
        self.url = url
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_map = self.load_class_map(csv_path)
        self.num_classes = len(self.class_map)

        # Transforms parameters
        self.scale = (0.5, 1.0)
        self.ratio = (3. / 4., 4. / 3.)

    def load_class_map(self, csv_path):
        # Using latin-1 encoding to avoid UnicodeDecodeError on special characters
        df = pd.read_csv(csv_path, encoding='latin-1')
        return dict(zip(df['character'], df['id']))

    def preprocess(self, sample):
        # Find image key
        image_key = None
        for key in ["image", "jpg", "jpeg", "png", "webp"]:
            if key in sample:
                image_key = key
                break
        
        if image_key is None:
            # Debug: Print keys if image not found (first few times)
            if not hasattr(self, "_log_missing_key_count"): self._log_missing_key_count = 0
            if self._log_missing_key_count < 5:
                print(f"Skipping sample: No image key found. Available keys: {list(sample.keys())}")
                self._log_missing_key_count += 1
            return None

        # Decode image
        try:
            image_bytes = sample[image_key]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None # Skip broken images

        # Decode JSON for class
        # If 'json' key missing, treat as unknown/unconditioned
        try:
            if "json" in sample:
                meta = json.loads(sample["json"])
            else:
                meta = {} # Empty meta implies unknown character
            
            char_name = meta.get("character", "unknown")
            # Map to self.num_classes if not found. This aligns with the DiT null token index,
            # effectively treating unknown characters as "unconditioned" samples.
            class_id = self.class_map.get(char_name, self.num_classes)
        except Exception as e:
            print(f"Error parsing metadata: {e}")
            return None

        # Random Resized Crop logic
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)
        
        # Original size
        W, H = image.size
        
        # Relative coords: top, left, height, width
        # i (top), j (left), h, w
        # Normalize by H, W
        rel_coords = [i / H, j / W, h / H, w / W]
        rel_coords = torch.tensor(rel_coords, dtype=torch.float32)

        # Apply crop and resize
        image = F.resized_crop(image, i, j, h, w, size=(self.image_size, self.image_size))
        
        # To Tensor and Normalize [-1, 1]
        image = F.to_tensor(image)
        image = (image - 0.5) * 2.0 

        return {
            "image": image,
            "class_id": class_id,
            "coords": rel_coords
        }

    def make_loader(self):
        dataset = (
            wds.WebDataset(self.url, nodesplitter=wds.split_by_node, handler=warn_and_continue,)
            .shuffle(1000)
            .map(self.preprocess, handler=warn_and_continue,)
            .select(lambda x: x is not None)
            .to_tuple("image", "class_id", "coords", handler=warn_and_continue,)
            .batched(self.batch_size, partial=False)
        )
        
        loader = DataLoader(
            dataset,
            batch_size=None, # Batched in webdataset
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader
