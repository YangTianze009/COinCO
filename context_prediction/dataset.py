import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Dataset class for loading images and objects data
class CustomDataset(Dataset):
    def __init__(self, image_paths1, image_paths2, labels, objects_df, transform=None):
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2
        self.labels = labels
        self.transform = transform
        
        # Extract coco_index from the image paths
        self.coco_indices = [int(os.path.basename(path).split('.')[0]) for path in image_paths1]
        
        # Create a mapping from coco_index to objects and replacement
        self.objects_data = {}
        for _, row in objects_df.iterrows():
            coco_idx = row['coco_index']
            objects = row['objects'].split(',') if isinstance(row['objects'], str) else []
            replacement = row['replacement_object']
            self.objects_data[coco_idx] = {'objects': objects, 'replacement': replacement}
    
    def __len__(self):
        return len(self.image_paths1)
    
    def __getitem__(self, idx):
        image_path1 = self.image_paths1[idx]
        image_path2 = self.image_paths2[idx]
        label = self.labels[idx]
        coco_idx = self.coco_indices[idx]
        
        try:
            image1 = Image.open(image_path1).convert("RGB")
            image2 = Image.open(image_path2).convert("RGB")
        except (OSError, IOError) as e:
            print(f"Skipping image {image_path1} or {image_path2} due to error: {e}")
            # Skip to the next valid data point
            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        # Get objects and replacement information for this coco_index
        objects_data = self.objects_data.get(coco_idx, {'objects': [], 'replacement': ''})
        objects_list = objects_data['objects']
        replacement_object = objects_data['replacement']
        objects_list.remove(replacement_object)
        return image1, image2, label, objects_list, replacement_object, coco_idx

# Image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
])