import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(
        self,
        directory,
        label_mode='int', # 'int', 'binary', 'categorical', or None
        color_mode='rgb', # 'rgb' or 'grayscale'
        image_size=(32, 32),
        interpolation='bilinear' # 'nearest' or 'bilinear'
    ):
        # Check valid
        if label_mode not in ('int', 'binary', 'categorical', None):
            raise ValueError("label_mode must be one of 'int', 'binary', 'categorical', or None.")
        if color_mode not in ('rgb', 'grayscale'):
            raise ValueError("color_mode must be 'rgb' or 'grayscale'.")
        if interpolation not in ('nearest', 'bilinear'):
            raise ValueError("interpolation must be 'nearest' or 'bilinear'.")

        self.directory = directory
        self.label_mode = label_mode
        self.color_mode = color_mode
        self.image_size = image_size
        self.interpolation = cv2.INTER_LINEAR if interpolation == 'bilinear' else cv2.INTER_NEAREST

        # Determine class names
        self.class_names = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )
        self.class_to_index = {name: idx for idx, name in enumerate(self.class_names)}
        # Take samples
        self.samples = []  
        for root, _, files in os.walk(directory):
            if root == directory:
                continue
            for fname in files:
                if fname.lower().endswith(('png','jpg')):
                    path = os.path.join(root, fname)
                    label = self.class_to_index[os.path.basename(root)]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)

        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        
        if self.color_mode == 'grayscale':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:  
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = cv2.resize(img, self.image_size, interpolation=self.interpolation)
        # img = img.astype(np.float32) / 255.0

        if self.color_mode == 'grayscale':
            img = np.expand_dims(img, -1) #(height, width, channel)
        img = np.transpose(img, (2, 0, 1)) #(channel, height, width)
        tensor = torch.from_numpy(img)

        if self.label_mode == 'int':
            return tensor, label
        if self.label_mode == 'binary':
            return tensor, torch.tensor(label, dtype=torch.float32)
        if self.label_mode == 'categorical':
            one_hot = torch.zeros(len(self.class_names), dtype=torch.float32)
            one_hot[label] = 1.0
            return tensor, one_hot