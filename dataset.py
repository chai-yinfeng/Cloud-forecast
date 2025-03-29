import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from tqdm import tqdm

class SequenceDataset(Dataset):
    def __init__(self, image_folder, input_frames, target_frames, resize_shape=(128, 128)):
        self.input_frames = input_frames
        self.target_frames = target_frames
        self.resize_shape = resize_shape

        self.images = self.load_and_preprocess_images(image_folder)
        self.data, self.targets = self.create_dataset(self.images)

    def load_and_preprocess_images(self, folder):
        images = []
        file_list = sorted(os.listdir(folder))
        for filename in tqdm(file_list, desc="Loading images"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, self.resize_shape, interpolation=cv2.INTER_AREA)
                img = np.array(img, dtype=np.float32) / 255.0
                images.append(img)
        return images

    def create_dataset(self, images):
        data, targets = [], []
        total_length = self.input_frames + self.target_frames
        for i in tqdm(range(len(images) - total_length + 1), desc="Building sequences"):
            seq_input = images[i:i + self.input_frames]
            seq_target = images[i + self.input_frames:i + total_length]

            data.append(seq_input)
            targets.append(seq_target)

        data = np.expand_dims(np.array(data), axis=2)  # (batch_size, input_frames, channel, height, width)
        targets = np.expand_dims(np.array(targets), axis=2)  # (batch_size, target_frames, channel, height, width)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
