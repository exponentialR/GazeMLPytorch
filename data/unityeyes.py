from __future__ import print_function, division
import os
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import json
import glob
import cv2
from utilities.preprocess import preprocess_eye_image


class UnityEyesDataset(Dataset):
    def __init__(self, image_directory: Optional[str] = None):
        if image_directory is None:
            image_directory = os.path.join(os.path.dirname(__file__), 'UnityEyes/imgs')
        self.image_paths = glob.glob(os.path.join(image_directory, '*.jpg'))
        self.image_paths = sorted(self.image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.json_paths = []
        for image_path in self.image_paths:
            index = os.path.splitext(os.path.basename(image_path))[0]
            self.json_paths.append(os.path.join(image_directory, f'{index}.json'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        full_image = cv2.imread(self.image_paths[index])
        with open(self.json_paths[index]) as f:
            json_data = json.load(f)

        eye_sample = preprocess_eye_image(full_image, json_data)
        sample = {'full_image': full_image, 'json_data': json_data}
        sample.update(eye_sample)
        return sample
