from __future__ import print_function, division

import glob
import os

import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class MPIIGaze(Dataset):
    def __init__(self, mpii_directory: str = 'datasets/MPIIGaze'):
        self.mpii_directory = mpii_directory
        evaluation_files = glob.glob(f'{mpii_directory}/Evaluation Subset/sample list for eye image/*.txt')

        self.eval_entries = []
        for eval_file in evaluation_files:
            person = os.path.splitext(os.path.basename(eval_file))[0]
            with open(eval_file) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line != '':
                        image_path, side = [x.strip() for x in line.split()]
                        day, image = image_path.split('/')
                        self.eval_entries.append({'day':day, 'image_name':image,
                                                 'person': person, 'side': side})

    def __len__(self):
        return len(self.eval_entries)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return self._load_sample(index)

    def _load_sample(self, i):
        entry = self.eval_entries[i]
        mat_path = os.path.join(self.mpii_directory, 'Data/Normalized', entry['person'], entry['day'] + '.mat')
        mat = sio.loadmat(mat_path)

        filenames = mat['filenames']
        row = np.argwhere(filenames == entry['image_name'])[0][0]
        side = entry['side']

        image = mat['data'][side][0, 0]['image'][0, 0][row]
        image = cv2.resize(image, (160, 96))
        image = cv2.equalizeHist(image)
        image = image / 255.
        image = image.astype(np.float32)
        if side == 'right':
            image = np.fliplr(image)

        (x, y, z) = mat['data'][side][0, 0]['gaze'][0, 0][row]
        theta = np.arcsin(-y)
        phi = np.arctan2(-x, -z)
        gaze = np.array([-theta, phi])

        return {
            'img': image,
            'gaze': gaze,
            'side': side
        }
