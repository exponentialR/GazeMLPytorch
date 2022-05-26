import torch
import cv2
import argparse
import numpy as np

from data.unityeyes import UnityEyesDataset
from torch.utils.data import DataLoader
from models.eyenet import EyeNet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Seting up pytorch

torch.backends.cudnn.enabled = False
torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device', device)

parser = argparse.ArgumentParser(description='Trains an EyeNet model')
parser.add_argument('--num_stack', type=int, default=3, help='Number of hourglass layers.')
parser.add_argument('--num_features', type=int, default=32, help='Number of feature maps to use.')
parser.add_argument('--num_landmarks', type=int, default=34, help='Number of landmarks to be predicted.')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='Number of epochs to iterate over all training examples.')
parser.add_argument('--start_from',
                    help='A model checkpoint file to begin training from. This overrides all other arguments.')
parser.add_argument('--output', default='checkpoint.pt', help='The output checkpoint filename')
args = parser.parse_args()


def validate(eyenet: EyeNet, val_loader: DataLoader) -> float:
    with torch.no_grad():
        val_losses = []
        for val_batch in val_loader:
            val_images = val_batch['image'].float().to(device)
            heatmaps = val_batch['heatmaps'].to(device)
            landmarks = val_batch['landmarks'].to(device)
            gaze = val_batch['gaze'].float().to(device)
            heatmaps_prediction, landmarks_prediction, gaze_prediction = eyenet.forward(val_images)
            heatmaps_loss, landmarks_loss, gaze_loss = eyenet.calculate_loss(
                heatmaps_prediction, heatmaps, landmarks_prediction, landmarks, gaze_prediction, gaze
            )
            loss = 1000 * heatmaps_loss + landmarks_loss + gaze_loss
            val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        return val_loss


def train_epoch(epoch: int, eyenet: EyeNet, optimizer, train_loader: DataLoader,
                val_loader: DataLoader, best_val_loss: float,
                checkpoint_fn: str, writer: SummaryWriter):
    print(len(train_loader))

train_epoch(train_loader=DataLoader)
