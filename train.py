import torch
import cv2
import argparse
import numpy as np
import os
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
    N = len(train_loader)
    for i_batch, sample_batched in enumerate(train_loader):
        i_batch += N * epoch
        images = sample_batched['image'].float().to(device)
        heatmaps_prediction, landmarks_prediction, gaze_prediction = eyenet.forward(images)
        heatmaps = sample_batched['heatmaps'].to(device)
        landmarks = sample_batched['landmarks'].float().to(device)
        gaze = sample_batched['gaze'].float().to(device)
        heatmaps_loss, landmarks_loss, gaze_loss = eyenet.calculate_loss(
            heatmaps_prediction, heatmaps, landmarks_prediction, landmarks, gaze_prediction, gaze
        )
        loss = 1000 * heatmaps_loss + landmarks_loss + gaze_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        hm = np.mean(heatmaps[-1, 8:16].cpu().detach().numpy(), axis=0)
        hm_pred = np.mean(heatmaps_prediction[-1, -1, 8:16].cpu().detach().numpy(), axis=0)
        norm_hm = cv2.normalize(hm, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_hm_pred = cv2.normalize(hm_pred, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if i_batch % 20 == 0:
            cv2.imwrite('true.jpg', norm_hm * 255)
            cv2.imwrite('pred.jpg', norm_hm_pred * 255)
            cv2.imwrite('eye.jpg', sample_batched['img'].numpy()[-1] * 255)

        writer.add_scalar('Training heatmap loss', heatmaps_loss.item(), i_batch)
        writer.add_scalar('Training landmarks loss', landmarks_loss.item(), i_batch)
        writer.add_scalar('Training gaze loss', gaze_loss.item(), i_batch)
        writer.add_scalar('Training loss', loss.item(), i_batch)

        if i_batch > 0 and i_batch % 20 == 0:
            val_loss = validate(eyenet=eyenet, val_loader=val_loader)
            writer.add_scalar('validation loss', val_loss, i_batch)
            print('Epoch', epoch, 'Validation loss', val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'num_stack': eyenet.num_stack,
                    'num_features': eyenet.num_features,
                    'num_landmarks': eyenet.num_landmarks,
                    'best_val_loss': best_val_loss,
                    'model_state_dict': eyenet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_fn)

def train(eyenet:EyeNet, optimizer, num_epochs: int, best_val_loss:float, checkpoint_fn:str):
    timestr = datetime.now().strftime("%m%d%Y-%H%M%S")
    writer = SummaryWriter(f'runs/eyenet-{timestr}')
    dataset = UnityEyesDataset()
    N = len(dataset)
    VN = 160
    TN = N - VN
    train_set, val_set = torch.utils.data.random_split(dataset, (TN, VN))

    train_loader = DataLoader(train_set, batch_size=16, shuffle = True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=True)

    for i in range(num_epochs):
        best_val_loss = train_epoch(epoch=i,eyenet=eyenet,
                    optimizer=optimizer,train_loader=train_loader,val_loader=val_loader,
                    best_val_loss=best_val_loss,checkpoint_fn=checkpoint_fn,
                    writer=writer)

def main():
    learning_rate = 4 * 1e-4

    if args.start_from:
        start_from = torch.load(args.start_from, map_loaction=device)
        num_stack = start_from['num_stack']
        num_features = start_from['num_features']
        num_landmarks = start_from['num_landmarks']
        best_val_loss = start_from['best_val_loss']
        eyenet = EyeNet(num_stack=num_stack, num_features=num_features, num_landmarks=num_landmarks).to(device)
        optimizer = torch.optim.Adam(eyenet.parameters(), lr=learning_rate)
        eyenet.load_state_dict(start_from['model_state_dict'])
        optimizer.load_state_dict(start_from['optimizer_state_dict'])
    elif os.path.exists(args.output):
        raise Exception(f'Out file {args.output} already exists')
    else:
        num_stack = args.num_stack
        num_features = args.num_features
        num_landmarks = args.num_landmarks
        best_val_loss = float('inf')
        eyenet = EyeNet(num_stack=num_stack, num_features=num_features, num_landmarks=num_landmarks).to(device)
        optimizer = torch.optim.Adam(eyenet.parameters(), lr=learning_rate)

    train(
        eyenet=eyenet,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        best_val_loss=best_val_loss,
        checkpoint_fn=args.output
    )


if __name__ == '__main__':
    main()
