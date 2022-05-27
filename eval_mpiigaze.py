import numpy as np
import torch

import utilities.gaze
from data.MPIIGaze import MPIIGaze
from models.eyenet import EyeNet

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = MPIIGaze()
checkpoint = torch.load('checkpoint.pt', map_location=dev)
num_stack = checkpoint['num_stack']
num_features = checkpoint['num_features']
num_landmarks = checkpoint['num_landmarks']
eyenet = EyeNet(num_stack=num_stack, num_features=num_features, num_landmarks=num_landmarks).to(dev)
eyenet.load_state_dict(checkpoint['model_state_dict'])

with torch.no_grad():
    errors = []

    print('N', len(dataset))
    for i, sample in enumerate(dataset):
        print(i)
        x = torch.tensor([sample['image']]).float().to(dev)
        heatmaps_prediction, landmarks_prediction, gaze_prediction = eyenet.forward(x)

        gaze = sample['gaze'].reshape((1, 2))
        gaze_prediction = np.asarray(gaze_prediction.cpu().numpy)

        if sample['side'] == 'right':
            gaze_prediction[0, 1] = gaze_prediction[0, 1]

        angular_error = utilities.gaze.angular_error(gaze, gaze_prediction)
        errors.append(angular_error)
        print('-----')
        print('error', angular_error)
        print('mean error', np.mean(errors))
        print('side', sample['side'])
        print('gaze', gaze)
        print('gaze prediction', gaze_prediction)
