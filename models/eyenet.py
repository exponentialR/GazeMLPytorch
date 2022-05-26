import torch
from torch import nn
from models.layers import Conv, Hourglass, Pool, Residual
from models.losses import HeatmapLoss
from utilities.softargmax import softargmax2d


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, batch_norm=False)

    def forward(self, x):
        return self.conv(x)


class EyeNet(nn.Module):
    def __init__(self, num_stack, num_features, num_landmarks, batch_norm=False, increase=0, **kwargs):
        super(EyeNet, self).__init__()

        self.image_width = 160
        self.image_height = 96
        self.num_stack = num_stack
        self.num_features = num_features
        self.num_landmarks = num_landmarks

        self.heatmap_width = self.image_width / 2
        self.heatmap_height = self.image_height / 2

        self.num_stack = num_stack
        self.pre_network = nn.Sequential(
            Conv(1, 64, 7, 1, batch_norm=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, num_features)
        )
        self.pre_network_2 = nn.Sequential(
            Conv(1, 64, 7, 1, batch_norm=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, num_features)
        )

        self.hourglass = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, num_features, batch_norm, increase), ) for i in range(num_stack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(num_features, num_features),
                Conv(num_features, num_features, 1, batch_norm=True, relu=True)
            ) for i in range(num_stack)
        ])

        self.outputs = nn.ModuleList(
            [Conv(num_features, num_landmarks, 1, relu=False, batch_norm=False) for i in range(num_stack)])
        self.merge_features = nn.ModuleList([Merge(num_features, num_features) for i in range(num_stack - 1)])
        self.merge_predictions = nn.ModuleList([Merge(num_landmarks, num_features) for i in range(num_stack - 1)])

        self.gaze_fc1 = nn.Linear(
            in_features=int(num_features * self.image_width * self.image_height / 64 + num_landmarks * 2),
            out_features=256)
        self.gaze_fc2 = nn.Linear(in_features=256, out_features=2)

        self.num_stack = num_stack
        self.heatmapLoss = HeatmapLoss()
        self.landmarkLoss = nn.MSELoss
        self.gaze_loss = nn.MSELoss

    def forward(self, images):
        # images of size 1, height, width
        x = images.unsqueeze(1)
        x = self.pre_network(x)
        gaze_x = self.pre_network_2(x)
        gaze_x = gaze_x.flatten(start_dim=1)

        combined_hm_preds = []
        for i in torch.arange(self.num_stack):
            hourglass = self.hourglass[i](x)
            feature = self.features[i](hourglass)
            predictions = self.outputs[i](feature)
            combined_hm_preds.append(predictions)
            if i < self.num_stack - 1:
                x = x + self.merge_predictions[i](predictions) + self.merge_features[i](feature)

        heatmaps_output = torch.stack(combined_hm_preds, 1)
        landmarks_output = softargmax2d(predictions)  # N x num_landmarks x 2

        # Gaze
        gaze = torch.cat((gaze_x, landmarks_output.flatten(start_dim=1)), dim=1)
        gaze = self.gaze_fc1(gaze)
        gaze = nn.functional.relu(gaze)
        gaze = self.gaze_fc2(gaze)

        return heatmaps_output, landmarks_output, gaze

    def calculate_loss(self, combined_hm_predictions, heatmaps, landmarks_prediction, landmarks, gaze_predicton, gaze):
        combined_loss = []
        for i in range(self.num_stacks):
            combined_loss.append(self.heatmapLoss(combined_hm_predictions[:, i, :], heatmaps))

        heatmap_loss = torch.stack(combined_loss, dim=1)
        landmark_loss = self.landmarkLoss(landmarks_prediction, landmarks)
        gaze_loss = self.gaze_loss(gaze_predicton, gaze)

        return torch.sum(heatmap_loss), landmark_loss, 1000 * gaze_loss
