This repository contains the implementation of Appearance based gaze learning.

My attempt at reciprocating results achieved in the paper.

Next step is training SLYKGaze.
To achieve this, the following are to be done:

TODO:
- Pre-append a BYOL adaptation neural network to extract latent representation in the image.
- Preprocess ETH-X Gaze dataset 
- The latent representation is used to enhance the the dataset for training SLYKGaze, and enhance input of the output model for performing inference.
- Modify GazeML model, add uncertainty network (BNN) after fcn layer.
- Add a Linear regression network as a new layer, taking gaze angles and eye landmarks as input.
- The linear Regression model network predicts gaze point towards the Point of Regard
