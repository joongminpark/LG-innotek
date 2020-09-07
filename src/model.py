from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from typing import Dict


##########
# Layers #
##########
class Flatten(nn.Module):
    """Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to 2-dimensional Tensor
    of shape [batch_size, d1*d2*...*dn].
    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class GlobalMaxPool1d(nn.Module):
    """Performs global max pooling over the entire length of a batched 1D tensor
    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return nn.functional.max_pool1d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


class GlobalAvgPool2d(nn.Module):
    """Performs global average pooling over the entire height and width of a batched 2D tensor
    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return nn.functional.avg_pool2d(input, kernel_size=input.size()[2:]).view(-1, input.size(1))


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.
    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


def feature_extractor(num_input_channels=3) -> nn.Module:
    """Creates afeature extractor as used in LG serial data
    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain.
    """
    return nn.Sequential(
        conv_block(num_input_channels, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        conv_block(64, 64),
    )

##########
# Models #
##########
class LG_model(nn.Module):
    """Performs 3 input extractor, and classfy label
    # Arguments
        input: X, Y, Z images
    """
    def __init__(self, num_input_channels=3):
        super(LG_model, self).__init__()
        self.feature_extractor = feature_extractor(num_input_channels)
        self.global_average = GlobalAvgPool2d()
        self.classifier = nn.Linear(64, 2)

    def forward(self, img_x, img_y, img_z):
        # dim: (N,C,H,W)
        output_x = self.feature_extractor(img_x)
        output_y = self.feature_extractor(img_y)
        output_z = self.feature_extractor(img_z)

        # concatenate channel
        output_x_y_z = torch.cat([output_x, output_y, output_z], dim=1)

        # average pooling
        avg_x_y_z = self.global_average(output_x_y_z)

        logit = self.classifier(avg_x_y_z)

        return logit