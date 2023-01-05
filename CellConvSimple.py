import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from Grid import Grid
import math


class CellConvSimple(nn.Module):
    '''
    Simple conv autoencoder implementation but conv autoencoder style, without fc layer out
    '''

    def __init__(self, output_shape=(100, 100, 4), observability='partial'):
        '''

        :param block_class: ResidualBlock class to use to make layer modules
        :param layers: List of length 4 where each ind is the number of residual
                block_classs in the layer corresponding to its index
        :param output_shape: (grid_h, grid_w, channels) of the shape of the output
        :param observability: If partial, cell only receives input from neighbors when predicting next frame
        '''
        super(CellConvSimple, self).__init__()
        self.output_shape = output_shape

        self.layer0 = nn.Conv2d(9, 64, 3, stride=1, padding=1)
        self.layer1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.layer3 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.layer4 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.layer5 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        # self.layer6 = nn.Conv2d(128, 4, 3, stride=1, padding=1)
        self.layer6 = nn.Conv2d(128, 9, 3, stride=1, padding=1)
        self.encoder = nn.Sequential(
            self.layer0,
            nn.ReLU(True),
            self.layer1,
            nn.ReLU(True),
            self.layer2,
            nn.ReLU(True),
            self.layer3,
            nn.ReLU(True),
            # Layers 4-6 for partial predictions
            self.layer4,
            nn.ReLU(True),
            self.layer5,
            nn.ReLU(True),
            self.layer6,
            nn.ReLU(True),
        )
        self.up1 = nn.ConvTranspose2d(512, 128, stride=2, kernel_size=3)  # (1, 128, 7, 7)
        self.up2 = nn.ConvTranspose2d(128, 32, stride=3, kernel_size=5)  # (1, 32, 23, 23)
        self.up3 = nn.ConvTranspose2d(32, 8, stride=3, kernel_size=5)  # (1, 8, 71, 71)
        self.up4 = nn.ConvTranspose2d(8, 4, stride=2, kernel_size=2, padding=21)  # (1, 4, 100, 100)
        self.lrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def deconvolve(self, x):
        x = self.up1(x)
        x = self.lrelu(x)
        x = self.up2(x)
        x = self.lrelu(x)
        x = self.up3(x)
        x = self.lrelu(x)
        x = self.up4(x)
        x = self.tanh(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        # x = self.deconvolve(x) # No deconvolve for partial observability
        return x

    '''
    Represent the first and last layers of the network as flattened numpy arrays of those parameters, concatenated
    '''
    def getNetworkParamVector(self, device):
        first_params = self.layer0.parameters().__next__().detach()
        last_params = self.layer3.parameters().__next__().detach()
        first, last = CellConvSimple.firstLastParams(device, first_params, last_params)
        first = first.cpu().detach().numpy().flatten()
        last = last.cpu().detach().numpy().flatten()
        network_param_vector = np.concatenate([first, last])
        return network_param_vector

    @staticmethod
    def getParams(layer):
        size = 1
        for dim in layer.shape:
            size *= dim
        return size

    '''
    Reduces the size of first and last layer params to smaller number of params by first reducing channel numbers w 
    1-d convolutions, then simply truncating if that does not work... may need to choose one strat or the other
    '''
    @staticmethod
    def firstLastParams(device='mps', *args):
        layers_params = []
        for layer_params in args:
            size = CellConvSimple.getParams(layer_params)

            # Reducing channel size until params are manageable. If flattening channels to 1 ch isn't enough,
            # reduce samples in batch by half until under 1500
            last_out = 0
            while size > 1500:
                out = layer_params.shape[1] // 2 if layer_params.shape[1] // 2 != 0 else 1
                if last_out == out:
                    with torch.no_grad():
                        dim0 = layer_params.shape[0] // 2
                        layer_params = layer_params[:dim0, :layer_params.shape[1],
                                       :layer_params.shape[2], :layer_params.shape[3]]
                conv = torch.nn.Conv2d(in_channels=layer_params.shape[1], out_channels=out, kernel_size=1).to(device)
                layer_params = conv(layer_params.to(device))
                size = CellConvSimple.getParams(layer_params)
                last_out = out

            layers_params.append(layer_params)
        return layers_params

    @staticmethod
    def sigmoid(x):
        sig = 1 / (1 + math.exp(-x))
        return sig

    def getNetworkColor(self):
        # Note: only using first 5 weights
        # sum of first layer weights, sum of last layer weights, sum of middle layer weights
        first_params = self.layer0.parameters().__next__().cpu().detach().numpy()
        middle_params = self.layer1.parameters().__next__().cpu().detach().numpy()
        last_params = self.layer3.parameters().__next__().cpu().detach().numpy()
        color = [self.sigmoid(np.average(first_params[:10])),
                 self.sigmoid(np.average(last_params[:10])/2),
                 self.sigmoid(np.average(middle_params[:10]))]
        # Normalize color to not be gray
        color = np.subtract(color, np.min(color))
        color = np.divide(color, np.max(color) - np.min(color))
        color = np.multiply(color, 255)
        return color

    '''
    Convert from (n, c, h, w) format of network output to (h, w, c) format of the grid
    '''
    @staticmethod
    def reshape_output(output: torch.Tensor, output_shape):
        output = torch.squeeze(output)
        output = torch.reshape(output, output_shape)
        return output

'''
Computes the loss of a cell based on the predicted state of the whole grid (color and fitness) vs actual
'''
def CA_Loss(y_pred, y):
    next_frame_pred = Grid.getColorChannels(y_pred)
    target_frame = Grid.getColorChannels(y)
    fit_preds = Grid.getFitnessChannels(y_pred)
    fit_targets = Grid.getFitnessChannels(y)
    with torch.enable_grad():
        frame_loss = F.mse_loss(next_frame_pred, torch.from_numpy(target_frame))
        fit_loss = F.mse_loss(fit_preds, torch.from_numpy(fit_targets))
        losses = torch.tensor([frame_loss, fit_loss])
        norm_loss = torch.sum(losses) / len(losses)
    return norm_loss.requires_grad_()

'''
Computes the loss of a cell based on the predicted state of a partial 3x3 grid of neighbors (color and fitness channels) vs actual
'''
def partial_CA_Loss(pred, actual, x, y):
    next_frame_pred = pred[:, :, 0:3]
    target_frame = actual[:, :, 0:3]
    fit_preds = pred[:, :, -1]
    fit_targets = actual[:, :, -1]
    with torch.enable_grad():
        frame_loss = F.mse_loss(next_frame_pred, torch.from_numpy(target_frame))
        fit_loss = F.mse_loss(fit_preds, torch.from_numpy(fit_targets))
        losses = torch.tensor([frame_loss, fit_loss])
        norm_loss = torch.sum(losses) / len(losses)
    return norm_loss.requires_grad_()
