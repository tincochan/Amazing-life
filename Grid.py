import numpy as np
import torch
from numpy import full
from Cell import Cell

class Grid():

    def __init__(self, cell_size, grid_size, network_params_size, device):
        self.cell_size = cell_size
        self.size = grid_size
        self.channels = grid_size[2]
        self.res = (grid_size[0] * cell_size, grid_size[1] * cell_size)
        # Inits data with black cells by default, no network is configured
        self.data = full(self.size, Cell(network_params_size, device).vector())

    @staticmethod
    def getColorChannels(data):
        return data[:, :, :3]

    @staticmethod
    def getFitnessChannels(data):
        return data[:, :, -2:-1]

    #TODO make 3x3 state of color channels based on which x, y pos of cell is passed in
    # Note: need to preserve gradient of tensors
    @staticmethod
    def getPartialColorChannels(data, x, y):
        # vector_neighbors = np.zeros(shape=(4, 3, 3))
        vector_neighbors = torch.zeros((3, 3, 3))
        # Get cell's neighbors, 3x3
        for nx in range(-1, 2):
            for ny in range(-1, 2):
                vector_neighbors[nx + 1][ny + 1][0] = data[x + nx, y + ny, 0]
                vector_neighbors[nx + 1][ny + 1][1] = data[x + nx, y + ny, 1]
                vector_neighbors[nx + 1][ny + 1][2] = data[x + nx, y + ny, 2]
        # return torch.cat(vector_neighbors)
        return vector_neighbors

    @staticmethod
    def getPartialFitnessChannels(data, x, y):
        vector_neighbors = torch.zeros((3, 3, 1))
        for nx in range(-1, 2):
            for ny in range(-1, 2):
                vector_neighbors[nx + 1][ny + 1][0] = data[x + nx, y + ny, -1] # -1 fitness channel
        return vector_neighbors
        # return torch.tensor(vector_neighbors)
