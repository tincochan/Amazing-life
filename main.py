from sys import exit
from math import ceil
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from Cell import Cell
from CellConv import CellConv
from ResidualBlock import ResidualBlock
from CellConvSimple import CellConvSimple
import pygame
from Grid import Grid
from CellConvSimple import partial_CA_Loss

# Constants
DEBUG = False
OBSERVABILITY = 'partial'
CELL_SIZE = 10  # pixels
GRID_W = 100  # cells
GRID_H = 100  # cells
FIT_CHANNELS = 1
MOVE_CHANNELS = 5
NUM_EPOCHS = 1
# Hyperparams
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.001
MOMENTUM = 0.97
RANDOM_MOVE_CHANCE = 0.1
# FREEZE = True
FREEZE = False
CLOCK = pygame.time.Clock()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Device", device)

'''Sets up the size constant of the channels that each cell in the grid will hold. We will have 3 for color, 
some amount for representing the parameters of the conv net at that location, and some to represent the fitness of 
the cell. '''


# TODO Biochemical signaling channels? movement channel
def setChannels(first_params, last_params):
    network_params_size = ((first_params.cpu().detach().numpy().flatten().size +
                            last_params.cpu().detach().numpy().flatten().size))
    # 3 channels for rgb, + network params size + fitness channels
    CHANNELS = 3 + network_params_size + MOVE_CHANNELS + FIT_CHANNELS
    return CHANNELS


'''
This class continuously draws the grid on the pygame window according to cell update rules and dynamics
'''


class CAGame():

    def __init__(self):
        # cell_net = CellConv(ResidualBlock, [3, 4, 6, 3], observability=OBSERVABILITY).to(device)
        cell_net = CellConvSimple().to(device)
        first_params = cell_net.layer0.parameters().__next__().detach()
        last_params = cell_net.layer3.parameters().__next__().detach()
        # first_params, last_params = CellConv.firstLastParams(first_params, last_params)
        first_params, last_params = CellConvSimple.firstLastParams(device, first_params, last_params)
        self.network_params_size = ((first_params.cpu().detach().numpy().flatten().size +
                                     last_params.cpu().detach().numpy().flatten().size))
        CHANNELS = setChannels(first_params, last_params)
        OUTPUT_SHAPE = (GRID_H, GRID_W, CHANNELS)
        cell_net.output_shape = OUTPUT_SHAPE

        # Representation Invariant:
        # intermediate_cell_grid should always begin the frame in the same state as self.cell_grid
        # Grid of Cell objects, corresponding with their vectorized forms stored below for computation
        self.cell_grid: [[Cell]] = [[0] * GRID_W for _ in range(GRID_H)]
        # Has extra dim so multiple cells can occupy the same place in the grid; (100, 100, 1)
        # Intermediate cell grid holds cells after they try to move, but before they have been eaten to form the
        # final next frame of self.cell_grid
        self.intermediate_cell_grid = [[[] for col in range(GRID_W)] for row in range(GRID_H)]
        for r in range(GRID_H):
            for c in range(GRID_W):
                new_cell = Cell(self.network_params_size, device)
                new_cell.y = r
                new_cell.x = c
                self.cell_grid[r][c] = new_cell
                self.intermediate_cell_grid[r][c].append(new_cell)
        self.checkRep()

        # Grid, holding vectorized cells in data used to actually get loss of cells
        self.grid = Grid(CELL_SIZE, grid_size=(GRID_W, GRID_H, CHANNELS),
                         network_params_size=self.network_params_size, device=device)
        self.screen = pygame.display.set_mode(self.grid.res)
        self.empty_vector = np.zeros((CHANNELS))
        pygame.display.set_caption("Cellular Automata", "CA")

    '''
    Enforces updating the corresponding grid data when the cell object changes
    '''

    def updateCellGrid(self, cell, x, y):
        self.cell_grid[y][x] = cell
        self.grid.data[y][x] = cell.vector()
        cell.x = x
        cell.y = y

    """ Moves cell based on their movement vectors"""
    def moveCellsInIntermediateCellGrid(self, cell, x, y):
        # Assume that cell.move contains direction already
        movement_vector = cell.getMovement(x, y, self.grid)
        direction = np.argmax(movement_vector)  # [0, 0, 0, 0, 0]
        #  stay, left, right, up, down
        if direction == 1:
            next_pos = x - 1, y
        elif direction == 2:
            next_pos = x + 1, y
        elif direction == 3:
            next_pos = x, y - 1
        elif direction == 4:
            next_pos = x, y + 1
        else:  # 0
            next_pos = x, y

        nx, ny = next_pos
        # If the intermediate grid is empty at the place we're moving to, set it to the new cell
        if np.allclose(self.intermediate_cell_grid[ny][nx][0].color, 0):
            self.intermediate_cell_grid[ny][nx][0] = cell
        # Otherwise, append the cell to the competing cells at that spot
        else:
            self.intermediate_cell_grid[ny][nx].append(cell)
        self.checkIntermediateEmpty()
        # Make intermediate cell at x, y be empty if cell moved away
        if y != ny or x != nx:
            cell = Cell(network_param_size=self.network_params_size, device=device)
            self.intermediate_cell_grid[y][x] = [cell]
        self.checkIntermediateEmpty()

    def getPartialFrame(self, cell, frame_size=(3, 3)):  # will break if cells are on the border
        vector_neighbors = np.zeros(shape=(frame_size[0], frame_size[1], 9))
        x = cell.x
        y = cell.y
        # print('cell accessed partial frame at: (' + str(x) + ', ' + str(y) + ')')
        for nx in range(-1, 2):
            for ny in range(-1, 2):
                if x + nx >= GRID_W or y + ny >= GRID_H:
                    vector_neighbors[ny + 1][nx + 1][:3] = self.empty_vector[:3]
                    vector_neighbors[ny + 1][nx + 1][3:9] = self.empty_vector[-6:]
                else:
                    vector_neighbors[ny + 1][nx + 1][:3] = self.grid.data[y + ny, x + nx, :3]
                    vector_neighbors[ny + 1][nx + 1][3:9] = self.grid.data[y + ny, x + nx, -6:]
        return vector_neighbors

    def getFullFrame(self):
        vector_neighbors = np.zeros(shape=(3, 3, 9))
        vector_neighbors[:, :, 3] = self.grid.data[:, :, :3]
        vector_neighbors[:, :, 3:9] = self.grid.data[:, :, -6:]
        return vector_neighbors

    def testCellConv(self, num_cells=20):
        assert(num_cells < (GRID_H - 1) * (GRID_W - 1))
        color = [256, 0, 0]
        x_locations = np.arange(1, GRID_W - 1)  # Can't spawn on the borders
        y_locations = np.arange(1, GRID_H - 1)  # Can't spawn on the borders
        # Generate random xy spawn coordinates
        xs = np.random.choice(x_locations, num_cells, replace=True)
        ys = np.random.choice(y_locations, num_cells, replace=True)
        xy = np.squeeze(np.dstack((xs, ys)))
        # nothing, left, right, up, down
        directions = [0, 1, 2, 3, 4]
        print("\nGenerating {} cells!".format(num_cells))

        # Add initial cells to the grid objects
        for i in (range(0, num_cells)):
            valid_directions = directions.copy()
            cell_net = CellConvSimple().to(device)
            cell = Cell(color=color, network_param_size=self.network_params_size,
                        network=cell_net, fitness=10, device=device)
            x, y = xy[i]
            # print('generated cell at: (' + str(x) + ', ' + str(y) + ')')
            if x == 98:
                valid_directions.remove(2)
            if x == 1:
                valid_directions.remove(1)
            if y == 98:
                valid_directions.remove(4)
            if y == 1:
                valid_directions.remove(3)
            cell.move = [0, 0, 0, 0, 0]
            if not FREEZE:
                direction = np.random.choice(valid_directions)
                cell.move[direction] = 1
            if i == num_cells // 2:
                print("Halfway done!")
            # Put the generated cell into the cell grid
            self.updateCellGrid(cell, x, y)
            self.refreshIntermediateCellGrid()
            self.checkRep()
        print("Generated {} cells".format(num_cells))


    # Loop through each cell, get movement, update intermediate cell grid
    # after temporarily moving all cells, loop through again and see if any moved to the same place
    # if so call eatCells to break the tie and updateCellGrid, else just updateCellGrid
    def moveCell(self, cell):
        # for y, row in enumerate(self.cell_grid):
        #     for x, cell in enumerate(row):
        if cell.network:
            movement_vector = Cell.getMovement(cell.x, cell.y, self.grid, p=RANDOM_MOVE_CHANCE)
            if FREEZE:
                cell.move = [1, 0, 0, 0, 0]
            else:
                cell.move = movement_vector
            self.moveCellsInIntermediateCellGrid(cell, cell.x, cell.y)

    ''' After cells in intermediate cell grid reflect movements, check if there are eating cell conflicts, resolve them'''
    def resolveIntermediateCellGrid(self):
        for y, row in enumerate(self.intermediate_cell_grid):
            for x, cell in enumerate(row):
                # Remove cells that moved away if intermediate cell grid is black and cell grid is not
                if cell[0].color.all() == 0 and (not self.cell_grid[y][x].color.all() == 0):
                    self.cell_grid[y][x] = Cell(self.network_params_size, device)
                    self.grid.data[y][x] = self.empty_vector
                if len(cell) > 1:
                    dominant_cell = self.eatCells(x, y)
                    self.updateCellGrid(dominant_cell, x, y)
                else:
                    self.updateCellGrid(cell[0], x, y)
        # After these updates, intermediate and cell grid should be same
        self.checkRep()



    # If cells move on top of each other, check how to break ties / which gets eaten and which replicates
    # Free the memory used by the eaten cell and allocate new instance of replicated
    def eatCells(self, x, y):
        # call if computed updated grid has two cells
        # assumes more than one cell at self.intermediate_cell_grid
        dominant_cell = self.intermediate_cell_grid[y][x][0]
        for cell in self.intermediate_cell_grid[y][x]:
            if cell.fitness > dominant_cell.fitness:
                dominant_cell = cell
        return dominant_cell

    # do for every cell, add neighbors' fitness predictions to cell as we go
    # update each cell's fitness at end of running all training of cells in grid
    def updateCell(self, node: Cell, previous_grid=None):
        vector_neighbors = np.zeros(shape=(9, 3, 3))
        neighbors = []
        x = node.x
        y = node.y
        # Get cell's neighbors, 3x3
        for nx in range(-1, 2):
            for ny in range(-1, 2):
                vector_neighbors[0][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, 0]
                vector_neighbors[1][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, 1]
                vector_neighbors[2][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, 2]
                vector_neighbors[3][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, -6]
                vector_neighbors[4][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, -5]
                vector_neighbors[5][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, -4]
                vector_neighbors[6][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, -3]
                vector_neighbors[7][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, -2]
                vector_neighbors[8][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, -1]
                neighbor = self.cell_grid[y + ny][x + nx]
                neighbors.append(neighbor)

        node.last_neighbors = vector_neighbors
        # After we update the cell, update the previous neighbors to the current grid config
        # Removes the network params from the grid state
        # full_state = np.dstack((self.grid.data[:, :, :3], self.grid.data[:, :, -1]))
        # pred, loss = CellConv.train_module(node, full_state=full_state, prev_state=previous_grid, num_epochs=NUM_EPOCHS)
        pred = self.train_module(node, num_epochs=NUM_EPOCHS)
        # todo update cell.fitness property based on loss
        return pred

    ''' 
    Can switch between passing in full previous state or only partially observable prev state / neighbors
    '''
    def train_module(self, cell, full_state=None, prev_state=None, num_epochs=1):
        net = cell.network

        # note, can't run more than one epoch w partial-partial structure
        for epoch in (range(num_epochs)):
            net = net.float()
            input = torch.from_numpy(cell.last_neighbors.astype(np.double))
            # Adds dimension to input so that it has n, c, w, h format for pytorch
            input = input[None, :, :, :]
            input = input.float().requires_grad_()
            input = input.to(device)
            next_pred = net(input)
            partial_pred_shape = (3, 3, 9)  # 9 channels: 3 color, 5 movement, 1 fitness
            # next_full_state_pred = CellConvSimple.reshape_output(next_full_state_pred, full_state.shape)
            next_pred = CellConvSimple.reshape_output(next_pred, partial_pred_shape)
            # take movement from the middle cell of the output
            if FREEZE:
                cell.move = [0, 0, 0, 0, 0]
            else:
                cell.move = next_pred[1][1][-6:]
            # give movement to the grid which updates intermediate grid
            self.moveCell(cell)
            # once movements have all been calculated, give next frame to cell and backprop the loss
            # call resolve intermediate...

        # return next_pred, loss.item()
        return next_pred

    def cellBackprop(self, cell):
        net = cell.network
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,  # TODO Check weight decay params
            momentum=MOMENTUM)
        next_frame = self.getPartialFrame(cell)  # default numpy (3, 3, 9)
        loss = partial_CA_Loss(cell.pred.cpu(), next_frame, cell.x, cell.y)
        # print('pred: ', next_full_state_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cell.updateColor()
        # cell.updateFitness()
        return loss.item()

    '''Sets intermediate cell grid to be the same as cell_grid'''
    def refreshIntermediateCellGrid(self):
        # Clear intermediate cell grid for next iteration
        self.intermediate_cell_grid = [[ [] for col in range(GRID_W)] for row in range(GRID_H)]
        for r in range(GRID_H):
            for c in range(GRID_W):
                self.intermediate_cell_grid[r][c].append(self.cell_grid[r][c])

    # MARK: pygame stuff
    '''
    Handle keyboard presses and other events
    '''

    def eventHandler(self):
        # Handles events sent by the user
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # find the cords to "place" a cell
                    (Mx, My) = pygame.mouse.get_pos()
                    Nx, Ny = ceil(Mx / self.grid.cell_size), ceil(My / self.grid.cell_size)
                    # XXX todo place some color of cell there / init new cell
                    # if self.grid.data[Nx, Ny].key == 0:
                    #     self.GameFlip.get(Nx, Ny).key = 1
                    # else:
                    #     self.GameFlip.get(Nx, Ny).key = 0

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.type == pygame.QUIT:
                    pygame.display.quit(), exit()

    '''
    Draw everything on the screen
    '''

    def draw(self):
        # draws cells onto the screen
        for x, row in enumerate(self.grid.data):
            for y, cell in enumerate(row):
                # if self.GameFlip.get(node.Xm, node.Ym).key == 1:
                # rect is (left, top, width, height)
                pygame.draw.rect(self.screen, Cell.getCellColor(x, y, self.grid),
                                 rect=((x) * self.grid.cell_size,
                                       (y) * self.grid.cell_size,
                                       self.grid.cell_size, self.grid.cell_size))

        # draw lines on the grid
        for column in range(1, GRID_W):
            pygame.draw.line(self.screen, "gray", (column * self.grid.cell_size, 0),
                             (column * self.grid.cell_size, GRID_H * self.grid.cell_size))

        for row in range(1, GRID_H):
            pygame.draw.line(self.screen, "gray", (0, row * self.grid.cell_size),
                             (GRID_W * self.grid.cell_size, row * self.grid.cell_size))

    def startGame(self):
        iterations = 100
        pbar = tqdm(total=iterations)
        itr = 0
        self.testCellConv(num_cells=50)
        running = True
        while running:
            print("iteration", itr)
            CLOCK.tick(70)  # Makes game run at 70 fps or slower
            self.checkRep()
            for row in self.cell_grid:
                for cell in row:
                    if cell.network:
                        pred = self.updateCell(cell)
                        cell.pred = pred
            self.checkIntermediateEmpty()
            self.resolveIntermediateCellGrid()
            for i, row in enumerate(self.cell_grid):
                for j, cell in enumerate(row):
                    cell.x = j  # x is col
                    cell.y = i  # y is row
                    if cell.network:
                        # print('cell backprop called at: (', cell.x, ', ', cell.y, ')')
                        # print('iteration called at: (', j, ', ', i, ')')
                        loss = self.cellBackprop(cell)
                        cell.losses.append(loss)
            self.checkIntermediateEmpty()
            self.refreshIntermediateCellGrid()

            self.draw()
            self.eventHandler()
            pygame.display.flip()
            itr += 1
            pbar.update(1)
            if itr == iterations:
                running = False
        pbar.close()

        count = 0
        final_cell_losses = []
        cellcount = 0
        for i, row in enumerate(self.cell_grid):
            for j, cell in enumerate(row):
                if cell.network:
                    cellcount += 1
                    final_cell_losses.append(cell.losses[-1])
                    if count < 15:  # plot [#] number of cells
                        plt.title('Loss vs. Epoch for cell (' + str(j) + ', ' + str(i) + ')')
                        plt.ylabel('Loss')
                        plt.xlabel('Epoch')
                        # plt.xlim((0, 100))
                        plt.plot(np.arange(len(cell.losses)), cell.losses, 'g-')
                        # plt.legend(loc="upper right")
                        plt.show()
                        count += 1
        print('cellcount ', cellcount)
        print('avg. cell loss:', np.sum(final_cell_losses) / len(final_cell_losses))

    def checkRep(self):
        if DEBUG:
            for y, row in enumerate(self.intermediate_cell_grid):
                for x, col in enumerate(row):
                    try:
                        assert np.allclose(col[0].color, self.cell_grid[y][x].color)
                    except AssertionError as e:
                        raise AssertionError("intermediate diff from cell grid at (" + str(x) + ", " + str(y) + ")")

    def checkIntermediateEmpty(self):
        if DEBUG:
            all_empty = True
            empty_cell = Cell(self.network_params_size, device)
            for y, row in enumerate(self.intermediate_cell_grid):
                for x, col in enumerate(row):
                    # if col[0] != empty_cell:
                    if col[0].network:
                        all_empty = False
            if all_empty:
                raise AssertionError("intermediate grid is probably entirely empty")


def main():
    ca = CAGame()
    ca.startGame()


if __name__ == '__main__':
    main()
