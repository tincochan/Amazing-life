# amazing life

Cellular automata where each cell contains a convolutional network. The rules of replication of a cell depend on the accuracy of the convolution network's accuracy of predicting the next frame of the cellular automata. The goal is that the conv net that most accurately understands the environment into which it was placed will visually proliferate in the cellular automata, providing an interesting glimpse into the process of meta-cognition.

Game:  
Some conv nets with randomized weights are spawned on a 100x100 grid.  

We begin one iteration of the game:

Each conv net receives input about the  

    1) Color of itself and the surrounding 8 cells  
    2) Weights of the first and last layers of itself and the surrounding 8 cells  
    3) 'Fitness' of itself and surrounding 8 cells  

Each conv net cell then predicts the color value and fitness of the entire 100x100 grid  
Each cell also outputs a direction of movement  

Loss for each prediction is generated based on the differences  

The cell's loss is backpropagated to its weights  
The cell's color is updated based on how the weights have changed  
The cell's fitness score is calculated based on how accurate its prediction was 
(as well as how fit its neighbors predicted it to be)  
The grid then updates the display based on the movement vectors outputted by cells  
This completes one iteration of the game.  It continues until you're bored??  

Cells that move on top of each other break ties with the fitter cell eating the less fit cell.  

