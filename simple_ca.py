import random
from cellular_automaton import CellularAutomaton, MooreNeighborhood, CAWindow, EdgeRule
from typing import Sequence


'''
This is an example of a CA made with prebuilt libraries instead of pygame.
'''
class CA(CellularAutomaton):

    def __init__(self):
        super().__init__(dimension=[200, 200],
                         neighborhood=MooreNeighborhood(
                             EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS))

    def init_cell_state(self, __) -> Sequence:
        rand = random.randrange(0, 101, 1)
        init = max(.0, float(rand - 99))
        return [init * random.randint(0, 3)]

    def evolve_rule(self, __, neighbors_last_states: Sequence) -> Sequence:
        return self._neighborhood.get_neighbor_by_relative_coordinate(neighbors_last_states, (-1, -1))


def state_to_color(current_state: Sequence) -> Sequence:
    # Red, green, blue colors depending on state, black if none of those states
    return (255 if current_state[0] == 1 else 0,
            255 if current_state[0] == 2 else 0,
            255 if current_state[0] == 3 else 0)


if __name__ == "__main__":
    CAWindow(cellular_automaton=CA(),
             window_size=(1000, 830),
             state_to_color_cb=state_to_color).run()


# By hand:
# 2d matrix of coordinates of cells
# for cell in matrix, run update func
# update matrix of cell color, underlying state values as well (other channels besides rgb to contain state info)
# plot updates with imshow or pygame