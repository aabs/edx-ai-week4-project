import logging
import sys

import time

from BaseAI_3 import BaseAI
from CompositeCalculation import CompositeUtilityCalculator
from FastGrid import FastGrid
from algorithms import *

deadline_offset = 0.1  # mandated solution timeout for exercise is .1 secs
max_depth_allowed = 4  # how deep to search for solutions

# some constants for initialising alpha and beta values in minimax
plus_infinity = float(sys.maxsize)
minus_infinity = -1.0 * plus_infinity

logging.basicConfig(
    filename="2048.log",
    format="%(levelname)-10s %(asctime)s %(message)s",
    level=logging.INFO
)
log = logging.getLogger('app' + __name__)


class PlayerAI(BaseAI):
    def __init__(self):
        self.fitness = CompositeUtilityCalculator()

    def getMove(self, slowgrid):
        grid = FastGrid(slowgrid)
        ctx = SolutionContext(board=grid
                              , depth=0
                              , timeout=time.process_time() + 0.1
                              , previous_move=None
                              , fn_fitness=lambda c, s: self.fitness.compute_utility(s.board)
                              , fn_terminate=lambda c, s: ctx.depth == 4 or s.board.canMove())
        solutions = [(m, minimax(ctx, Solution(move=m,
                                               board=ctx.board.move(m.direction),
                                               is_max=True)))
                     for m in grid.get_moves(True)]

        result = max(solutions, key=lambda s: s[1])
        return result[0].direction

# some resources
# http://www.wikihow.com/Beat-2048#Step_by_Step_Strategy_Guide_sub
# http://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048
