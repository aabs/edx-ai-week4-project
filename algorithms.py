# a file to simply implement some key algorithms in a procedural fashion
from collections import namedtuple

import logging


class SolutionContext:
    def __init__(self
                 , board=None
                 , depth=None
                 , timeout=None
                 , previous_move=None
                 , fn_fitness=None
                 , fn_terminate=None):
        self.previous_move = previous_move
        self.board = board
        self.depth = depth
        self.timeout = timeout
        self.fn_fitness = fn_fitness
        self.fn_terminate = fn_terminate


class Solution:
    def __init__(self
                 , move=None
                 , board=None
                 , is_max=None):
        self.board = board
        self.is_max = is_max
        self.move = move


MinMove = namedtuple('MinMove', ['is_max', 'x', 'y', 'tile', 'prob'])
MaxMove = namedtuple('MaxMove', ['is_max', 'direction'])

log = logging.getLogger('app' + __name__)

def minimax(context: SolutionContext, solution: Solution):
    log.info("minimax")
    if context.fn_terminate(context, solution):
        return context.fn_fitness(context, solution)
    moves = solution.board.get_moves(not solution.is_max)

    if solution.is_max:
        results = []
        for m in moves:
            new_context, new_solution = creat_call_vars(m, context, solution)
            results.append(minimax(context=new_context,
                                   solution=new_solution))
        return max(results)
    else:
        results = []
        for m in moves:
            new_context, new_solution = creat_call_vars(m, context, solution)
            r = minimax(context=new_context, solution=new_solution)
            r2 = r * new_solution.move.prob
            results.append(r2)
        return min(results)


def creat_call_vars(move, context, solution):
    new_context = SolutionContext(board=solution.board,
                                  depth=context.depth + 1,
                                  timeout=context.timeout,
                                  previous_move=move,
                                  fn_fitness=context.fn_fitness,
                                  fn_terminate=context.fn_terminate)
    new_solution = Solution(move=move,
                            board=new_context.board.move(move),
                            is_max=not solution.is_max)
    return new_context, new_solution
