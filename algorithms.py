# a file to simply implement some key algorithms in a procedural fashion
from collections import namedtuple

import logging


class SolutionContext:
    def __init__(self
                 , board=None
                 , alpha=None
                 , beta=None
                 , depth=None
                 , timeout=None
                 , previous_move=None
                 , fn_fitness=None
                 , fn_terminate=None):
        self.beta = beta
        self.alpha = alpha
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

def minimax_with_ab_pruning(context: SolutionContext, solution: Solution):
    log = logging.getLogger('PlayerAI')

    if context.fn_terminate(context, solution):
        return context.fn_fitness(context, solution)

    moves = solution.board.get_moves(solution.is_max)

    if solution.is_max:
        best_result = -float("inf")
        for m in moves:
            new_context, new_solution = creat_call_vars(m, context, solution)
            result = minimax_with_ab_pruning(context=new_context, solution=new_solution)
            best_result = max(result, best_result)
            context.alpha = max(best_result, context.alpha)
            if context.alpha <= context.beta:
                log.debug("alpha cut")
                break
        return best_result
    else:
        # PROBLEM:
        #  - MIN is not playing to minimise the eventual score of MAX, it is generating tiles at random
        #    The result from MIN should be the average score achieved given the move by MAX.
        #    So, how is beta calculated to allow the alpha-beta pruning algorithm to be implemented?
        # KNOWN:
        #  - MIN should return the average across all possible moves
        #  - MAX can maximise the alpha based on that
        #  - MIN will be called on across several possible moves by MAX
        # IDEA:
        #  - Just set beta to the average?
        #
        acc = 0.0
        for m in moves:
            new_context, new_solution = creat_call_vars(m, context, solution)
            r = minimax_with_ab_pruning(context=new_context, solution=new_solution)
            acc += r * new_solution.move.prob
        avg_score = acc / (len(moves) / 2)
        context.beta = min(context.beta, avg_score)
        return avg_score


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
