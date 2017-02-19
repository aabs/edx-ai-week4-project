import cProfile
import unittest

import time
from encodings.punycode import selective_find

from BaseDisplayer_3 import BaseDisplayer
from ComputerAI_3 import ComputerAI
from Displayer_3 import Displayer
from GameManager_3 import GameManager
from Grid_3 import Grid
from PlayerAI_3 import PlayerAI
import cma
import collections
import sys

directions = [UP, DOWN, LEFT, RIGHT] = range(4)


class Player3Tests(unittest.TestCase):
    def test_can_create_player_a_i(self):
        sut = PlayerAI()
        self.assertIsNotNone(sut)

    def test_can_create_grid_to_design(self):
        sut = Grid()
        self.assertIsNotNone(sut)
        self.assertEqual(4, sut.size)
        sut.setCellValue((0, 0), 2)
        self.assertEqual(sut.getCellValue((0, 0)), 2)

    def test_can_compute_any_monotonicity_score(self):
        g = self.create_monotonic_grid()
        sut = self.create_player()
        actual = sut.calculate_monotonicity2(g)
        self.assertIsNotNone(actual)

    def test_signs_are_right(self):
        g1 = self.create_monotonic_grid()
        g2 = self.create_anti_monotonic_grid()
        sut = self.create_player()
        a1 = sut.calculate_monotonicity2(g1)
        a2 = sut.calculate_monotonicity2(g2)
        self.assertLess(a2, a1)

    def test_monotonic_grids_should_have_positive_score(self):
        g1 = self.create_monotonic_grid()
        sut = self.create_player()
        a1 = sut.calculate_monotonicity2(g1)
        self.assertLess(0, a1)

    def test_anti_monotonic_grids_should_have_negative_score(self):
        g1 = self.create_anti_monotonic_grid()
        sut = self.create_player()
        a1 = sut.utility2(g1)
        self.assertGreater(0, a1)

    def test_uniform_grid_should_have_perfect_score(self):
        g1 = self.create_smooth_grid()
        sut = self.create_player()
        a1 = sut.calculate_smoothness(g1)
        self.assertEqual(a1, 1.0)

    def test_empty_grid_smoothness_is_zero(self):
        g1 = self.create_empty_grid()
        sut = self.create_player()
        a1 = sut.calculate_smoothness(g1)
        self.assertEqual(a1, 0)

    def create_player(self) -> PlayerAI:
        return PlayerAI()

    def create_empty_grid(self) -> Grid:
        return self.create_grid(0)

    def create_smooth_grid(self) -> Grid:
        return self.create_grid(2)

    def create_grid(self, val) -> Grid:
        sut = Grid()
        s = 4
        for x in range(s):
            for y in range(s):
                sut.setCellValue((x, y), val)
        return sut

    def create_grid_from(self, val) -> Grid:
        sut = Grid()
        for row in range(4):
            for col in range(4):
                sut.setCellValue((row, col), val[row][col])
        return sut

    def create_anti_monotonic_grid(self) -> Grid:
        sut = Grid()
        s = 4
        for x in range(s):
            for y in range(s):
                v = pow(2, ((s - 1 - x) + (s - 1 - y)))
                sut.setCellValue((x, y), v if v > 1 else 0)
        return sut

    def create_monotonic_grid(self) -> Grid:
        sut = Grid()
        s = 4
        for x in range(s):
            for y in range(s):
                v = pow(2, (x + y))
                sut.setCellValue((x, y), v if v > 1 else 0)
        return sut

    def test_pairs(self):
        g = self.create_grid(2)
        sut = self.create_player()
        len = sum(1 for _ in sut.generate_all_pairs(g))
        self.assertEqual(4 * 9, len)

    def test_kernel(self):
        p = PlayerAI()
        sut = p.compute_kernel()
        self.assertIsNotNone(sut)

    def test_kernel_2(self):
        p = PlayerAI()
        l = p.compute_kernel(create_snake=True, ramp_amplification=1.5)
        self.assertIsNotNone(l)
        g = p.list_to_grid(l)
        self.assertEqual(g[0][3] + 1, g[1][3])

    def test_against_dumb_move_1(self):
        """Take from a real gameplay scenarios where a fatal wrong move was made:
            Computer's turn:

              256      32     128      2

               4       16      16      64

               32      4       2       2

               4       0       4       4

            Player's Turn:DOWN  (should have been RIGHT)

              256      0      128      2

               4       32      16      64

               32      16      2       2

               4       4       4       4


            RIGHT: [
                [256, 32, 128, 2],
                [0,   4,  32,  64],
                [0,   32, 4,   4],
                [0,   0,  4,   8]
                ]
            """
        p = PlayerAI()
        gstart = self.create_grid_from([[256, 32, 128, 2],
                                        [4, 16, 16, 64],
                                        [32, 4, 2, 2],
                                        [4, 0, 4, 4]])
        gDOWN = self.create_grid_from([[256, 0, 128, 2],
                                       [4, 32, 16, 64],
                                       [32, 16, 2, 2],
                                       [4, 4, 4, 4]])
        ustart = p.utility6(gstart)
        udown = p.utility6(gDOWN)
        self.assertGreater(ustart, udown)
        gright = gstart.clone()
        gright.move(RIGHT)
        uright = p.utility6(gright)
        self.assertGreater(uright, udown)
        available_moves = gstart.getAvailableMoves()
        self.assertNotEquals(1, len(available_moves))
        suggestedMove = p.getMove(gstart)
        self.assertNotEquals(suggestedMove, DOWN)

    def test_against_dumb_move_2(self):
        """Take from a real gameplay scenarios where a fatal wrong move was made:

Computer's turn:

   32  16  32 2
   4   16  4  2
   4   2   8  2
   2   4   8  2

Player's Turn:DOWN (should have been UP)

   0   0   0   0
   32  32  32  0
   8   2   4   4
   2   4   16  4

UP would have been:
   32  32  32  4
   8   2   4   4
   2   4   16  0
   0   0   0   0
                                """
        p = PlayerAI()
        gstart = self.create_grid_from([[32, 16, 32, 2],
                                        [4, 16, 4, 2],
                                        [4, 2, 8, 2],
                                        [2, 4, 8, 2]])
        gdown = gstart.clone()
        gdown.move(DOWN)
        gup = gstart.clone()
        gup.move(UP)
        ustart = p.utility6(gstart)
        udown = p.utility6(gdown)
        uup = p.utility6(gup)
        self.assertGreater(ustart, udown)
        self.assertGreater(uup, udown, "the UP move should have a higher score than the DOWN move")
        available_moves = gstart.getAvailableMoves()
        self.assertNotEquals(1, len(available_moves), "there are two options (UP or DOWN)")
        pr = cProfile.Profile()
        pr.enable()
        chosen_move = p.getMove(gstart)
        pr.disable()
        pr.print_stats(sort="tottime")
        self.assertNotEquals(chosen_move, DOWN, "down is the inferior move, and should not have been chosen")
        # clue: when run at full speed it chooses DOWN, but when slowed down it picks UP

    def test_weights_kernel_is_symetrical(self):
        p = PlayerAI()
        sut = p.compute_kernel()
        self.assertAlmostEqual(sut[0] + sut[15], 0.0, 4)

    def test_can_compute_score(self):

        g1 = self.create_smooth_grid()
        sut = self.create_player()
        a1 = sut.utility4(g1)
        self.assertEqual(a1, 1.0)


class GameBuilder:
    def __init__(self):
        self.grid = Grid()
        self.game_manager = GameManager()
        self.playerAI = PlayerAI()
        self.computerAI = ComputerAI()
        self.displayer = BaseDisplayer()

    def build(self) -> GameManager:
        self.game_manager.setDisplayer(self.displayer)
        self.game_manager.setPlayerAI(self.playerAI)
        self.game_manager.setComputerAI(self.computerAI)
        return self.game_manager

    def with_displayer(self, displayer: BaseDisplayer):
        self.displayer = displayer
        return self


class GameplayTests(unittest.TestCase):
    def test_can_run_game(self):
        sut = GameBuilder().with_displayer(Displayer()).build()
        sut.start()
        # self.assertIsNotNone(sut.playerAI.transcript)
        # self.assertGreater(len(sut.playerAI.transcript), 0)
        # print("moves: ", len(sut.playerAI.transcript))

    def test_optimise_player_weights(self):
        # options = {'CMA_diagonal': 100, 'seed': 1234, 'verb_time': 0}
        # res = cma.fmin(self.run_solution, [1.0] * 4, 1.0, options)
        # print(res)
        es = cma.CMAEvolutionStrategy([1.0, 1.0, -1.0, 1.0], 2.0)
        while not es.stop():
            solutions = es.ask()
            sys.stdout = CaptureOutput()
            results = [4096 - self.run_solution(x) for x in solutions]
            sys.stdout = sys.__stdout__
            print("results: ", results)
            es.tell(solutions, results)
        es.result_pretty()

    def run_solution(self, solution: list) -> int:
        sut = GameBuilder().build()
        sut.playerAI.set_weights(solution[0], solution[1], solution[2], solution[3])
        sut.start()
        return sut.grid.getMaxTile()


class CaptureOutput:
    def write(self, message):
        pass
