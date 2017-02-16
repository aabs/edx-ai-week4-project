import unittest

from BaseDisplayer_3 import BaseDisplayer
from ComputerAI_3 import ComputerAI
from GameManager_3 import GameManager
from Grid_3 import Grid
from PlayerAI_3 import PlayerAI
import cma


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
        a1 = sut.calculate_monotonicity2(g1)
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


class GameplayTests(unittest.TestCase):
    def test_can_run_game(self):
        sut = GameBuilder().build()
        sut.start()
        self.assertIsNotNone(sut.playerAI.transcript)
        self.assertGreater(len(sut.playerAI.transcript), 0)
        print("moves: ", len(sut.playerAI.transcript))

    def test_optimise_player_weights(self):
        es = cma.CMAEvolutionStrategy(5 * [1.0], 0.5)
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [self.run_solution(x) for x in solutions])
        es.result_pretty()

    def run_solution(self, solution: list) -> int:
        sut = GameBuilder().build()
        sut.playerAI.set_weights(solution[0], solution[1], solution[2], solution[3], solution[4])
        sut.start()
        return sut.grid.getMaxTile()
