import unittest

from Grid_3 import Grid
from PlayerAI_3 import PlayerAI


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
