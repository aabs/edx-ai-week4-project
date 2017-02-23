import cProfile
import unittest
import random

import math

import sys

from FastGrid import FastGrid
from Grid_3 import Grid
from PlayerAI_3 import PlayerAI


class ABTestingBase(unittest.TestCase):
    def setup(self):
        self.profiler = None

    def start_profiling(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def stop_profiling(self):
        self.profiler.disable()

    def display_profiling_summary(self, default_sort_order='ncalls'):
        # sort_col = "tottime"
        # sort_col = "ncalls"
        self.profiler.print_stats(sort=default_sort_order)

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

    def create_slowgrid_from(self, val) -> Grid:
        sut = Grid()
        for row in range(4):
            for col in range(4):
                sut.setCellValue((row, col), val[row][col])
        return sut

    def create_grid_from(self, newboard, size=4) -> FastGrid:
        sut = FastGrid()
        sut.board = newboard
        sut.size = size
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

    def create_player(self) -> PlayerAI:
        return PlayerAI()

    def create_random_grid(self) -> Grid:
        r = Grid()
        tiles = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

        for x in range(4):
            for y in range(4):
                r.setCellValue((x, y), random.choice(tiles))
        return r

    def suppress_output(self):
        sys.stdout = CaptureOutput()

    def allow_output(self):
        sys.stdout = sys.__stdout__
