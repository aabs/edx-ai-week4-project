from array import array

from Util import Util

directionVectors = (UP_VEC, DOWN_VEC, LEFT_VEC, RIGHT_VEC) = ((-1, 0), (1, 0), (0, -1), (0, 1))
vecIndex = [UP, DOWN, LEFT, RIGHT] = range(4)


class FastGrid:
    def __init__(self, g=None):
        self.board = Util.slowgrid_to_array(g) if g is not None else None
        self.hashcode = None
        self.score = None
        self.size = g.size if g is not None else 4

    # Make a Deep Copy of This Object
    def clone(self):
        result = FastGrid()
        result.board = array('i',self.board)
        result.size = self.size
        return result

    def __getitem__(self, c):
        x, y = c
        width = self.size
        if x < 0 or x >= width or y < 0 or y >= width:
            return None
        return self.board[(y * width) + x]

    def __setitem__(self, c, v):
        x, y = c
        if  x < 0 or x >= self.size or y < 0 or y >= self.size:
            return None
        self.board[(y * self.size) + x] = v

    # Insert a Tile in an Empty Cell
    def insertTile(self, pos, value):
        self[pos] = value

    def setCellValue(self, pos, value):
        self[pos] = value

    @property
    def moves(self):
        return self.get_available_moves()

    # Return All Available Moves
    def get_available_moves(self):
        result = []
        for i in vecIndex:
            if self.canMoveWith(directionVectors[i]):
                result.append(i)
        return result

    def canMoveWith(self, vec):
        m, n = vec
        width = self.size
        for x in range(width):
            for y in range(width):
                # If Current Cell is Filled
                valxy = self[x, y]
                if valxy:
                    valmn = self[x + n, y + m]

                    # If Value is the Same or Adjacent Cell is Empty
                    if valmn == valxy or valmn == 0:
                        return True

        return False

    # Return All the Empty c\Cells
    def get_available_cells(self):
        cells = []

        for x in range(self.size):
            for y in range(self.size):
                if self[x, y] == 0:
                    cells.append((x, y))

        return cells

    # Return the Tile with Maximum Value
    def getMaxTile(self):
        return max(self.board)

    # Check If Able to Insert a Tile in Position
    def canInsert(self, pos):
        return self[pos] == 0

    # Move the Grid
    def move(self, dir: int):
        result = self.clone()
        success = False
        if dir == UP:
            success = result.moveUD(False)
        elif dir == DOWN:
            success = result.moveUD(True)
        elif dir == LEFT:
            success = result.moveLR(False)
        elif dir == RIGHT:
            success = result.moveLR(True)
        return success, result

    # Move Up or Down
    def moveUD(self, down):
        r = range(self.size - 1, -1, -1) if down else range(self.size)

        moved = False

        for x in range(self.size):
            cells = []

            for y in r:
                cell = self[x, y]

                if cell != 0:
                    cells.append(cell)

            self.merge(cells)

            for y in r:
                value = cells.pop(0) if cells else 0

                if self[x, y] != value:
                    moved = True

                self[x, y] = value

        return moved

    # move left or right
    def moveLR(self, right):
        r = range(self.size - 1, -1, -1) if right else range(self.size)

        moved = False

        for y in range(self.size):
            cells = []

            for x in r:
                cell = self[x, y]

                if cell != 0:
                    cells.append(cell)

            self.merge(cells)

            for x in r:
                value = cells.pop(0) if cells else 0

                if self[x, y] != value:
                    moved = True

                self[x, y] = value

        return moved

    # Merge Tiles
    def merge(self, cells):
        if len(cells) <= 1:
            return cells

        i = 0

        while i < len(cells) - 1:
            if cells[i] == cells[i + 1]:
                cells[i] *= 2

                del cells[i + 1]

            i += 1

    def canMove(self):
        for x in range(self.size):
            for y in range(self.size):

                val = self[x, y]
                if val == 0:
                    return True

                for m,n in directionVectors:

                    # If Current Cell is Filled
                    if val != 0:
                        adj_cell_value = self[x + n, y + m]

                        # If Value is the Same or Adjacent Cell is Empty
                        if adj_cell_value is not None and (adj_cell_value == val or adj_cell_value == 0):
                            return True
        return False


    def crossBound(self, pos):
        return pos[0] < 0 or pos[0] >= self.size or pos[1] < 0 or pos[1] >= self.size
