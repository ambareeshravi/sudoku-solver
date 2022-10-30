import numpy as np
from tqdm import tqdm

class Sudoku:
    '''
    Pass the input as a 9x9 np array with empty spaces as 0s
    '''
    def __init__(
        self,
        input: np.ndarray,
    ) -> None:
        self._validate_input(input)
        self._grid = input
        self._generate_grid_properties()

    @property
    def grid(self,):
        return self._grid

    def print(self):
        row_div = lambda: print("+" + "-"*9 + "+" + "-"*9 + "+" + "-"*8 + "+")
        for (row_index, row) in enumerate(self.grid):
            row = list(map(str, row))
            if row_index % 3 == 0:
                row_div()
            print(
                "| ", " ".join(row[:3]), " | ", " ".join(row[3:6]), " | ", " ".join(row[6:9]), "|"
            )
        row_div()

    def _generate_grid_properties(self):
        grid_indices = list()
        prev_xi, prev_yi = 0, 0
        for xi in range(3, 10, 3):
            prev_yi = 0
            for yi in range(3, 10, 3):
                grid_indices.append([(prev_xi, xi), (prev_yi, yi)])
                prev_yi = yi
            prev_xi = xi
        self._grid_indices = np.array(grid_indices)

        self._unfit = np.empty(self._grid.shape, dtype=object)
        for i in np.ndindex(self._unfit.shape[0]):
            for j in np.ndindex(self._unfit.shape[1]):
                self._unfit[i][j] = set()
        
        self._tried = self._unfit.copy()

        self._completed_row_indices = list()
        self._completed_col_indices = list()

        self._truth = np.zeros(self._grid.shape)
        for xi in range(0, 9):
            for yi in range(0, 9):
                if self._grid[xi, yi] != 0:
                    self._truth[xi, yi] = 1
        
    def _get_grid_indices_from_pos(self, pos_x, pos_y):
        x = (pos_x // 3) * 3
        y = (pos_y // 3) * 3
        return [(x, x+3), (y, y+3)]

    def _get_grid_values_from_pos(self, pos_x, pos_y):
        (xa,xb), (ya,yb) = self._get_grid_indices_from_pos(pos_x=pos_x, pos_y=pos_y)
        return self._grid[xa:xb, ya:yb]

    def _validate_input(self, input: np.ndarray):
        for i in np.unique(input):
            assert i in np.arange(0, 10), \
                "Value out of bounds - should be between 1 and 9 or 0 for empty"
    
    def _check_unique(self, x: np.ndarray):
        y = x.flatten()
        y.sort()
        return set(np.unique(y)) == set(np.arange(1, 10))

    def _check_val_pos(self, value, pos):
        (pos_x, pos_y) = pos
        (xa, xb), (ya, yb) = self._get_grid_indices_from_pos(*pos)
        return not (
            value in self._grid[:, pos_y] or \
            value in self._grid[pos_x, :] or \
            value in self._grid[xa:xb, ya:yb]
        )

    def _check_row_complete(self, pos):
        pos_x, pos_y = pos
        return self._check_unique(self._grid[:, pos_y])

    def _check_col_complete(self, pos):
        pos_x, pos_y = pos
        return self._check_unique(self._grid[pos_x, :])

    def _check_grid_complete(self, pos):
        grid_vals = self._get_grid_values_from_pos(*pos)
        return self._check_unique(grid_vals)
    
    def _others(self, x):
        l = list(range(1, 10))
        l.remove(x)
        return np.array(set(l))
    
    def _find_missing(self, items):
        return list(set(list(range(1, 10))) - set(items))

    def is_solved(self):
        for row_index in range(self._grid.shape[0]):
            row_solved = self._check_unique(self._grid[row_index, :])
            if not row_solved: return False
        for col_index in range(self._grid.shape[1]):
            col_solved = self._check_unique(self._grid[:, col_index])
            if not col_solved: return False
        for ((xa,xb), (ya,yb)) in self._grid_indices:
            grid_solved = self._check_unique(self._grid[xa:xb, ya:yb])
            if not grid_solved: return False
        return True

    def solve(self):
        for it in range(5000000):
            if self.is_solved():
                print(f"Solved the given Sudoku in {it} iterations! :)")
                return
            
            for xi in range(0, 9):
                for yi in range(0, 9):
                    if self._truth[xi, yi]:
                        continue

                    if self._check_row_complete((xi,yi)) \
                        or self._check_col_complete((xi,yi)) \
                        or self._check_grid_complete((xi, yi)):
                        continue

                    unfit_nums = set(
                        self._grid[:, yi].tolist() + \
                        self._grid[xi, :].tolist() + \
                        self._get_grid_values_from_pos(xi, yi).flatten().tolist()
                    )
                    self._unfit[xi][yi].update(unfit_nums)

                    missing = self._find_missing(unfit_nums)
                    
                    if len(missing) == 1:
                        self._grid[xi, yi] = missing[0]

                    if np.unique(self._truth).tolist() == [1]:
                        return
        
        print(f"Could not solve the given Sudoku! :/")
        
    def __call__(self):
        self.print()
        self.solve()
        self.print()

    @classmethod
    def get_test_input(cls):
        return np.array([
            [5,8,1,6,7,2,4,3,9],
            [7,9,2,8,4,3,6,5,1],
            [3,6,4,5,9,1,7,8,2],
            [4,3,8,9,5,7,2,1,6],
            [2,5,6,1,8,4,9,7,3],
            [1,7,9,3,2,6,8,4,5],
            [8,4,5,2,1,9,3,6,7],
            [9,1,3,7,6,8,5,2,4],
            [6,2,7,4,3,5,1,9,8]
        ])

if __name__ == '__main__':
    for percentage in range(10, 80, 10):
        input = Sudoku.get_test_input()
        n_erase = round(percentage * 9 * 9 / 100)
        indices_to_erase = np.random.choice(np.arange(0,9), size=(n_erase,2))
        print(f"Erasing {percentage}% i.e. {n_erase} values from the original input with indices shape: {indices_to_erase.shape}")
        
        for (x,y) in indices_to_erase:
            input[x][y] = 0
        
        solver = Sudoku(input = input)
        solver()

        np.testing.assert_array_equal(solver.grid, Sudoku.get_test_input())
        print("="*60)
        print()
        print()
        del solver