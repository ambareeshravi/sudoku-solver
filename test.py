from sudoku import Sudoku as SudokuGenerator
from main import Sudoku, np

def to_array(board):
    input_board = np.array(board)
    input_board[input_board == None] = 0
    input_board = input_board.astype(np.uint8)
    return input_board

for difficulty in range(10, 55, 5):
    print(f"Testing with {difficulty}% difficulty")
    difficulty = difficulty/100
    puzzle = SudokuGenerator(3).difficulty(difficulty)
    input = to_array(puzzle.board)
    solution = puzzle.solve()
    output = to_array(solution.board)

    solver = Sudoku(input = input)
    solver()
    np.testing.assert_array_equal(solver.grid, output)
    del puzzle, input, output

    print()
