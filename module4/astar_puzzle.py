from simpleai.search import SearchProblem, astar
from simpleai.search.viewers import ConsoleViewer

GOAL_STATE = ((1, 2, 3),
              (4, 5, 6),
              (7, 8, 0))  # 0 represents the empty tile

class Puzzle8Problem(SearchProblem):
    def __init__(self, initial_state):
        super(Puzzle8Problem, self).__init__(initial_state)

    def actions(self, state):
        actions = []
        row, col = self.find_zero(state)

        if row > 0:
            actions.append('up')
        if row < 2:
            actions.append('down')
        if col > 0:
            actions.append('left')
        if col < 2:
            actions.append('right')
        return actions

    def result(self, state, action):
        row, col = self.find_zero(state)
        new_state = [list(r) for r in state]

        if action == 'up':
            new_row, new_col = row - 1, col
        elif action == 'down':
            new_row, new_col = row + 1, col
        elif action == 'left':
            new_row, new_col = row, col - 1
        elif action == 'right':
            new_row, new_col = row, col + 1

        new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
        return tuple(tuple(r) for r in new_state)

    def is_goal(self, state):
        return state == GOAL_STATE

    def heuristic(self, state):
        # Manhattan distance
        distance = 0
        for r in range(3):
            for c in range(3):
                value = state[r][c]
                if value != 0:
                    goal_r, goal_c = divmod(value - 1, 3)
                    distance += abs(goal_r - r) + abs(goal_c - c)
        return distance

    def find_zero(self, state):
        for r in range(3):
            for c in range(3):
                if state[r][c] == 0:
                    return r, c

def get_initial_state():
    print("Enter your 8-puzzle starting state (use 0 for empty tile).")
    print("Example: Enter '2 8 3' for the first row.")
    state = []
    for i in range(3):
        while True:
            try:
                row = list(map(int, input(f"Enter row {i+1} (3 numbers separated by spaces): ").split()))
                if len(row) == 3 and all(0 <= num <= 8 for num in row):
                    state.append(tuple(row))
                    break
                else:
                    print("Invalid input. Please enter exactly 3 numbers between 0 and 8.")
            except ValueError:
                print("Invalid input. Please enter numbers only.")
    return tuple(state)

if __name__ == '__main__':
    initial_state = get_initial_state()
    problem = Puzzle8Problem(initial_state)
    viewer = ConsoleViewer()
    result = astar(problem, graph_search=True, viewer=viewer)

    print("\nSolution found!")
    for action, state in result.path():
        print("Move:", action)
        for row in state:
            print(row)
        print()

