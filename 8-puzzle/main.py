import heapq
from collections import deque

GOAL_STATE = [1, 2, 3, 4, 5, 6, 7, 8, 0]

def print_board(state):
    for i in range(0, 9, 3):
        print(state[i:i+3])
    print()

def get_neighbors(state):
    neighbors = []
    index = state.index(0)
    row, col = divmod(index, 3)
    moves = {
        'up': -3,
        'down': 3,
        'left': -1,
        'right': 1
    }

    for move, delta in moves.items():
        new_index = index + delta
        if move == 'up' and row == 0: continue
        if move == 'down' and row == 2: continue
        if move == 'left' and col == 0: continue
        if move == 'right' and col == 2: continue

        new_state = state[:]
        new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
        neighbors.append(new_state)
    return neighbors

def manhattan_distance(state):
    distance = 0
    for i, value in enumerate(state):
        if value == 0:
            continue
        goal_pos = GOAL_STATE.index(value)
        current_row, current_col = divmod(i, 3)
        goal_row, goal_col = divmod(goal_pos, 3)
        distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance

def bfs(start):
    queue = deque([[start]])
    visited = set()
    while queue:
        path = queue.popleft()
        state = path[-1]
        if state == GOAL_STATE:
            return path
        state_tuple = tuple(state)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for neighbor in get_neighbors(state):
            if tuple(neighbor) not in visited:
                queue.append(path + [neighbor])
    return None

def astar(start):
    heap = []
    heapq.heappush(heap, (manhattan_distance(start), [start]))
    visited = set()
    while heap:
        _, path = heapq.heappop(heap)
        state = path[-1]
        if state == GOAL_STATE:
            return path
        state_tuple = tuple(state)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        for neighbor in get_neighbors(state):
            if tuple(neighbor) not in visited:
                new_path = path + [neighbor]
                cost = len(new_path) + manhattan_distance(neighbor)
                heapq.heappush(heap, (cost, new_path))
    return None

if __name__ == "__main__":
    initial_state = [1, 2, 3, 4, 0, 5, 6, 7, 8]
    
    print("=== BFS ===")
    bfs_path = bfs(initial_state)
    for step in bfs_path:
        print_board(step)

    print("=== A* ===")
    a_star_path = astar(initial_state)
    for step in a_star_path:
        print_board(step)
