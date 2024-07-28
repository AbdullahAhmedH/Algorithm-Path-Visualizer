import csv
import heapq
from collections import deque

import pygame

# Constants
WIDTH, HEIGHT = 600, 600  # 480, 480 #
GRID_SIZE = 16         #20 
ROWS, COLS = 35, 35  #23, 23

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pathfinding Visualizer")

def load_maze_from_csv(filename):
    maze = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            maze.append(list(map(int, row)))
    return maze

maze = load_maze_from_csv('maze2.csv')

if len(maze) != ROWS or any(len(row) != COLS for row in maze):
    raise ValueError("Maze dimensions do not match specified ROWS and COLS")

start = (1, 1) #1,1
end = (33, 33) #27,1
selected_algorithm = "A*"

def draw_grid():
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y))

def draw():
    screen.fill(WHITE)
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE
            if maze[row][col] == 1:
                color = BLACK
            elif (row, col) == start:
                color = GREEN
            elif (row, col) == end:
                color = RED
            pygame.draw.rect(screen, color, (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    draw_grid()
    pygame.display.flip()

def get_neighbors(row, col):
    neighbors = []
    if row > 0:
        neighbors.append((row-1, col))
    if row < ROWS - 1:
        neighbors.append((row+1, col))
    if col > 0:
        neighbors.append((row, col-1))
    if col < COLS - 1:
        neighbors.append((row, col+1))
    return neighbors

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, end):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {row: {col: float('inf') for col in range(COLS)} for row in range(ROWS)}
    g_score[start[0]][start[1]] = 0
    f_score = {row: {col: float('inf') for col in range(COLS)} for row in range(ROWS)}
    f_score[start[0]][start[1]] = heuristic(start, end)

    open_set_hash = {start}
    traversed_path = set()

    while open_set:
        current = heapq.heappop(open_set)[1]
        open_set_hash.remove(current)
        traversed_path.add(current)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, traversed_path

        for neighbor in get_neighbors(*current):
            if maze[neighbor[0]][neighbor[1]] == 1:
                continue
            tentative_g_score = g_score[current[0]][current[1]] + 1
            if tentative_g_score < g_score[neighbor[0]][neighbor[1]]:
                came_from[neighbor] = current
                g_score[neighbor[0]][neighbor[1]] = tentative_g_score
                f_score[neighbor[0]][neighbor[1]] = tentative_g_score + heuristic(neighbor, end)
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor[0]][neighbor[1]], neighbor))
                    open_set_hash.add(neighbor)

        # Draw the traversed cells without flickering
        pygame.draw.rect(screen, YELLOW, (current[1] * GRID_SIZE, current[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        draw_grid()
        pygame.display.flip()
        pygame.time.wait(50)

    return [], traversed_path

def dijkstra(start, end):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {row: {col: float('inf') for col in range(COLS)} for row in range(ROWS)}
    g_score[start[0]][start[1]] = 0

    open_set_hash = {start}
    traversed_path = set()

    while open_set:
        current = heapq.heappop(open_set)[1]
        open_set_hash.remove(current)
        traversed_path.add(current)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, traversed_path

        for neighbor in get_neighbors(*current):
            if maze[neighbor[0]][neighbor[1]] == 1:
                continue
            tentative_g_score = g_score[current[0]][current[1]] + 1
            if tentative_g_score < g_score[neighbor[0]][neighbor[1]]:
                came_from[neighbor] = current
                g_score[neighbor[0]][neighbor[1]] = tentative_g_score
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (g_score[neighbor[0]][neighbor[1]], neighbor))
                    open_set_hash.add(neighbor)

        # Draw the traversed cells without flickering
        pygame.draw.rect(screen, YELLOW, (current[1] * GRID_SIZE, current[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        draw_grid()
        pygame.display.flip()
        pygame.time.wait(50)

    return [], traversed_path

def bfs(start, end):
    queue = deque([start])
    came_from = {}
    visited = {start}
    traversed_path = set()

    while queue:
        current = queue.popleft()
        traversed_path.add(current)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, traversed_path

        for neighbor in get_neighbors(*current):
            if maze[neighbor[0]][neighbor[1]] == 1 or neighbor in visited:
                continue
            visited.add(neighbor)
            came_from[neighbor] = current
            queue.append(neighbor)

        # Draw the traversed cells without flickering
        pygame.draw.rect(screen, YELLOW, (current[1] * GRID_SIZE, current[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        draw_grid()
        pygame.display.flip()
        pygame.time.wait(50)

    return [], traversed_path

def dfs(start, end):
    stack = [start]
    came_from = {}
    visited = {start}
    traversed_path = set()

    while stack:
        current = stack.pop()
        traversed_path.add(current)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, traversed_path

        for neighbor in get_neighbors(*current):
            if maze[neighbor[0]][neighbor[1]] == 1 or neighbor in visited:
                continue
            visited.add(neighbor)
            came_from[neighbor] = current
            stack.append(neighbor)

        # Draw the traversed cells without flickering
        pygame.draw.rect(screen, YELLOW, (current[1] * GRID_SIZE, current[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        draw_grid()
        pygame.display.flip()
        pygame.time.wait(50)

    return [], traversed_path

def main():
    global start, end, selected_algorithm
    running = True
    path = []
    traversed_path = set()

    while running:
        draw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if selected_algorithm == "A*":
                        path, traversed_path = a_star(start, end)
                    elif selected_algorithm == "Dijkstra":
                        path, traversed_path = dijkstra(start, end)
                    elif selected_algorithm == "BFS":
                        path, traversed_path = bfs(start, end)
                    elif selected_algorithm == "DFS":
                        path, traversed_path = dfs(start, end)
                elif event.key == pygame.K_1:
                    selected_algorithm = "A*"
                elif event.key == pygame.K_2:
                    selected_algorithm = "Dijkstra"
                elif event.key == pygame.K_3:
                    selected_algorithm = "BFS"
                elif event.key == pygame.K_4:
                    selected_algorithm = "DFS"

        if traversed_path:
            for cell in traversed_path:
                pygame.draw.rect(screen, YELLOW, (cell[1] * GRID_SIZE, cell[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            draw_grid()
            pygame.display.flip()

        if path:
            for cell in path:
                pygame.draw.rect(screen, BLUE, (cell[1] * GRID_SIZE, cell[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
