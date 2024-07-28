import csv
import heapq

import pygame

from OpenGL.GL import *
from OpenGL.GLUT import *

# Constants
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 16
ROWS, COLS = 35, 35

# Colors
WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)
GRAY = (0.8, 0.8, 0.8)
RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
BLUE = (0.0, 0.0, 1.0)
YELLOW = (1.0, 1.0, 0.0)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.OPENGL)
pygame.display.set_caption("Pathfinding Visualizer with OpenGL")

# Initialize OpenGL
glClearColor(WHITE[0], WHITE[1], WHITE[2], 1.0)  # Background color
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
glMatrixMode(GL_MODELVIEW)

def load_maze_from_csv(filename):
    maze = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            row = list(map(int, row))
            maze.append(row)
    return maze

maze = load_maze_from_csv('maze2.csv')

if len(maze) != ROWS or any(len(row) != COLS for row in maze):
    raise ValueError("Maze dimensions do not match specified ROWS and COLS")

start = (1, 1)
end = (33, 33)

def draw_grid():
    glColor3f(*GRAY)
    glBegin(GL_LINES)
    for x in range(0, WIDTH, GRID_SIZE):
        glVertex2f(x, 0)
        glVertex2f(x, HEIGHT)
    for y in range(0, HEIGHT, GRID_SIZE):
        glVertex2f(0, y)
        glVertex2f(WIDTH, y)
    glEnd()

def draw():
    glClear(GL_COLOR_BUFFER_BIT)
    for row in range(ROWS):
        for col in range(COLS):
            if maze[row][col] == 1:
                glColor3f(*BLACK)
            elif (row, col) == start:
                glColor3f(*GREEN)
            elif (row, col) == end:
                glColor3f(*RED)
            else:
                glColor3f(*WHITE)
            glBegin(GL_QUADS)
            glVertex2f(col * GRID_SIZE, row * GRID_SIZE)
            glVertex2f(col * GRID_SIZE + GRID_SIZE, row * GRID_SIZE)
            glVertex2f(col * GRID_SIZE + GRID_SIZE, row * GRID_SIZE + GRID_SIZE)
            glVertex2f(col * GRID_SIZE, row * GRID_SIZE + GRID_SIZE)
            glEnd()
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

        # Draw the current state
        draw()
        draw_traversed_path(traversed_path)
        pygame.display.flip()
        pygame.time.wait(50)

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

        # Draw the current state
        draw()
        draw_traversed_path(traversed_path)
        pygame.display.flip()
        pygame.time.wait(50)

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

    return [], traversed_path

def bfs(start, end):
    queue = [start]
    came_from = {}
    traversed_path = set()
    visited = set()
    visited.add(start)

    while queue:
        current = queue.pop(0)
        traversed_path.add(current)

        # Draw the current state
        draw()
        draw_traversed_path(traversed_path)
        pygame.display.flip()
        pygame.time.wait(50)

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
            came_from[neighbor] = current
            queue.append(neighbor)
            visited.add(neighbor)

    return [], traversed_path

def dfs(start, end):
    stack = [start]
    came_from = {}
    traversed_path = set()
    visited = set()
    visited.add(start)

    while stack:
        current = stack.pop()
        traversed_path.add(current)

        # Draw the current state
        draw()
        draw_traversed_path(traversed_path)
        pygame.display.flip()
        pygame.time.wait(50)

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
            came_from[neighbor] = current
            stack.append(neighbor)
            visited.add(neighbor)

    return [], traversed_path

def draw_traversed_path(traversed_path):
    for current in traversed_path:
        glColor3f(*YELLOW)
        glBegin(GL_QUADS)
        glVertex2f(current[1] * GRID_SIZE, current[0] * GRID_SIZE)
        glVertex2f(current[1] * GRID_SIZE + GRID_SIZE, current[0] * GRID_SIZE)
        glVertex2f(current[1] * GRID_SIZE + GRID_SIZE, current[0] * GRID_SIZE + GRID_SIZE)
        glVertex2f(current[1] * GRID_SIZE, current[0] * GRID_SIZE + GRID_SIZE)
        glEnd()

selected_algorithm = "a_star"

def run_algorithm():
    global selected_algorithm
    if selected_algorithm == "a_star":
        path, traversed_path = a_star(start, end)
    elif selected_algorithm == "djikstra":
        path, traversed_path = dijkstra(start, end)
    elif selected_algorithm == "bfs":
        path, traversed_path = bfs(start, end)
    elif selected_algorithm == "dfs":
        path, traversed_path = dfs(start, end)
    

    # Draw the path
    for row, col in path:
        glColor3f(*BLUE)
        glBegin(GL_QUADS)
        glVertex2f(col * GRID_SIZE, row * GRID_SIZE)
        glVertex2f(col * GRID_SIZE + GRID_SIZE, row * GRID_SIZE)
        glVertex2f(col * GRID_SIZE + GRID_SIZE, row * GRID_SIZE + GRID_SIZE)
        glVertex2f(col * GRID_SIZE, row * GRID_SIZE + GRID_SIZE)
        glEnd()
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                handle_keypress((pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)))
                waiting = False

def handle_keypress(event):
    global selected_algorithm
    if event.key == pygame.K_SPACE:
        run_algorithm()
    elif event.key == pygame.K_1:
        selected_algorithm = "a_star"
        print("selected algorithm =", selected_algorithm)
    elif event.key == pygame.K_2:
        selected_algorithm = "djikstra"
        print("selected algorithm =", selected_algorithm)
    elif event.key == pygame.K_3:
        selected_algorithm = "bfs"
        print("selected algorithm =", selected_algorithm)
    elif event.key == pygame.K_4:
        selected_algorithm = "dfs"
        print("selected algorithm =", selected_algorithm)

def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                handle_keypress(event)

        draw()

    pygame.quit()

if __name__ == "__main__":
    main()
