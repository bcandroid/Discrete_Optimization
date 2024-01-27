
def color_graph(adjacency_matrix):
    num_nodes = len(adjacency_matrix)
    colors = [-1] * num_nodes
    available_colors = list(range(num_nodes))

    for node in range(num_nodes):
        for neighbor in range(num_nodes):
            if adjacency_matrix[node][neighbor] and colors[neighbor] != -1:
                available_colors[colors[neighbor]] = -1

        for c in available_colors:
            if c != -1:
                colors[node] = c
                break

        available_colors = list(range(num_nodes))

    chromatic_number = max(colors) + 1
    return chromatic_number, colors

# Read input from input.txt
with open('input.txt', 'r') as input_file:
    lines = input_file.readlines()

n, e = map(int, lines[0].split())
adjacency_matrix = [[0] * n for _ in range(n)]

for i in range(1, e + 1):
    u, v = map(int, lines[i].split())
    adjacency_matrix[u][v] = 1
    adjacency_matrix[v][u] = 1

import random

def evaluate_coloring(adjacency_matrix, coloring):
    conflicts = 0
    num_nodes = len(adjacency_matrix)

    for node in range(num_nodes):
        for neighbor in range(num_nodes):
            if adjacency_matrix[node][neighbor] and coloring[neighbor] == coloring[node]:
                conflicts += 1

    return conflicts

def tabu_search(adjacency_matrix, initial_coloring, tabu_size=10, max_iterations=1000):
    current_coloring = initial_coloring.copy()
    current_conflicts = evaluate_coloring(adjacency_matrix, current_coloring)
    tabu_list = []

    for _ in range(max_iterations):
        best_neighbor_colorings = []

        for node in range(len(adjacency_matrix)):
            for color in range(max(current_coloring) + 2):
                if color != current_coloring[node]:
                    neighbor_coloring = current_coloring.copy()
                    neighbor_coloring[node] = color
                    conflicts = evaluate_coloring(adjacency_matrix, neighbor_coloring)
                    best_neighbor_colorings.append((conflicts, neighbor_coloring))

        best_neighbor_colorings.sort(key=lambda x: x[0])
        best_conflicts, best_neighbor = best_neighbor_colorings[0]

        if best_conflicts < current_conflicts and best_neighbor not in tabu_list:
            current_coloring = best_neighbor
            current_conflicts = best_conflicts
            tabu_list.append(best_neighbor)
            
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

    chromatic_number = max(current_coloring) + 1
    return chromatic_number, current_coloring

def local_search(adjacency_matrix, initial_coloring, max_iterations=1000):
    current_coloring = initial_coloring.copy()
    current_conflicts = evaluate_coloring(adjacency_matrix, current_coloring)

    for _ in range(max_iterations):
        node1, node2 = random.sample(range(len(adjacency_matrix)), 2)
        current_coloring[node1], current_coloring[node2] = current_coloring[node2], current_coloring[node1]
        
        new_conflicts = evaluate_coloring(adjacency_matrix, current_coloring)

        if new_conflicts < current_conflicts:
            current_conflicts = new_conflicts
        else:
            # Revert the swap if it doesn't lead to improvement
            current_coloring[node1], current_coloring[node2] = current_coloring[node2], current_coloring[node1]

    chromatic_number = max(current_coloring) + 1
    return chromatic_number, current_coloring

# Read input from input.txt
with open('input.txt', 'r') as input_file:
    lines = input_file.readlines()

n, e = map(int, lines[0].split())
adjacency_matrix = [[0] * n for _ in range(n)]

for i in range(1, e + 1):
    u, v = map(int, lines[i].split())
    adjacency_matrix[u][v] = 1
    adjacency_matrix[v][u] = 1


def is_safe(node, color, graph, color_assignment):
    for neighbor in range(len(graph)):
        if graph[node][neighbor] == 1 and color_assignment[neighbor] == color:
            return False
    return True

def bound(node, graph, color_assignment):
    max_color = max(color_assignment)
    for neighbor in range(len(graph)):
        if graph[node][neighbor] == 1 and color_assignment[neighbor] != -1:
            max_color = max(max_color, color_assignment[neighbor])
    return max_color

def branch_and_bound_coloring(graph, current_coloring, node, chromatic_number, best_solution):
    if node == len(graph):
        actual_chromatic_number = max(current_coloring) + 1
        if actual_chromatic_number < best_solution[0]:
            best_solution[0] = actual_chromatic_number
            best_solution[1] = current_coloring.copy()
        return

    for color in range(chromatic_number + 2):
        if is_safe(node, color, graph, current_coloring):
            current_coloring[node] = color
            next_chromatic_number = max(chromatic_number, color)
            
            # Bound step to prune unnecessary branches
            if bound(node, graph, current_coloring) < best_solution[0]:
                branch_and_bound_coloring(graph, current_coloring, node + 1, next_chromatic_number, best_solution)
            
            current_coloring[node] = -1  # Backtrack

    # Reset the current_coloring[node] after trying all colors
    current_coloring[node] = -1


def branch_and_bound_color_graph(graph):
    num_nodes = len(graph)
    best_solution = [float('inf'), []]  # [chromatic_number, coloring]
    current_coloring = [-1] * num_nodes

    branch_and_bound_coloring(graph, current_coloring, 0, 0, best_solution)

    return best_solution[0], best_solution[1]





# Greedy Initial Coloring
initial_chromatic_number, initial_coloring = color_graph(adjacency_matrix)
print("Initial Chromatic Number:", initial_chromatic_number)
print("Initial Coloring:", ' '.join(map(str, initial_coloring)))

# Branch and Bound
branch_and_bound_chromatic_number, branch_and_bound_coloring = branch_and_bound_color_graph(adjacency_matrix)
print("Branch and Bound Chromatic Number:", branch_and_bound_chromatic_number)
print("Branch and Bound Coloring:", ' '.join(map(str, branch_and_bound_coloring)))

# Tabu Search
tabu_chromatic_number, tabu_search_coloring = tabu_search(adjacency_matrix, branch_and_bound_coloring)
print("Tabu Search Chromatic Number:", tabu_chromatic_number)
print("Tabu Search Coloring:", ' '.join(map(str, tabu_search_coloring)))

# Local Search
local_chromatic_number, refined_coloring = local_search(adjacency_matrix, tabu_search_coloring)
print("Final Refined Chromatic Number:", local_chromatic_number)
print("Final Refined Coloring:", ' '.join(map(str, refined_coloring)))

# Print the results
print("Chromatic Number:", max(refined_coloring) + 1)
print("Coloring:", ' '.join(map(str, refined_coloring)))
