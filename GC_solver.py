import numpy as np
from collections import defaultdict
import random
import time
import sys
# Function to order nodes by degree in descending order
def degree_heuristic_ordering(adjacency_matrix):
    # Sort nodes based on their degrees in descending order
    return sorted(range(len(adjacency_matrix)), key=lambda x: np.sum(adjacency_matrix[x]), reverse=True)

# Function to check if assigning 'color' to 'node' is safe
def is_safe(node, color, adjacency_matrix, coloring):
    # Check if the assigned color conflicts with neighboring nodes
    for neighbor in range(len(adjacency_matrix)):
        if adjacency_matrix[node][neighbor] and coloring[neighbor] == color:
            return False
    return True

# Depth-First Search (DFS) based coloring algorithm
def dfs_coloring(adjacency_matrix, m, coloring, node_order):
    if not node_order:
        return True

    node = node_order.pop(0)

    for color in range(1, m+1):
        if is_safe(node, color, adjacency_matrix, coloring):
            coloring[node] = color
            # Recursive call to explore further coloring possibilities
            if dfs_coloring(adjacency_matrix, m, coloring, node_order):
                return True
            coloring[node] = 0  # Backtrack if no color is valid

    return False

# Function to find a valid graph coloring using DFS-based algorithm
def graph_coloring(adjacency_matrix):
    num_nodes = len(adjacency_matrix)
    max_colors = num_nodes  # Theoretical maximum number of colors
    coloring = np.zeros(num_nodes, dtype=int)
    node_order = degree_heuristic_ordering(adjacency_matrix)

    # Increase the number of colors until a valid coloring is found
    while not dfs_coloring(adjacency_matrix, max_colors, coloring, node_order.copy()):
        max_colors += 1

    return coloring

# Function to evaluate the number of conflicts in a coloring
def evaluate_coloring(adjacency_matrix, coloring):
    conflicts = 0
    num_nodes = len(adjacency_matrix)

    # Check conflicts by comparing colors of neighboring nodes
    for node in range(num_nodes):
        for neighbor in range(num_nodes):
            if adjacency_matrix[node][neighbor] and coloring[neighbor] == coloring[node]:
                conflicts += 1

    return conflicts

# Local search algorithm to refine the graph coloring
def local_search(adjacency_matrix, initial_coloring, max_iterations=1000):
    current_coloring = initial_coloring.copy()
    current_conflicts = evaluate_coloring(adjacency_matrix, current_coloring)

    for _ in range(max_iterations):
        # Randomly swap colors between two nodes
        node1, node2 = random.sample(range(len(adjacency_matrix)), 2)
        current_coloring[node1], current_coloring[node2] = current_coloring[node2], current_coloring[node1]
        
        new_conflicts = evaluate_coloring(adjacency_matrix, current_coloring)

        # Accept the swap if it reduces conflicts
        if new_conflicts < current_conflicts:
            current_conflicts = new_conflicts
        else:
            # Revert the swap if it doesn't lead to improvement
            current_coloring[node1], current_coloring[node2] = current_coloring[node2], current_coloring[node1]

    chromatic_number = max(current_coloring) + 1
    return chromatic_number, current_coloring

# Main execution
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: %s input_file" % sys.argv[0])
        sys.exit()

    file_name = sys.argv[1]
    start_time_1 = time.time()
    
    # Read input from input.txt
    with open(file_name) as file:
        lines = file.readlines()
    
    n, e = map(int, lines[0].split())
    adjacency_matrix = [[0] * n for _ in range(n)]
    
    # Populate the adjacency matrix based on input edges
    for i in range(1, e + 1):
        u, v = map(int, lines[i].split())
        adjacency_matrix[u][v] = 1
        adjacency_matrix[v][u] = 1
    
    # Initial graph coloring using degree-based heuristic
    coloring = graph_coloring(adjacency_matrix)
    chromatic_number = max(coloring)
    coloring = coloring - 1  # Adjust colors to start from 0
    
    # Local Search to refine the graph coloring
    local_chromatic_number, refined_coloring = local_search(adjacency_matrix, coloring)
    
    # Print results
    print(local_chromatic_number)
    print(' '.join(map(str, refined_coloring)))
    
    end_time_1 = time.time()
    execution_time_1 = end_time_1 - start_time_1
    # print(f"Execution Time: {execution_time_1} seconds")
