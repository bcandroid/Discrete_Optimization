import numpy as np

def read_knapsack_input(file_path):
    with open(file_path, 'r') as file:
        n, capacity = map(int, file.readline().split())
        values, weights = zip(*(map(int, line.split()) for line in file.readlines()))
    return n, capacity, values, weights

def knapsack_branch_and_bound(n, capacity, values, weights):
    class Node:
        def __init__(self, level, value, weight, include):
            self.level = level
            self.value = value
            self.weight = weight
            self.include = include

    def bound(node):
        if node.weight > capacity:
            return 0
        bound_value = node.value
        j = node.level + 1
        total_weight = node.weight
        while j < n and total_weight + weights[j] <= capacity:
            total_weight += weights[j]
            bound_value += values[j]
            j += 1
        if j < n:
            bound_value += (capacity - total_weight) * (values[j] / weights[j])
        return bound_value

    priority_queue = []
    root = Node(level=-1, value=0, weight=0, include=[])
    best_node = root

    while root.level < n - 1:
        i = root.level + 1
        left_child = Node(level=i, value=root.value + values[i], weight=root.weight + weights[i], include=root.include + [i])
        right_child = Node(level=i, value=root.value, weight=root.weight, include=root.include)

        if left_child.weight <= capacity and left_child.value > best_node.value:
            best_node = left_child

        if bound(left_child) > best_node.value:
            priority_queue.append(left_child)

        if bound(right_child) > best_node.value:
            priority_queue.append(right_child)

        if priority_queue:
            priority_queue.sort(key=lambda x: bound(x), reverse=True)
            root = priority_queue.pop(0)
        else:
            break

    return best_node.value, best_node.include

# Tabu Search functions
def tabu_search(solution, tabu_list, max_iterations, capacity, weights):
    current_solution = solution.copy()
    best_solution = solution.copy()

    for _ in range(max_iterations):
        neighbors = generate_neighbors(current_solution, capacity, weights)
        best_neighbor = None

        for neighbor in neighbors:
            if not any(np.array_equal(neighbor, tabu) for tabu in tabu_list):
                if best_neighbor is None or evaluate_solution(neighbor, values) > evaluate_solution(best_neighbor, values):
                    best_neighbor = neighbor

        if best_neighbor is not None:
            tabu_list.append(current_solution.copy())

            current_solution = best_neighbor

            if evaluate_solution(current_solution, values) > evaluate_solution(best_solution, values):
                best_solution = current_solution

            if len(tabu_list) > 10:  # Adjust the tabu list size as needed
                tabu_list.pop(0)

    return best_solution

def generate_neighbors(solution, capacity, weights):
    neighbors = []

    # Swap operation
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            neighbor = solution.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)

    # Insert operation
    for i in range(len(solution)):
        for j in range(len(solution)):
            if i != j:
                neighbor = np.delete(solution, i)
                neighbor = np.insert(neighbor, j, solution[i])
                neighbors.append(neighbor)

    # 2-Opt operation
    for i in range(len(solution) - 1):
        for j in range(i + 2, len(solution)):
            neighbor = np.concatenate([solution[:i+1], solution[i+1:j][::-1], solution[j:]])
            neighbors.append(neighbor)

    # 3-Opt operation (Optional, as it can be more complex)
    # ...

    return [n for n in neighbors if np.sum(n * weights) <= capacity]


def evaluate_solution(solution, values):
    total_profit = np.sum(solution * values)
    # Implement your logic to handle constraints, if necessary
    return total_profit
# Example usage:
file_path = 'knapsack_input.txt'
n, capacity, values, weights = read_knapsack_input(file_path)
best_value, best_selection = knapsack_branch_and_bound(n, capacity, values, weights)

# Convert the best selection to a binary array
initial_solution = np.zeros(n)
initial_solution[best_selection] = 1

# Apply Tabu Search to further improve the solution
tabu_list = []
improved_solution = tabu_search(initial_solution, tabu_list, 100, capacity, weights)
final_profit = evaluate_solution(improved_solution, values)

print(int(final_profit))
print(" ".join(map(str, map(int, improved_solution))))
