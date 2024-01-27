import numpy as np
import time
import sys

class Node:
    def __init__(self, level, value, weight, include):
        self.level = level
        self.value = value
        self.weight = weight
        self.include = include

def read_knapsack_input(file_path):
    with open(file_path) as file:
        lines = file.readlines()

    # Read the number of items (n) and knapsack capacity from the first line
    n, capacity = map(int, lines[0].split())

    # Read the values and weights of each item from subsequent lines
    data = [tuple(map(int, line.split())) for line in lines[1:] if line.strip()]

    if len(data) < n:
        raise ValueError("Insufficient data lines in the input file.")

    # Separate values and weights into separate lists
    values, weights = zip(*data)
    return n, capacity, values, weights

def knapsack_branch_and_bound(n, capacity, values, weights):
    def bound(node):
        if node.weight > capacity:
            return 0

        bound_value = node.value
        j = node.level + 1
        total_weight = node.weight

        # Include items until the capacity is exhausted or all items are included
        while j < n and total_weight + weights[j] <= capacity:
            total_weight += weights[j]
            bound_value += values[j]
            j += 1

        # If there are remaining items, add their fractional value to the bound
        if j < n:
            # Linear relaxation: add the fractional part of the next item
            bound_value += (capacity - total_weight) * (values[j] / weights[j])

        return bound_value

    priority_queue = []
    root = Node(level=-1, value=0, weight=0, include=[])

    # Create a list of items sorted by value-to-weight ratio in descending order
    sorted_items = sorted(range(n), key=lambda x: values[x] / weights[x], reverse=True)

    best_node = root
    count = 0
    while root.level < n - 1:
        i = root.level + 1
        index = sorted_items[i]
        left_child = Node(level=i, value=root.value + values[index], weight=root.weight + weights[index], include=root.include + [index])
        right_child = Node(level=i, value=root.value, weight=root.weight, include=root.include)

        # Update the best_node if left_child is feasible and has a higher value
        if left_child.weight <= capacity and left_child.value > best_node.value:
            best_node = left_child

        # Add left and right child nodes to the priority queue if their bound is higher
        if bound(left_child) > best_node.value:
            priority_queue.append(left_child)

        if bound(right_child) > best_node.value:
            priority_queue.append(right_child)

        # If the bound of both children is less than or equal to the current best_node value, increment count
        if max(bound(left_child), bound(right_child)) <= best_node.value:
            count += 1

        # Break the loop after 500 iterations
        if count == 500:
            break

        # Pruning: remove nodes with a lower bound than the current best_node value
        priority_queue = [node for node in priority_queue if bound(node) > best_node.value]

        # If the priority queue is not empty, select the node with the highest bound as the next root
        if priority_queue:
            priority_queue.sort(key=lambda x: bound(x), reverse=True)
            root = priority_queue.pop(0)
        else:
            break

    return best_node.value, best_node.include

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: %s input_file" % sys.argv[0])
        sys.exit()

    file_name = sys.argv[1]
    # Example usage
    start_time = time.time()
    n, capacity, values, weights = read_knapsack_input(file_name)
    best_value, best_selection = knapsack_branch_and_bound(n, capacity, values, weights)
    solution_to_improve = np.zeros(n)
    solution_to_improve[best_selection] = 1
    print(best_value)
    print(" ".join(map(str, map(int, solution_to_improve))))
    end_time = time.time()
    execution_time = end_time - start_time
    #print(f"Execution Time: {execution_time} seconds")
