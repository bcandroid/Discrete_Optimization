import itertools
import math
import random


# Function to calculate the Euclidean distance between two points
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((abs(x1 - x2))**2 + (abs(y1 - y2))**2)

# Function to find the nearest neighbor for the next customer
def find_nearest_neighbor(unvisited_customers, current_customer, locations):
    min_distance = float('inf')
    nearest_neighbor = None

    for customer in unvisited_customers:
        distance = euclidean_distance(locations[current_customer][1], locations[current_customer][2],
                                      locations[customer][1], locations[customer][2])
        if distance < min_distance:
            min_distance = distance
            nearest_neighbor = customer

    return nearest_neighbor, min_distance

def initial_solution(customers, num_vehicles, capacity):
    num_customers = len(customers)
    unvisited_customers = list(range(1, num_customers))
    routes = []

    for _ in range(num_vehicles):
        current_capacity = 0
        current_customer = 0  # Start at the warehouse
        route = [current_customer]

        while unvisited_customers:
            nearest_customer, distance = find_nearest_neighbor(unvisited_customers, current_customer, customers)
            if current_capacity + customers[nearest_customer][0] <= capacity:
                current_customer = nearest_customer
                current_capacity += customers[current_customer][0]
                route.append(current_customer)
                unvisited_customers.remove(current_customer)
            else:
                break  # No more capacity, end the route

        route.append(0)  # Return to the warehouse
        routes.append(route)

    return routes


# Function to calculate the total distance of a route
def calculate_route_distance(route, customers):
    return sum(euclidean_distance(customers[route[i]][1], customers[route[i]][2],
                                  customers[route[i + 1]][1], customers[route[i + 1]][2])
               for i in range(len(route) - 1))

# Tabu Search implementation
# Tabu Search implementation
def tabu_search(initial_solution, customers, num_vehicles, capacity, num_iterations):
    best_solution = initial_solution.copy()
    best_distance = sum(calculate_route_distance(route, customers) for route in best_solution)

    tabu_list = {(i, j): 0 for i in range(1, num_vehicles + 1) for j in range(i + 1, num_vehicles + 1)}

    for iteration in range(num_iterations):
        for i in range(1, num_vehicles + 1):
            for j in range(i + 1, num_vehicles + 1):
                new_solution = best_solution.copy()
                new_solution[i - 1], new_solution[j - 1] = new_solution[j - 1], new_solution[i - 1]

                new_distance = sum(calculate_route_distance(route, customers) for route in new_solution)

                if new_distance < best_distance and tabu_list[(i, j)] <= iteration:
                    best_solution = new_solution.copy()
                    best_distance = new_distance

                    # Set tabu tenure for the swapped vehicles
                    tabu_list[(i, j)] = iteration + 10

    return best_solution, best_distance


def local_search(routes, customers, num_vehicles, capacity, num_iterations):
    best_solution = routes.copy()
    best_distance = sum(calculate_route_distance(route, customers) for route in best_solution)

    for iteration in range(num_iterations):
        for i in range(num_vehicles):
            for j in range(1, len(routes[i]) - 1):
                for k in range(j + 1, len(routes[i]) - 1):
                    new_solution = routes.copy()
                    new_solution[i][j], new_solution[i][k] = new_solution[i][k], new_solution[i][j]

                    new_distance = sum(calculate_route_distance(route, customers) for route in new_solution)

                    if new_distance < best_distance:
                        best_solution = new_solution.copy()
                        best_distance = new_distance

    return best_solution, best_distance

# Read input from input.txt
with open('vrp_input.txt', 'r') as input_file:
    lines = input_file.readlines()

num_customers, num_vehicles, capacity = map(int, lines[0].split())
customers = []

for i in range(1, num_customers + 1):
    d, x, y = map(float, lines[i].split())
    customers.append((d, x, y))


def branch_and_bound_vrp(customers, num_vehicles, capacity):
    best_solution = initial_solution(customers, num_vehicles, capacity)
    best_distance = sum(calculate_route_distance(route, customers) for route in best_solution)

    def explore_routes(current_routes, remaining_customers):
        nonlocal best_solution, best_distance

        if not remaining_customers:
            current_distance = sum(calculate_route_distance(route, customers) for route in current_routes)
            if current_distance < best_distance:
                best_solution = [route.copy() for route in current_routes]
                best_distance = current_distance
            return

        # Implement branching and bounding logic here
        for vehicle_idx in range(num_vehicles):
            for customer in remaining_customers:
                # Check capacity constraint before adding the customer to the route
                if sum(customers[customer]) + sum(customers[i][0] for i in current_routes[vehicle_idx]) <= capacity:
                    new_routes = [route.copy() for route in current_routes]
                    new_routes[vehicle_idx].append(customer)
                    new_remaining_customers = remaining_customers.copy()
                    new_remaining_customers.remove(customer)

                    explore_routes(new_routes, new_remaining_customers)

    explore_routes(best_solution, list(range(1, len(customers))))  # Start with unvisited customers

    return best_solution, best_distance



# Generate initial solution using Branch and Bound
initial_routes, initial_distance = branch_and_bound_vrp(customers, num_vehicles, capacity)
print("{:.1f}".format(initial_distance))
for route in initial_routes:
    print(" ".join(map(str,route)))
# Apply Tabu Search to improve the solution
final_routes, final_distance = tabu_search(initial_routes, customers, num_vehicles, capacity, num_iterations=100)
print("{:.1f}".format(final_distance))
for route in final_routes:
    print(" ".join(map(str, route)))

# Apply Local Search to further improve the solution
local_final_routes, local_final_distance = local_search(final_routes, customers, num_vehicles, capacity, num_iterations=100)

# Print the results
print("{:.1f}".format(local_final_distance))
for route in local_final_routes:
    print(" ".join(map(str, route)))
