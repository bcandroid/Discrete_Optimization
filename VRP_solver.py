import itertools
import math
import random
import time
import sys

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
    # Get the total number of customers
    num_customers = len(customers)
    
    # Create a list of unvisited customers (excluding the warehouse)
    unvisited_customers = list(range(1, num_customers))
    
    # Initialize the list to store the routes
    routes = []

    # Sort unvisited customers by value-to-weight ratio in descending order
    unvisited_customers.sort(key=lambda c: customers[c][0] / max(customers[c][1], 1), reverse=True)

    # Iterate through each vehicle to create initial routes
    for _ in range(num_vehicles):
        current_capacity = 0
        current_customer = 0  # Start at the warehouse
        route = [current_customer]

        # Build the route until there are no more unvisited customers or capacity is exceeded
        while unvisited_customers:
            nearest_customer, distance = find_nearest_neighbor(unvisited_customers, current_customer, customers)
            
            # Check if adding the nearest customer exceeds the vehicle capacity
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


import heapq

def local_search(routes, customers, num_vehicles, capacity, num_iterations):
    # Initialize the best solution with the given routes
    best_solution = routes.copy()
    best_distance = sum(calculate_route_distance(route, customers) for route in best_solution)

    # Initialize the priority queue with the initial solution
    node_queue = [(best_distance, routes)]

    # Iterate through a given number of iterations
    for iteration in range(num_iterations):
        # Check if there are no more nodes to explore
        if not node_queue:
            break  # No more nodes to explore

        # Pop the node with the current best distance and routes
        current_distance, current_routes = heapq.heappop(node_queue)

        # Iterate through each vehicle's route
        for i in range(num_vehicles):
            # Iterate through each customer in the route
            for j in range(1, len(current_routes[i]) - 1):
                count = 0
                # Swap the position of the customer with subsequent customers in the route
                for k in range(j + 1, len(current_routes[i]) - 1):
                    # Create a new solution by swapping customers
                    new_solution = current_routes.copy()
                    new_solution[i][j], new_solution[i][k] = new_solution[i][k], new_solution[i][j]

                    # Calculate the distance of the new solution
                    new_distance = sum(calculate_route_distance(route, customers) for route in new_solution)

                    # Check if the new solution is better than the current best solution
                    if new_distance < current_distance:
                        # Update the best-known solution
                        best_solution = new_solution.copy()
                        best_distance = new_distance

                        # Add the promising node to the priority queue
                        heapq.heappush(node_queue, (best_distance, best_solution))
                    else:
                        count += 1
                    # Limit the count to avoid excessive iterations
                    if count == 100:
                        break

    return best_solution, best_distance




def branch_and_bound_vrp(customers, num_vehicles, capacity):
    # Generate an initial solution using the initial_solution function
    best_solution = initial_solution(customers, num_vehicles, capacity)

    # Check if an initial solution is found
    if best_solution is None:
        print("Unable to find an initial solution.")
        return [], float('inf')

    # Calculate the total distance of the initial solution
    best_distance = sum(calculate_route_distance(route, customers) for route in best_solution)

    def explore_routes(current_routes, remaining_customers, fractional=False):
        nonlocal best_solution, best_distance

        # Check if there are no remaining unvisited customers
        if not remaining_customers:
            # Calculate the total distance of the current solution
            current_distance = sum(calculate_route_distance(route, customers) for route in current_routes)
            
            # Update the best-known solution if the current solution is better
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

                    # Pruning: Stop exploring if the current partial solution is worse than the best-known solution
                    partial_distance = sum(calculate_route_distance(route, customers) for route in new_routes)

                    # Linear Relaxation: Allow fractional solutions
                    if fractional:
                        remaining_capacity = capacity - sum(customers[i][0] for i in new_routes[vehicle_idx])
                        remaining_demand = sum(customers[i][0] for i in new_remaining_customers)
                        remaining_customers_ratio = min(remaining_capacity / remaining_demand, 1.0)
                        partial_distance += remaining_customers_ratio * sum(
                            calculate_route_distance(route, customers) for route in new_routes)

                    # Stop exploring further if the current partial solution is worse than the best-known solution
                    if partial_distance >= best_distance:
                        return

                    # Recursively explore the new routes with the updated set of remaining customers
                    explore_routes(new_routes, new_remaining_customers, fractional)

    # Call the explore_routes function with fractional=True for linear relaxation
    explore_routes(best_solution, list(range(1, len(customers))), fractional=True)

    return best_solution, best_distance


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: %s input_file" % sys.argv[0])
        sys.exit()


    start_time = time.time()
    file_name = sys.argv[1]
    # Read input from input.txt
    with open(file_name) as file:
        lines = file.readlines()

    num_customers, num_vehicles, capacity = map(int, lines[0].split())
    customers = []
    
    for i in range(1, num_customers + 1):
        d, x, y = map(float, lines[i].split())
        customers.append((d, x, y))
    
    # Generate initial solution using Branch and Bound
    initial_routes, initial_distance = branch_and_bound_vrp(customers, num_vehicles, capacity)
    #print("{:.1f}".format(initial_distance))
    #for route in initial_routes:
        #print(" ".join(map(str,route)))
    
    # Apply Local Search to further improve the solution
    local_final_routes, local_final_distance = local_search(initial_routes, customers, num_vehicles, capacity, num_iterations=100)
    
    # Print the results
    print("{:.1f}".format(local_final_distance))
    for route in local_final_routes:
        print(" ".join(map(str, route)))
    
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    #print(f"Program çalışma süresi: {execution_time} saniye")
