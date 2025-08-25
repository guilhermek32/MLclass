import random
import requests
import re
import pickle
import os
import math
import numpy as np

# Function to wrap values between 0 and 360 degrees
def wrap_angle(angle):
    return int(angle) % 360  # Ensure the result is always an integer

# Function to load the cache from a file
def load_cache(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
    return {}

# Function to save the cache to a file
def save_cache(cache, file_path):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"Error saving cache: {e}")

# Function to simulate antenna communication with different phi and theta values
def simulate_antenna(phi1, theta1, phi2, theta2, phi3, theta3, cache):
    # Check if the result is already in the cache
    key = (phi1, theta1, phi2, theta2, phi3, theta3)
    if key in cache:
        return cache[key]  # Return the cached result

    # URL with dynamic query parameters
    url = f'http://localhost:8080/antenna/simulate?phi1={phi1}&theta1={theta1}&phi2={phi2}&theta2={theta2}&phi3={phi3}&theta3={theta3}'
    
    try:
        # Send GET request to the server
        response = requests.get(url)
        
        # Print the full response text for debugging
        print(f"Response Text: {response.text}")
        
        # Check if the request was successful
        if response.status_code == 200:
            # Get the raw response text
            response_text = response.text.strip()
            
            # Attempt to find the first number (floating-point or integer) in the response
            match = re.match(r"([-+]?\d*\.\d+|\d+)", response_text)  # Match the first number in the string
            if match:
                # Extract the matched numeric value and return it as a float
                value = float(match.group(0))
                cache[key] = value  # Store the result in the cache
                return value
            else:
                print(f"Error: No valid number found in the response. Response text: {response_text}")
                return None
        else:
            print(f"Request failed with status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error during request: {e}")
        return None  # In case of error, return None

# Simulated Annealing algorithm to optimize phi and theta values
def simulated_annealing(initial_phi1, initial_theta1, initial_phi2, initial_theta2, initial_phi3, initial_theta3, cache_file, max_iterations=2000, initial_temperature=100, cooling_rate=0.99):
    # Load the cache from the file
    cache = load_cache(cache_file)

    current_phi1, current_theta1, current_phi2, current_theta2, current_phi3, current_theta3 = initial_phi1, initial_theta1, initial_phi2, initial_theta2, initial_phi3, initial_theta3
    current_value = simulate_antenna(current_phi1, current_theta1, current_phi2, current_theta2, current_phi3, current_theta3, cache)
    
    # If the initial solution is invalid, return None immediately
    if current_value is None:
        print("Initial simulation failed.")
        return None, []

    best_phi1, best_theta1, best_phi2, best_theta2, best_phi3, best_theta3 = current_phi1, current_theta1, current_phi2, current_theta2, current_phi3, current_theta3
    best_value = current_value

    # List to store the optimization history for plotting
    history = [(best_phi1, best_theta1, best_phi2, best_theta2, best_phi3, best_theta3, best_value)]

    temperature = initial_temperature  # Start with high temperature
    for iteration in range(max_iterations):
        # Perturb one of the angles slightly (randomly)
        neighbor_phi1 = wrap_angle(current_phi1 + random.randint(-10, 10))  # Perturb by ±4 degrees
        neighbor_theta1 = wrap_angle(current_theta1 + random.randint(-10, 10))
        neighbor_phi2 = wrap_angle(current_phi2 + random.randint(-10, 10))
        neighbor_theta2 = wrap_angle(current_theta2 + random.randint(-10, 10))
        neighbor_phi3 = wrap_angle(current_phi3 + random.randint(-10, 10))
        neighbor_theta3 = wrap_angle(current_theta3 + random.randint(-10, 10))
        
        # Evaluate the neighbor solution
        neighbor_value = simulate_antenna(neighbor_phi1, neighbor_theta1, neighbor_phi2, neighbor_theta2, neighbor_phi3, neighbor_theta3, cache)
        
        if neighbor_value is not None:
            # Calculate the change in energy (response value)
            delta_e = neighbor_value - current_value

            # If the new solution is better or if a random chance occurs (based on temperature), accept the new solution
            if delta_e > 0 or random.random() < math.exp(delta_e / temperature):
                current_phi1, current_theta1, current_phi2, current_theta2, current_phi3, current_theta3 = neighbor_phi1, neighbor_theta1, neighbor_phi2, neighbor_theta2, neighbor_phi3, neighbor_theta3
                current_value = neighbor_value

                # If the new solution is the best one, update best
                if current_value > best_value:
                    best_phi1, best_theta1, best_phi2, best_theta2, best_phi3, best_theta3 = current_phi1, current_theta1, current_phi2, current_theta2, current_phi3, current_theta3
                    best_value = current_value

        # Store the current best values in the history list
        history.append((best_phi1, best_theta1, best_phi2, best_theta2, best_phi3, best_theta3, best_value))

        # Cool down the temperature
        temperature *= cooling_rate

    # Save the cache to the file
    save_cache(cache, cache_file)
    
    return (best_phi1, best_theta1, best_phi2, best_theta2, best_phi3, best_theta3, best_value), history

# Initial guess (values can be chosen randomly or based on some heuristic)
initial_phi1, initial_theta1 = 0, 0
initial_phi2, initial_theta2 = 0, 0
initial_phi3, initial_theta3 = 0, 0

# Cache file path
cache_file = "antenna_cache.pkl"

# Perform simulated annealing to find the optimal solution and store the optimization history
result, history = simulated_annealing(initial_phi1, initial_theta1, initial_phi2, initial_theta2, initial_phi3, initial_theta3, cache_file)

print("Simulated Annealing Optimization Result:")
print(f"Initial values: phi1={initial_phi1}, theta1={initial_theta1}, phi2={initial_phi2}, theta2={initial_theta2}, phi3={initial_phi3}, theta3={initial_theta3}")
print(f"Result: {result}")

if result:
    best_phi1, best_theta1, best_phi2, best_theta2, best_phi3, best_theta3, best_value = result
    print(f"Best values found: phi1={best_phi1}, theta1={best_theta1}, phi2={best_phi2}, theta2={best_theta2}, phi3={best_phi3}, theta3={best_theta3}")
    print(f"Best simulated antenna response: {best_value}")
else:
    print("Optimization failed due to invalid simulations.")
