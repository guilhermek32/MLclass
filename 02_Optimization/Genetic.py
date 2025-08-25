import random
import pickle
import os
import requests
import re

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

# Genetic algorithm for optimizing the antenna parameters
def genetic_algorithm(cache_file, population_size=500, generations=100, mutation_rate=0.1, tournament_size=5):
    cache = load_cache(cache_file)

    # Initialize the population with random values
    population = []
    for _ in range(population_size):
        phi1 = random.randint(0, 360)
        theta1 = random.randint(0, 360)
        phi2 = random.randint(0, 360)
        theta2 = random.randint(0, 360)
        phi3 = random.randint(0, 360)
        theta3 = random.randint(0, 360)
        fitness = simulate_antenna(phi1, theta1, phi2, theta2, phi3, theta3, cache)
        population.append((phi1, theta1, phi2, theta2, phi3, theta3, fitness))

    # Genetic algorithm loop
    for generation in range(generations):
        # Select the best individuals (tournament selection)
        selected_parents = []
        for _ in range(population_size // 2):
            tournament = random.sample(population, tournament_size)
            tournament.sort(key=lambda x: x[6], reverse=True)  # Sort by fitness
            selected_parents.append(tournament[0][:6])  # Take the best individual

        # Crossover to generate offspring
        # Crossover to generate offspring
        offspring = []
        for i in range(0, len(selected_parents), 2):
            parent1 = list(selected_parents[i])
            parent2 = list(selected_parents[i + 1] if i + 1 < len(selected_parents) else selected_parents[0])
            crossover_point = random.randint(1, 5)  # Random crossover point between 1 and 5
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            offspring.append(child1)
            offspring.append(child2)

        # Mutation
        for child in offspring:
            if random.random() < mutation_rate:
                mutate_index = random.randint(0, 5)
                child[mutate_index] = random.randint(0, 360)
        # Evaluate fitness of the offspring
        population = []
        for child in offspring:
            fitness = simulate_antenna(*child, cache)
            population.append((child[0], child[1], child[2], child[3], child[4], child[5], fitness))

        # Sort the population by fitness and keep the best ones
        population.sort(key=lambda x: x[6], reverse=True)
        population = population[:population_size]  # Keep only the best individuals

        # Save the cache to the file
        save_cache(cache, cache_file)

        # Print the best solution of this generation
        best_solution = population[0]
        print(f"Generation {generation + 1}: Best fitness = {best_solution[6]}")

    # Return the best solution found
    return population[0][:6], population[0][6]  # Return the best parameters and fitness

# Initial guess (values can be chosen randomly or based on some heuristic)
initial_phi1, initial_theta1 = 10, 180
initial_phi2, initial_theta2 = 1, 182
initial_phi3, initial_theta3 = 182, 156

# Cache file path
cache_file = "antenna_cache.pkl"

# Perform genetic algorithm to find the optimal solution
# Perform genetic algorithm to find the optimal solution
best_solution, best_fitness = genetic_algorithm(cache_file)
print("Genetic Algorithm Optimization Result:")
print(f"Best values found: {best_solution}")
print(f"Best simulated antenna response: {best_fitness}")
