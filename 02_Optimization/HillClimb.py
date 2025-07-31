import random
import requests
import random
import requests
import re

import random
import requests
import re

# Function to simulate antenna communication with different phi and theta values
def simulate_antenna(phi1, theta1, phi2, theta2, phi3, theta3):
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
                return float(match.group(0))
            else:
                print(f"Error: No valid number found in the response. Response text: {response_text}")
                return None
        else:
            print(f"Request failed with status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error during request: {e}")
        return None  # In case of error, return None

# Function to wrap values between 0 and 360 degrees
def wrap_angle(angle):
    return int(angle) % 360  # Ensure the result is always an integer

# Hill Climbing algorithm to optimize phi and theta values
def hill_climbing(initial_phi1, initial_theta1, initial_phi2, initial_theta2, initial_phi3, initial_theta3, max_iterations=10000):
    current_phi1, current_theta1, current_phi2, current_theta2, current_phi3, current_theta3 = initial_phi1, initial_theta1, initial_phi2, initial_theta2, initial_phi3, initial_theta3
    current_value = simulate_antenna(current_phi1, current_theta1, current_phi2, current_theta2, current_phi3, current_theta3)
    
    # If the initial solution is invalid, return None immediately
    if current_value is None:
        print("Initial simulation failed.")
        return None

    best_phi1, best_theta1, best_phi2, best_theta2, best_phi3, best_theta3 = current_phi1, current_theta1, current_phi2, current_theta2, current_phi3, current_theta3
    best_value = current_value
    
    for _ in range(max_iterations):
        # Generate neighbors by perturbing one parameter (ensure the values stay within 0 to 360)
        neighbor_phi1 = wrap_angle(current_phi1 + random.randint(-1, 1))  # Perturb by ±5 degrees, rounded to an integer
        neighbor_theta1 = wrap_angle(current_theta1 + random.randint(-1, 1))
        neighbor_phi2 = wrap_angle(current_phi2 + random.randint(-1, 1))
        neighbor_theta2 = wrap_angle(current_theta2 + random.randint(-1, 1))
        neighbor_phi3 = wrap_angle(current_phi3 + random.randint(-1, 1))
        neighbor_theta3 = wrap_angle(current_theta3 + random.randint(-1, 1))
        
        # Evaluate the neighbors
        neighbor_value = simulate_antenna(neighbor_phi1, neighbor_theta1, neighbor_phi2, neighbor_theta2, neighbor_phi3, neighbor_theta3)
        
        # If the simulation was successful and the new value is better, update
        if neighbor_value is not None and neighbor_value > best_value:  # Assuming we want to maximize the value
            best_phi1, best_theta1, best_phi2, best_theta2, best_phi3, best_theta3 = neighbor_phi1, neighbor_theta1, neighbor_phi2, neighbor_theta2, neighbor_phi3, neighbor_theta3
            best_value = neighbor_value
            current_phi1, current_theta1, current_phi2, current_theta2, current_phi3, current_theta3 = best_phi1, best_theta1, best_phi2, best_theta2, best_phi3, best_theta3
    
    return best_phi1, best_theta1, best_phi2, best_theta2, best_phi3, best_theta3, best_value


# Initial guess (values can be chosen randomly or based on some heuristic)
initial_phi1, initial_theta1 = 30, 10
initial_phi2, initial_theta2 = 32, 30
initial_phi3, initial_theta3 = 20, 10

# Perform hill climbing to find the optimal solution
result = hill_climbing(initial_phi1, initial_theta1, initial_phi2, initial_theta2, initial_phi3, initial_theta3)

print("Hill Climbing Optimization Result:")
print(f"Initial values: phi1={initial_phi1}, theta1={initial_theta1}, phi2={initial_phi2}, theta2={initial_theta2}, phi3={initial_phi3}, theta3={initial_theta3}")
print(f"Result: {result}")

if result:
    best_phi1, best_theta1, best_phi2, best_theta2, best_phi3, best_theta3, best_value = result
    print(f"Best values found: phi1={best_phi1}, theta1={best_theta1}, phi2={best_phi2}, theta2={best_theta2}, phi3={best_phi3}, theta3={best_theta3}")
    print(f"Best simulated antenna response: {best_value}")
else:
    print("Optimization failed due to invalid simulations.")
