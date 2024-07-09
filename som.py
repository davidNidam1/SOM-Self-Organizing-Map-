import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the data
digits_data = pd.read_csv('digits_test.csv', header=None)
digits_labels = pd.read_csv('digits_keys.csv', header=None)

# Convert data to numpy arrays
digits_data = digits_data.values
digits_labels = digits_labels.values.flatten()

# Normalize the data
scaler = MinMaxScaler()
digits_data = scaler.fit_transform(digits_data)

# SOM dimensions
grid_size = (10, 10)
num_neurons = grid_size[0] * grid_size[1]
input_dim = digits_data.shape[1]

# Initialize the neurons with random weights
np.random.seed(42)  # For reproducibility
weights = np.random.rand(num_neurons, input_dim)

# Training parameters
num_iterations = 1000
initial_learning_rate = 0.1
initial_radius = max(grid_size) / 2
time_constant = num_iterations / np.log(initial_radius)

# Function to find the Best Matching Unit (BMU)
def find_bmu(input_vector, weights):
    distances = np.linalg.norm(weights - input_vector, axis=1)
    bmu_index = np.argmin(distances)
    return bmu_index, weights[bmu_index]

# Training the SOM
for iteration in range(num_iterations):
    # Decrease learning rate and radius over time
    learning_rate = initial_learning_rate * np.exp(-iteration / num_iterations)
    radius = initial_radius * np.exp(-iteration / time_constant)
    
    for input_vector in digits_data:
        bmu_index, bmu_weight = find_bmu(input_vector, weights)
        
        # Get BMU coordinates in the grid
        bmu_x, bmu_y = divmod(bmu_index, grid_size[1])
        
        # Update the weights of the BMU and its neighbors
        for i in range(num_neurons):
            neuron_x, neuron_y = divmod(i, grid_size[1])
            distance_to_bmu = np.sqrt((neuron_x - bmu_x) ** 2 + (neuron_y - bmu_y) ** 2)
            if distance_to_bmu <= radius:
                influence = np.exp(-(distance_to_bmu ** 2) / (2 * (radius ** 2)))
                weights[i] += learning_rate * influence * (input_vector - weights[i])

# Assign each input to the nearest neuron
neuron_assignments = np.zeros(len(digits_data), dtype=int)
for i, input_vector in enumerate(digits_data):
    bmu_index, _ = find_bmu(input_vector, weights)
    neuron_assignments[i] = bmu_index

# Determine the dominant digit for each neuron
dominant_digit = np.zeros(num_neurons, dtype=int)
dominant_percentage = np.zeros(num_neurons, dtype=float)
for i in range(num_neurons):
    assigned_digits = digits_labels[neuron_assignments == i]
    if len(assigned_digits) > 0:
        dominant_digit[i] = np.bincount(assigned_digits).argmax()
        dominant_percentage[i] = np.bincount(assigned_digits).max() / len(assigned_digits) * 100

# Plot the SOM
plt.figure(figsize=(10, 10))
for i in range(num_neurons):
    plt.text(i % grid_size[1], i // grid_size[1], str(dominant_digit[i]),
             ha='center', va='center', fontsize=12, color='black',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.text(i % grid_size[1], i // grid_size[1] + 0.3, f'{dominant_percentage[i]:.1f}%',
             ha='center', va='center', fontsize=8, color='red')

plt.xlim(-0.5, grid_size[1] - 0.5)
plt.ylim(-0.5, grid_size[0] - 0.5)
plt.gca().invert_yaxis()
plt.title('SOM Clustering of Digits')
plt.show()
