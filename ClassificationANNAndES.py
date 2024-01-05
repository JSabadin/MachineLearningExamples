
## CLASIFICATION on XOR problem using ANN wth ES optimisation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Architecture
input_size = 2
hidden_size = 20
output_size = 2

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(1, hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(1, output_size)

# Forward function
def forward(x, W1, b1, W2, b2):
    hidden = np.tanh(np.dot(x, W1) + b1)
    output = np.dot(hidden, W2) + b2
    return hidden, output

# Training Data (XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])

# Evolutionary Strategy
population_size = 100
generations = 100
mutation_rate = 0.05

for generation in range(generations):
    # Mutate weights and biases
    new_W1 = np.array([W1 + mutation_rate*np.random.randn(*W1.shape) for _ in range(population_size)])
    new_b1 = np.array([b1 + mutation_rate*np.random.randn(*b1.shape) for _ in range(population_size)])
    new_W2 = np.array([W2 + mutation_rate*np.random.randn(*W2.shape) for _ in range(population_size)])
    new_b2 = np.array([b2 + mutation_rate*np.random.randn(*b2.shape) for _ in range(population_size)])

    # Evaluate population
    scores = []
    for i in range(population_size):
        _, output = forward(X, new_W1[i], new_b1[i], new_W2[i], new_b2[i])
        score = -np.mean(np.square(y - output))
        scores.append(score)

    # Select the best individual
    best_idx = np.argmax(scores)
    W1, b1 = new_W1[best_idx], new_b1[best_idx]
    W2, b2 = new_W2[best_idx], new_b2[best_idx]

    print(f"Generation {generation}, Best Score: {-scores[best_idx]}")

# Create meshgrid for 3D plotting
x_range = np.linspace(-0.5, 1.5, 30)
y_range = np.linspace(-0.5, 1.5, 30)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# Initialize plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Evaluate and plot neurons
Z1 = np.zeros((30, 30))
Z2 = np.zeros((30, 30))
for ix, x_val in enumerate(x_range):
    for iy, y_val in enumerate(y_range):
        _, output = forward(np.array([[x_val, y_val]]), W1, b1, W2, b2)
        Z1[ix, iy] = output[0][0]
        Z2[ix, iy] = output[0][1]

# Plot both output neurons on the same plot
surface1 = ax.plot_surface(X_grid, Y_grid, Z1, cmap='viridis', alpha=0.7)
surface2 = ax.plot_surface(X_grid, Y_grid, Z2, cmap='plasma', alpha=0.7)

# Plot the training points
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
_, training_outputs = forward(training_inputs, W1, b1, W2, b2)
ax.scatter(training_inputs[:, 0], training_inputs[:, 1], training_outputs[:, 0], c='r', marker='o', s=100, label='First Neuron Output for Training Points')
ax.scatter(training_inputs[:, 0], training_inputs[:, 1], training_outputs[:, 1], c='g', marker='x', s=100, label='Second Neuron Output for Training Points')

# Add color bar
fig.colorbar(surface1, ax=ax, pad=0.1, aspect=20)
fig.colorbar(surface2, ax=ax, pad=0.1, aspect=20)

# Set plot title and labels
ax.set_title("Output from Both Neurons and Training Points")
ax.set_xlabel("Input X")
ax.set_ylabel("Input Y")
ax.set_zlabel("Output Value")

plt.show()


