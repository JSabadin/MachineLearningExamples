# regression using ANN + ES
import numpy as np
import matplotlib.pyplot as plt

# Generate data
n_points = 100
true_a = 0.1
true_b = 2
true_c = 2
true_d = 1
true_e = 0.2
true_f = 1

x_data = np.linspace(-10, 10, n_points)
y_data =  x_data**2  + true_e * x_data**1 + true_f + 5*np.random.normal(0, 1, n_points)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Neural network with 20 neurons in hidden layer
def neural_network(weights, x):
    n_hidden = 20
    w_hidden = weights[:n_hidden] # Weights for hidden layer
    b_hidden = weights[n_hidden:2*n_hidden] # Biases for hidden layer
    w_output = weights[2*n_hidden:3*n_hidden] # Weights for output layer
    b_output = weights[3*n_hidden] # Bias for output layer
    
    hidden_layer = sigmoid(x[:, None] * w_hidden + b_hidden)
    output_layer = np.dot(hidden_layer, w_output) + b_output
    
    return output_layer.flatten()

# Loss function
def loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# Evolutionary Strategies (ES)
def evolutionary_strategies(n_generations=4000, population_size=100, mutation_strength=0.1):
    # Initialize weights
    n_hidden = 20
    population = np.random.normal(0, 1, (population_size, n_hidden*3 + 1))
    
    best_loss = []
    best_weights_history = []

    for generation in range(n_generations):
        # Evaluate fitness (loss)
        fitness = [loss(neural_network(ind, x_data), y_data) for ind in population]
        
        # Select the best individuals based on fitness
        sorted_indices = np.argsort(fitness)
        selected = population[sorted_indices[:population_size//2]]

        # Record the best loss and the corresponding weights
        best_loss.append(fitness[sorted_indices[0]])
        best_weights_history.append(population[sorted_indices[0]])
        
        # Create the next generation through mutation
        mean = np.mean(selected, axis=0)
        population = np.random.normal(mean, mutation_strength, (population_size, 3 * n_hidden + 1))
        
        # Print progress
        if generation % 100 == 0:
            print(f"Generation {generation}, Best Loss: {fitness[sorted_indices[0]]}")
    
    return best_loss, np.array(best_weights_history)

# Run the ES algorithm
best_loss, best_weights_history = evolutionary_strategies()

# Plotting the loss over generations
plt.figure()
plt.title("Loss over Generations")
plt.xlabel("Generation")
plt.ylabel("Loss")
plt.plot(best_loss)
plt.show()

# Plotting some weights over generations (feel free to modify this to examine different weights)
plt.figure()
plt.title("Some Weights over Generations")
plt.xlabel("Generation")
plt.ylabel("Weight Value")
plt.plot(best_weights_history[:, :3])  # Plotting the first 3 weights as an example
plt.legend(["Weight 1", "Weight 2", "Weight 3"])
plt.show()

# Plotting the final results
plt.figure()
plt.scatter(x_data, y_data, label='True Data')
plt.plot(x_data, neural_network(best_weights_history[-1], x_data), label='Neural Network Approximation', color='red')
plt.legend()
plt.show()