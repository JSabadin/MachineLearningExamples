## RBF for polimoial problem
import numpy as np
import matplotlib.pyplot as plt

# Generate data
n_points = 100
x_data = np.linspace(-10, 10, n_points)
y_data = x_data ** 2 + 0.2 * x_data + 1 + 5 * np.random.normal(0, 1, n_points)

# Gaussian basis function with norm
def gaussian_basis(x, mu, sigma=1):
    return np.exp(-np.linalg.norm(x[:, None] - mu, axis=1) ** 2 / (2 * sigma ** 2))

# Neural network with 20 neurons in hidden layer
def neural_network(weights, x):
    n_hidden = 20
    mu_hidden = weights[:n_hidden]
    w_output = weights[n_hidden:2*n_hidden]
    b_output = weights[2*n_hidden]
    
    hidden_layer = gaussian_basis(x[:, None], mu_hidden)
    output_layer = np.dot(hidden_layer, w_output) + b_output
    
    return output_layer.flatten()

# Loss function
def loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# Evolutionary Strategies (ES)
def evolutionary_strategies(n_generations=4000, population_size=100, mutation_strength=0.1):
    n_hidden = 20
    population = np.random.normal(0, 1, (population_size, 2*n_hidden + 1))
    
    best_loss = []
    best_weights_history = []
    
    for generation in range(n_generations):
        fitness = [loss(neural_network(ind, x_data), y_data) for ind in population]
        sorted_indices = np.argsort(fitness)
        selected = population[sorted_indices[:population_size//2]]
        
        best_loss.append(fitness[sorted_indices[0]])
        best_weights_history.append(population[sorted_indices[0]])

        mean = np.mean(selected, axis=0)
        population = np.random.normal(mean, mutation_strength, (population_size, 2 * n_hidden + 1))
        
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

# Plotting some centers over generations
plt.figure()
plt.title("Some Centers over Generations")
plt.xlabel("Generation")
plt.ylabel("Center Value")
plt.plot(best_weights_history[:, :3])
plt.legend(["Center 1", "Center 2", "Center 3"])
plt.show()

# Plotting the final results
plt.figure()
plt.scatter(x_data, y_data, label='True Data')
plt.plot(x_data, neural_network(best_weights_history[-1], x_data), label='RBF Network Approximation', color='red')
plt.legend()
plt.show()

# Plotting the shape of Gaussian basis functions after final optimization
final_centers = best_weights_history[-1][:20]
x_values_for_gaussian = np.linspace(-10, 10, 400)
plt.figure()
plt.title("Shape of Gaussian Basis Functions After Final Optimization")
for mu in final_centers:
    plt.plot(x_values_for_gaussian, gaussian_basis(x_values_for_gaussian, mu))
plt.xlabel("x")
plt.ylabel("y")
plt.show()