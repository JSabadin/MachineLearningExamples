import numpy as np
import random
import matplotlib.pyplot as plt

# Generate synthetic data
n_points = 100
true_a = 0.1
true_b = 2
true_c = 2
true_d = 1
true_e = 0.2
true_f = 1
x_data = np.linspace(-10, 10, n_points)
y_data = true_a * x_data**5 + true_b * x_data**4 + true_c * x_data**3 + true_d * x_data**2  + true_e * x_data**1 + true_f +   3000*np.random.normal(0, 1, n_points)

# Initialize population
population_size = 50
population = np.random.randn(population_size, 6)  # 3 genes for a, b and c

# Evolutionary parameters
mutation_strength = 0.1
num_generations = 1000
best_individual = None
best_score = 0  # Now we're maximizing the score

# To store the history
a_history = []
b_history = []
c_history = []
d_history = []
e_history = []
f_history = []



def fitness(individual):
    a, b, c, d, e, f = individual
    y_pred = a * x_data**5 + b * x_data**4 + c * x_data**3 + d * x_data**2 + e * x_data**1 + f
    mse = np.mean((y_data - y_pred)**2)
    return 1 / (mse + 1e-8)  # Adding a small constant to avoid division by zero

# Evolutionary loop
for generation in range(num_generations):
    # Evaluate fitness
    scores = [fitness(ind) for ind in population]
    
    # Keep the best individual
    max_score = max(scores)
    if max_score > best_score:
        best_score = max_score
        best_individual = population[np.argmax(scores)]
    
    a_history.append(best_individual[0])
    b_history.append(best_individual[1])
    c_history.append(best_individual[2])
    d_history.append(best_individual[3])
    e_history.append(best_individual[4])
    f_history.append(best_individual[5])
    
    # Select individuals to create the next generation (roulette wheel selection)
    fitness_values = np.array(scores)
    prob_select = fitness_values / np.sum(fitness_values)
    selected_indices = np.random.choice(range(population_size), size=population_size, p=prob_select)
    selected_population = population[selected_indices]
    
    # Mutation
    mutations = mutation_strength * np.random.randn(population_size, 6)
    next_generation = selected_population + mutations
    
    population = next_generation

    # Display
    if generation % 100 == 0:
        print(f"Generation {generation}, Best Score: {best_score}, Best Individual: {best_individual}")

# Plot evolution of a and b
plt.figure()
plt.subplot(2, 1, 1)
plt.title("Evolution of a, b, c, d, e and f over generations")
plt.plot(a_history, label="a")
plt.plot(b_history, label="b")
plt.plot(c_history, label="c")
plt.plot(a_history, label="d")
plt.plot(b_history, label="e")
plt.plot(c_history, label="f")
plt.legend()

# Plot final model vs true data
plt.subplot(2, 1, 2)
plt.scatter(x_data, y_data, label="True Data")
plt.plot(x_data, best_individual[0]*x_data**5 + best_individual[1]*x_data**4 + best_individual[2]*x_data**3 + best_individual[3]*x_data**2 + best_individual[4]*x_data**1 + best_individual[5]  , 'r-', label="Model")
plt.legend()
plt.show()

print(f"True values: a = {true_a}, b = {true_b}, c = {true_c}, d = {true_d}, e = {true_e}, f = {true_f}")
print(f"Found values: a = {best_individual[0]}, b = {best_individual[1]}, c = {best_individual[2]}, d = {best_individual[3]}, e = {best_individual[4]}, f = {best_individual[5]}")



