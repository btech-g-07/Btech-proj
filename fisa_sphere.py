import numpy as np
import random
import csv

# Population Initialization
def init_population(size, dim, lb, ub):
    population = np.array([[random.uniform(lb[i], ub[i]) for i in range(dim)] for _ in range(size)])
    return population

# Objective function definition
def myobj(solution):
    fitness = np.sum(solution**2)
    return fitness

def trim(solution, lb, ub):
    for i in range(len(solution)):
        if solution[i] > ub[i]:
            solution[i] = ub[i]
        if solution[i] < lb[i]:
            solution[i] = lb[i]
    return solution

# Calculating mean.
def mean(arr):
    mean_array = np.mean(arr, axis=0)
    return mean_array

# Finding out the value of best generations for FISA.
def mx_best_gen(population, fitness):
    mx_best = []
    for i in range(len(population)):
        curr = fitness[i]
        best = []
        for j in range(len(population)):
            if fitness[j] <= curr:
                best.append(population[j])
        
        if best:
            mx_best.append(np.mean(best))
        else:
            mx_best.append(curr)
    return mx_best

# Finding out the value of worst generations for FISA.
def mx_worst_gen(population, fitness):
    mx_worst = []
    for i in range(len(population)):
        curr = fitness[i]
        worst = []
        for j in range(len(population)):
            if fitness[j] > curr:
                worst.append(population[j])
        
        if worst:  # Check if worst is not empty
            mx_worst.append(np.mean(worst))
        else:
            mx_worst.append(curr)  # Handle the case when worst is empty
        
    return mx_worst

def fisa(population, dim, itr, objective_function, lb, ub):
    for j in range(itr):
        fitness_values = np.array([objective_function(individual) for individual in population])

        # Find the best and worst individuals in the current population
        best_index = np.argmin(fitness_values)
        worst_index = np.argmax(fitness_values)

        best_solution = population[best_index]
        best_fitness = fitness_values[best_index]

        # print("\nIteration ", j, "\nBest Solution:", best_solution)
        # print("Best Fitness:", best_fitness)

        mx_best = np.array(mx_best_gen(population, fitness_values))
        mx_worst = np.array(mx_worst_gen(population, fitness_values))

        new_pop = []
        new_fit = []

        # Update the position of each individual in the population
        for i in range(len(population)):
                r1= [0.5, 0.5]
                r2= [0.4, 0.4]
                new_sol = population[i] + r1*(np.abs(mx_best[i]) - population[i]) + r2*(population[i] - np.abs(mx_worst[i]))

                # Clip the new position to be within the bounds
                new_sol = trim(new_sol, lb, ub)
                new_pop.append(new_sol)

                new_fitness = objective_function(new_sol)
                new_fit.append(new_fitness)

                if new_fitness < fitness_values[i]:
                    population[i] = new_sol

    filename = "population.csv"
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write population data (one row per individual)
        for individual in population:
            writer.writerow(individual)  # Write each population value in a row

        # Write fitness data (one row per individual)
        for fitness in fitness_values:
            writer.writerow([fitness])

        # Write population data (one row per individual)
        for individual in new_pop:
            writer.writerow(individual)  # Write each population value in a row

        # Write fitness data (one row per individual)
        for fitness in new_fit:
            writer.writerow([fitness])

    # Return the best solution found
    best_solution = population[best_index]
    best_fitness = fitness_values[best_index]
    return best_solution, best_fitness

# Initialization of algorithm parameters
# pop_size = 25
dim = 2
itr = 1
lb = [-100, -100]
ub = [100, 100]

population =np.array([[67.8907, 25.6857],
                       [62.5171, -96.4914],
                       [-89.1315, 71.0286],
                       [-29.2615, -89.49],
                       [22.0788, 0.4925]])

# population = init_population(pop_size, dim, lb, ub)
# print("Initial Population:")
# print(population)

best_solution, best_fitness = fisa(population, dim, itr, myobj, lb, ub)

print("\nBest Solution:", best_solution)
print("Best Fitness:", best_fitness)
