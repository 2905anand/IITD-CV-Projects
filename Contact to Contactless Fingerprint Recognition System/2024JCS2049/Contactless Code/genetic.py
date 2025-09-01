
import numpy as np
import pandas as pd
import random



def apply_transformation(points, theta, scale, dx, dy):
    
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    
    S = np.array([[scale, 0],
                  [0, scale]])
    
    
    transformed_points = np.dot(points, (S @ R).T)
    
    
    transformed_points += np.array([dx, dy])
    
    return transformed_points


def compute_fitness(candidate, contactless_minutiae, contact_based_minutiae, threshold=5.0):
    
    theta, scale, dx, dy = candidate
    transformed_points = apply_transformation(contactless_minutiae, theta, scale, dx, dy)
    matches = 0

    
    for pt in transformed_points:
        distances = np.linalg.norm(contact_based_minutiae - pt, axis=1)
        if np.min(distances) < threshold:
            matches += 1

    return matches



def tournament_selection(population, fitnesses, tournament_size=3):
    
    selected_indices = random.sample(range(len(population)), tournament_size)
    best_candidate = max(selected_indices, key=lambda i: fitnesses[i])
    return population[best_candidate]

def crossover(parent1, parent2):
    
    theta = (parent1[0] + parent2[0]) / 2.0
    scale = (parent1[1] + parent2[1]) / 2.0
    dx = (parent1[2] + parent2[2]) / 2.0
    dy = (parent1[3] + parent2[3]) / 2.0
    return (theta, scale, dx, dy)

def mutate(candidate, mutation_rate=0.1):
    
    theta, scale, dx, dy = candidate
    if random.random() < mutation_rate:
        theta += np.random.normal(0, 0.1)
    if random.random() < mutation_rate:
        scale += np.random.normal(0, 0.05)
    if random.random() < mutation_rate:
        dx += np.random.normal(0, 1)
    if random.random() < mutation_rate:
        dy += np.random.normal(0, 1)
    return (theta, scale, dx, dy)



def genetic_algorithm(contactless_minutiae, contact_based_minutiae, 
                        population_size=50, generations=100, mutation_rate=0.1, threshold=5.0):
    
    
    population = []
    for _ in range(population_size):
        theta = np.random.uniform(-np.pi, np.pi)
        scale = np.random.uniform(0.8, 1.2)
        dx = np.random.uniform(-10, 10)
        dy = np.random.uniform(-10, 10)
        population.append((theta, scale, dx, dy))
    
    best_candidate = None
    best_fitness = -np.inf

    
    for generation in range(generations):
        fitnesses = [
            compute_fitness(candidate, contactless_minutiae, contact_based_minutiae, threshold)
            for candidate in population
        ]
        current_best = max(fitnesses)
        if current_best > best_fitness:
            best_fitness = current_best
            best_candidate = population[np.argmax(fitnesses)]
        print(f"Generation {generation}: Best Fitness = {current_best}")
        
        
        new_population = [population[np.argmax(fitnesses)]]
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population

    return best_candidate


def main():
    
    contactless_csv = "./contactless_minutiae_points/70_minutiae.csv"
    contact_based_csv = "./contact_minutiae_points/131_minutiae.csv"
    
    
    contactless_df = pd.read_csv(contactless_csv)
    contact_based_df = pd.read_csv(contact_based_csv)
    
    contactless_minutiae = contactless_df.iloc[:, :2].to_numpy() 
    contact_based_minutiae = contact_based_df.iloc[:, :2].to_numpy()  
    

    best_parameters = genetic_algorithm(
        contactless_minutiae, contact_based_minutiae,
        population_size=50,      
        generations=100,         
        mutation_rate=0.1,       
        threshold=5.0            
    )
    print("\nOptimized Transformation Parameters (theta, scale, dx, dy):", best_parameters)
    
    
    final_fitness = compute_fitness(best_parameters, contactless_minutiae, contact_based_minutiae, threshold=5.0)
    print("Final Similarity Score (number of matching minutiae):", final_fitness)

if __name__ == '__main__':
    main()