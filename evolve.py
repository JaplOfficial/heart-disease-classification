import numpy as np
import model as mod
import read
import tensorflow as tf
import random
import itertools
import stats


from tensorflow import keras

# Define the size of each layer.
input_size = 13
hidden_size = 20
output_size = 1

# Define the population size and mutation rate.
pop_size = 20
mutation_rate = 0.01

def preprocess():
    X_train, X_test, y_train, y_test = read.prepare_dataset()

    return X_train, y_train


def fitness(model, X_train_tensor, y_train_tensor):
    # Evaluate the model on the test data
    train_loss = model.evaluate(X_train_tensor, y_train_tensor)

    return train_loss

def init_pop():
    population = np.empty((pop_size,), dtype=object)

    # Fill the population array with random weights.
    for i in range(pop_size):
        model = mod.create_model()
        weights = mod.init_weights()
        model = mod.set_weights(model, weights)
        population[i] = model

    return population

def one_point_crossover(indiv1, indiv2):
    cross_point = int(len(indiv1) / 2)
    new_indiv = indiv1[: cross_point] + indiv2[cross_point:]
    return new_indiv

def uniform_crossover(indiv1, indiv2):
    new_indiv = []
    for i in range(len(indiv1)):
        if random.uniform(0, 1) < 0.5:
            new_indiv = new_indiv.append(indiv1[i])
        else:
            new_indiv = new_indiv.append(indiv2[i])
    pass

def get_model_weights(model):
    # Get the weights as numpy arrays
    weights = []
    for layer in model.layers:
        layer_weights = layer.get_weights()
        layer_weights_np = [w.ravel() for w in layer_weights] # flatten the weights
        flattened_weights = np.concatenate(layer_weights_np)
        weights.append(flattened_weights.tolist())
    return weights

def roullete(population):
    sum = 0.0
    acc_sum = []
    pool = np.empty((pop_size,), dtype=object)
    for model, fit in population:
        sum += 1 / (1 + fit)
        acc_sum.append(sum)
    print(acc_sum)
    for i in range(pop_size):
        slot = random.uniform(0, acc_sum[len(acc_sum) - 1])
        for j in range(len(acc_sum)):
            if slot < acc_sum[j]:
                #print(j, slot)
                pool[i] = population[j]
                break

    print(pool)
    return pool

def evolve(X_train_tensor, y_train_tensor, elitism_percentage = 0.2, mutation_rate = 0.05, delta = 0.2, gen_num = 40000):
    population = init_pop()
    history = []
    for gen in range(gen_num):
        print('----- GEN : ', gen, ' -----')
        fitnesses = []
        for i in range(pop_size):
            fitnesses.append([population[i], fitness(population[i], X_train_tensor, y_train_tensor)])

        if gen > 10:
            fit = [tup[1] for tup in fitnesses]
            history.append(fit)
            stats.plot_history(history)

        best_indivs = sorted(fitnesses, key=lambda x: x[1])

        elites = best_indivs[: int(pop_size * elitism_percentage)]
        pool = roullete(best_indivs)

        new_population = np.empty((pop_size,), dtype=object)

        for i in range(len(elites)):
            new_population[i] = elites[i][0]

        for indiv in range(int(pop_size * elitism_percentage), pop_size):
            indiv1 = pool[random.randint(0, len(pool) - 1)][0]
            indiv2 = pool[random.randint(0, len(pool) - 1)][0]

            weights1 = get_model_weights(indiv1)
            weights2 = get_model_weights(indiv2)

            weights1 = list(itertools.chain.from_iterable(weights1))
            weights2 = list(itertools.chain.from_iterable(weights2))

            weights = one_point_crossover(weights1, weights2)

            for i in range(len(weights)):
                if random.uniform(0, 1) < mutation_rate:
                    weights[i] += random.uniform(-delta, delta)

            weights = np.array(weights)

            model = mod.create_model()
            model = mod.set_weights(model, weights)
            new_population[indiv] = model

        population = new_population

    return fitnesses


X_train_tensor, y_train_tensor = preprocess()

print(evolve(X_train_tensor, y_train_tensor))
