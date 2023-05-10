import numpy as np
import model as mod
import read
import tensorflow as tf
import random
import itertools
import stats

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from tensorflow import keras

# Define the size of each layer.
input_size = 13
hidden_size = 20
output_size = 1
num_runs = 30

# Define the population size and mutation rate.
pop_size = 20
mutation_rate = 0.01

def preprocess():
    return read.prepare_dataset()


def fitness(model, X_train_tensor, y_train_tensor, X_test, y_test, type):
    if type == 'neural':
        y_pred = model.predict(X_test)
        accuracy = f1_score(y_test, np.round(y_pred), average='weighted')
    elif type == 'svm':
        model.fit(X_train_tensor, y_train_tensor)
        y_pred = model.predict(X_test)
        accuracy = f1_score(y_test, y_pred, average='weighted')
        #print(f1_score(y_test, y_pred, average=None))
    return accuracy

def init_pop(type):
    population = np.empty((pop_size,), dtype=object)

    if type == 'neural':
        # Fill the population array with random weights.
        for i in range(pop_size):
            model = mod.create_model()
            weights = mod.init_weights()
            model = mod.set_weights(model, weights)
            population[i] = model
    elif type == 'svm':
        # Fill the population array with random svm's.
        for i in range(pop_size):
            population[i] = mod.init_random_svm()

    return population

def uniform_mutation(new_indiv, delta):
    if random.uniform(0, 1) < mutation_rate:
        constant = random.uniform(-delta, delta)
        if new_indiv.C + constant > 0:
            new_indiv.C += constant

    if random.uniform(0, 1) < mutation_rate:
        constant = random.uniform(-delta, delta)
        if new_indiv.gamma + constant > 0:
            new_indiv.gamma += constant

    return new_indiv

def gaussian_mutation(new_indiv, sigma):
    if random.uniform(0, 1) < mutation_rate:
        delta_C = random.gauss(0, sigma)
        if new_indiv.C + delta_C > 0:
            new_indiv.C += delta_C

    if random.uniform(0, 1) < mutation_rate:
        delta_gamma = random.gauss(0, sigma)
        if new_indiv.gamma + delta_gamma > 0:
            new_indiv.gamma += delta_gamma

    return new_indiv

def one_point_crossover(indiv1, indiv2, type):
    if type == 'neural':
        cross_point = int(len(indiv1) / 2)
        new_indiv = indiv1[: cross_point] + indiv2[cross_point:]
    elif type == 'svm':
        new_indiv = mod.init_svm(indiv1.C, indiv2.gamma)
    return new_indiv

def uniform_crossover(indiv1, indiv2):
    new_indiv = []
    for i in range(len(indiv1)):
        if random.uniform(0, 1) < 0.5:
            new_indiv = new_indiv.append(indiv1[i])
        else:
            new_indiv = new_indiv.append(indiv2[i])

    return new_indiv

def blended_crossover(indiv1, indiv2):
    avg_c = float((indiv1.C + indiv2.C) / 2)
    avg_gamma = float((indiv1.gamma + indiv2.gamma) / 2)
    return mod.init_svm(avg_c, avg_gamma)

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
    for model, fit_val in population:
        sum += fit_val
        acc_sum.append(sum)

    for i in range(pop_size):
        slot = random.uniform(0, acc_sum[len(acc_sum) - 1])
        for j in range(len(acc_sum)):
            if slot < acc_sum[j]:
                #print(j, slot)
                pool[i] = population[j]
                break

    return pool

# Function that evolves the weights of a feedforward neural network
def evolve_nn(X_train_tensor, y_train_tensor, X_test, y_test, mut_type, cross_type, elitism_percentage = 0.2, mutation_rate = 0.3, delta = 2, sigma = 1, gen_num = 40000):
    population = init_pop('neural')
    history = []
    for gen in range(gen_num):
        print('----- GEN : ', gen, ' -----')
        fitnesses = []
        for i in range(pop_size):
            fitnesses.append([population[i], fitness(population[i], X_train_tensor, y_train_tensor, X_test, y_test, 'neural')])

        if gen > 10:
            fit = [tup[1] for tup in fitnesses]
            avg_f_score, best_f_score, accuracy = stats.fscore(population, X_train_tensor, y_train_tensor)
            history.append([fit, avg_f_score, best_f_score, accuracy])
            stats.plot_history(history)

        best_indivs = sorted(fitnesses, key=lambda x: x[1])
        elites = best_indivs[-int(pop_size * elitism_percentage):]

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

            weights = one_point_crossover(weights1, weights2, 'neural')

            for i in range(len(weights)):
                if random.uniform(0, 1) < mutation_rate:
                    weights[i] += random.uniform(-delta, delta)

            weights = np.array(weights)

            model = mod.create_model()
            model = mod.set_weights(model, weights)
            new_population[indiv] = model

        population = new_population

    return fitnesses, population



# Function that evolves the parameters of a support vector machine with an RBF kernel
def evolve_svm(X_train_tensor, y_train_tensor, X_test, y_test, mut_type, cross_type, elitism_percentage = 0.2, mutation_rate = 0.3, delta = 0.3, sigma = 1, gen_num = 40000):
    population = init_pop('svm')
    history = []
    for gen in range(gen_num):
        print('----- GEN : ', gen, ' -----')
        fitnesses = []

        for i in range(pop_size):
            fitnesses.append([population[i], fitness(population[i], X_train_tensor, y_train_tensor, X_test, y_test, 'svm')])

        fit = [tup[1] for tup in fitnesses]
        avg_acc, avg_fscore, best_fscore, best_model = stats.svm_scores(population, X_train_tensor, y_train_tensor, X_test, y_test)
        history.append([fit, avg_fscore, best_fscore, avg_acc])
        stats.plot_history(history)

        best_indivs = sorted(fitnesses, key=lambda x: x[1])
        elites = best_indivs[-int(pop_size * elitism_percentage):]

        pool = roullete(best_indivs)

        new_population = np.empty((pop_size,), dtype=object)

        for i in range(len(elites)):
            new_population[i] = elites[i][0]


        for indiv in range(int(pop_size * elitism_percentage), pop_size):
            indiv1 = pool[random.randint(0, len(pool) - 1)][0]
            indiv2 = pool[random.randint(0, len(pool) - 1)][0]

            if cross_type == 'blended':
                new_indiv = blended_crossover(indiv1, indiv2)
            elif cross_type == 'onepoint':
                new_indiv = one_point_crossover(indiv1, indiv2, 'svm')

            if mut_type == 'uniform':
                new_indiv = uniform_mutation(new_indiv, delta)
            elif mut_type == 'gaussian':
                new_indiv = gaussian_mutation(new_indiv, sigma)

            new_population[indiv] = new_indiv

        population = new_population

    return fitnesses, population
