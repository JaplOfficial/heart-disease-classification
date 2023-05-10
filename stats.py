import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel


def plot_history(history):
    medians = []
    bests = []
    avgs = []

    avgs_f1 = []
    bests_f1 = []

    avg_acc = []

    for gen, avg_f_score, best_f_score, acc in history:
        median = gen[int(len(gen) / 2)]
        avg = sum(gen) / len(gen)
        best = min(gen)

        avgs_f1.append(avg_f_score)
        bests_f1.append(best_f_score)

        avg_acc.append(acc)

        medians.append(median)
        avgs.append(avg)
        bests.append(best)

    plt.plot([i for i in range(len(history))], medians, label = "median")
    plt.plot([i for i in range(len(history))], avgs, label = "average")
    plt.plot([i for i in range(len(history))], bests, label = "best")

    plt.legend()
    plt.savefig("loss.jpg")
    plt.clf()

    plt.plot([i for i in range(len(history))], bests_f1, label = "best_fscore")
    plt.plot([i for i in range(len(history))], avgs_f1, label = "average_fscore")

    plt.legend()
    plt.savefig("fscore.jpg")
    plt.clf()

    plt.plot([i for i in range(len(history))], avg_acc, label = "average_accuracy")

    plt.legend()
    plt.savefig("accuracy.jpg")
    plt.clf()


def fscore(population, X_test, y_test):
    average = 0.0
    average_accuracy = 0.0
    best = 0.0

    for model in population:
        # Predict labels for test data using the trained neural network model
        y_pred = model.predict(X_test)

        y_pred_binary = np.round(y_pred).astype(int)

        # Compute weighted F1-score
        f1 = f1_score(y_test, y_pred_binary, average='weighted')
        accuracy = accuracy_score(y_test, y_pred_binary)
        average_accuracy += accuracy

        average += f1

        if f1 > best:
            best = f1

    return float(average / len(population)), best, float(average_accuracy / len(population))


def svm_scores(population, X_train, y_train, X_test, y_test):
    avg_acc = 0.0
    avg_fscore = 0.0
    best_fscore = 0.0
    best_model = None

    for i, model in enumerate(population):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute accuracy and f1-score
        acc = accuracy_score(y_test, y_pred)
        fscore = f1_score(y_test, y_pred, average='weighted')

        # Update average accuracy and f1-score
        avg_acc += acc
        avg_fscore += fscore

        # Update best f1-score and model
        if fscore > best_fscore:
            best_fscore = fscore
            best_model = model

    # Compute average accuracy and f1-score
    avg_acc /= len(population)
    avg_fscore /= len(population)

    return avg_acc, avg_fscore, best_fscore, best_model


def paired_t_test(A, B, alpha):

    t_statistic, p_value = ttest_rel(A, B)

    if p_value < alpha:
        return True
    else:
        return False
