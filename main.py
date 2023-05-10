import evolve
import json


def main():

    runs = 30
    generations = 200

    # Split the datasets into train / validation / testing
    X_train, X_val, X_test, y_train, y_val, y_test = evolve.preprocess()

    max_fit = []
    avg_fit = []

    for run in range(runs):
        print('RUN :', run)
        fitnesses, _ = evolve.evolve_nn(X_train, y_train, X_val, y_val, 'uniform', 'onepoint', gen_num=generations)
        fit = [tup[1] for tup in fitnesses]
        max_fit.append(max(fit))
        avg_fit.append(sum(fit) / len(fit))

    print("Maximum fitness over all runs: ", max_fit)
    print("Average fitness over all runs: ", avg_fit)


if __name__ == "__main__":
    main()
