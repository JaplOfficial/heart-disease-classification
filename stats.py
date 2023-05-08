import matplotlib.pyplot as plt

def plot_history(history):
    medians = []
    bests = []
    avgs = []

    for gen in history:
        median = gen[int(len(gen) / 2)]
        avg = sum(gen) / len(gen)
        best = min(gen)

        medians.append(median)
        avgs.append(avg)
        bests.append(best)

    plt.plot([i for i in range(len(history))], medians, label = "median")
    plt.plot([i for i in range(len(history))], avgs, label = "average")
    plt.plot([i for i in range(len(history))], bests, label = "best")

    plt.legend()
    plt.savefig("progress.jpg")
    plt.clf()
