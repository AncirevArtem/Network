import matplotlib.pyplot as plt
import numpy as np
import time


class Neuron:

    def __init__(self, random_border=1, activation=None):
        random_border = np.sqrt(random_border)
        self.weight = np.random.randint(-random_border, random_border)
        self.bias = np.random.randint(-random_border, random_border)


def main():
    sizes = np.array([10**i for i in range(1, 7)])
    t_neurons = np.array([])
    t_np = np.array([])

    for n in sizes:
        t0 = time.process_time()
        neurons = [Neuron() for _ in range(n)]
        t1 = time.process_time()
        t_neurons = np.append(t_neurons, t1 - t0)
        t0 = time.process_time()
        np_neurons = [np.random.randint(-1, 1) for _ in range(n)]
        t1 = time.process_time()
        t_np = np.append(t_np, t1 - t0)

    fig, ax = plt.subplots()
    plt.ioff()
    ax.plot(sizes, t_neurons, 'rx', sizes, t_np, 'b+',
            linestyle='solid')
    ax.fill_between(sizes, t_neurons, t_np, where=t_neurons > t_np,
                    interpolate=True, color='green', alpha=0.3)
    lgnd = ax.legend(['neurons', 'np_arrays'], loc='upper center', shadow=True)
    lgnd.get_frame().set_facecolor('#ffb19a')
    plt.show()


if __name__ == "__main__":
    main()