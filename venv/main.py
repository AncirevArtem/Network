import matplotlib.pyplot as plt
import numpy as np
import random


def sigmoid(x):
    """сигмоидальная функция, работает и с числами, и с векторами (поэлементно)"""
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)"""
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    """гиперболичесий тангенс, работает и с числами, и с векирпами(поэлементно)"""
    return 2 / (1 + np.exp(-2 * x)) - 1


def tanh_prime(x):
    """производная гиперболического тангенса,
    работает и с числами, и с векторами (поэлементно)"""
    return 1 - tanh(x) ** 2


# словарь функций - как ключи , и производных от них - как значение.
func_with_prime = {sigmoid: sigmoid_prime, tanh: tanh_prime}


class Network:

    def __init__(self, sizes, activations=[sigmoid], output=False):
        """
        Список ``sizes`` содержит количество нейронов в соответствующих слоях
        нейронной сети. К примеру, если бы этот лист выглядел как [2, 3, 1],
        то мы бы получили трёхслойную нейросеть, с двумя нейронами в первом
        (входном), тремя нейронами во втором (промежуточном) и одним нейроном
        в третьем (выходном, внешнем) слое. Смещения и веса для нейронных сетей
        инициализируются случайными значениями, подчиняющимися стандартному нормальному
        распределению. Обратите внимание, что первый слой подразумевается слоем,
        принимающим входные данные, поэтому мы не будем добавлять к нему смещение
        (делать это не принято, поскольку смещения используются только при
        вычислении выходных значений нейронов последующих слоёв)
        activations - функции активации для каждого слоя (кроме входного),
        по дефолту будет sigmoid
        """

        self.num_layers = len(sizes)
        assert self.num_layers >= 2, "Неверное число слоёв"
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.output = output
        self.num_activations = len(activations)
        assert self.num_activations == 1 or \
               self.num_layers == self.num_activations, "Неверное число функций активации"

        self.activations = activations

    def feedforward(self, a):
        """
        Вычислить и вернуть выходную активацию нейронной сети
        при получении ``a`` на входе (бывшее forward_pass).
        """
        for b, w, act in zip(self.biases, self.weights,self.activations):
            a = act(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """
        Обучить нейронную сеть, используя алгоритм стохастического
        (mini-batch) градиентного спуска.
        ``training_data`` - лист кортежей вида ``(x, y)``, где
        x - вход обучающего примера, y - желаемый выход (в формате one-hot).
        Роль остальных обязательных параметров должна быть понятна из их названия.
        Если предоставлен опциональный аргумент ``test_data``,
        то после каждой эпохи обучения сеть будет протестирована на этих данных
        и промежуточный результат обучения будет выведен в консоль.
        ``test_data`` -- это список кортежей из входных данных
        и номеров правильных классов примеров (т.е. argmax(y),
        если y -- набор ответов в той же форме, что и в тренировочных данных).
        Тестирование полезно для мониторинга процесса обучения,
        но может существенно замедлить работу программы.
        """

        if test_data is not None: n_test = len(test_data)
        n = len(training_data)
        success_tests = 0
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data is not None and self.output:
                success_tests = self.evaluate(test_data)
                print("Эпоха {0}: {1} / {2}".format(
                    j, success_tests, n_test))
            elif self.output:
                print("Эпоха {0} завершена".format(j))
        if test_data is not None:
            return success_tests / n_test

    def update_mini_batch(self, mini_batch, eta):
        """
        Обновить веса и смещения нейронной сети, сделав шаг градиентного
        спуска на основе алгоритма обратного распространения ошибки, примененного
        к одному mini batch.
        ``mini_batch`` - список кортежей вида ``(x, y)``,
        ``eta`` - величина шага (learning rate).
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        eps = eta / len(mini_batch)
        self.weights = [w - eps * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eps * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Вернуть кортеж ``(nabla_b, nabla_w)``,
        представляющий градиент для функции стоимости C_x.
        ``nabla_b`` и ``nabla_w`` - послойные списки массивов numpy,
        похожие на ``self.biases`` and ``self.weights``.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # прямой проход
        activation = x
        activations = [x]  # список для послойного хранения активаций
        zs = []  # список для послойного хранения z-векторов
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # обратный проход
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        """
        еременная l в цикле ниже используется не так, как описано во второй главе книги. 
        l = 1 означает последний слой нейронов, l = 2 – предпоследний, и так далее.
        Мы пользуемся преимуществом того, что в python можно использовать отрицательные индексы в массивах.
        """
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[1 - l].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Вернуть количество тестовых примеров, для которых нейронная сеть
        возвращает правильный ответ. Обратите внимание: подразумевается,
        что выход нейронной сети - это индекс, указывающий, какой из нейронов
        последнего слоя имеет наибольшую активацию.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Возвращает вектор частных производных (\partial C_x) / (\partial a)
        целевой функции по активациям выходного слоя.
        """
        return (output_activations - y)


def main():
    # data = np.loadtxt("data.csv", delimiter=",")
    # pears = data[:, 2] == 1
    # apples = np.logical_not(pears)
    # plt.scatter(data[apples][:, 0], data[apples][:, 1], color="red")
    # plt.scatter(data[pears][:, 0], data[pears][:, 1], color="green")
    # plt.xlabel("yellowness")
    # plt.ylabel("symmetry")
    # plt.show()
    #
    # # Подготовим данные
    #
    # X = data[:, :-1]
    # y = data[:, -1]
    #
    # X = np.hstack((np.ones((len(y), 1)), X))
    # y = y.reshape((len(y), 1))  # Обратите внимание на эту очень противную и важную строчку
    nn = Network([3,2] , activations=[sigmoid , sigmoid ])
    nn.biases = [np.array([[-1], [-1]])]
    nn.weights = [np.array([[-1, 1,-1], [ 1, -1, 1]])]
    x = np.array([[1], [2], [3]])
    y = np.array([[0], [1]])

    print(nn.backprop(x, y))

if __name__ == "__main__":
    main()
