import numpy as np
import matplotlib.pyplot as plt

X = np.array([-3])
Y = np.array([-1 / 10])
X_train = X
X_test = X


class Dense:
    def __init__(self, in_size, out_size, seed=0):
        self.x = None
        np.random.seed(seed)
        self.W = np.random.normal(scale=0.1, size=(out_size, in_size))
        self.b = np.random.normal(scale=0.1, size=out_size)

    def forward(self, x):
        self.x = x
        return np.dot(self.W, x.transpose()) + self.b

    def backward(self, delta_, lr_=1):
        dw = np.outer(delta_, self.x)
        db = delta_

        self.W = self.W + lr_ * dw
        self.b = self.b + lr_ * db

        return np.dot(delta_, self.W)


class FullyConnectedNeuralNetwork:

    def __init__(self):
        self.d1 = Dense(1, 2)
        self.a1 = None
        self.d2 = Dense(2, 1)
        self.a2 = None

    def forward(self, x):
        net_ = self.d1.forward(x)
        net_ = self.a1 = (1 - np.exp(-net_)) / (1 + np.exp(-net_))
        net_ = self.d2.forward(net_)
        net_ = self.a2 = (1 - np.exp(-net_)) / (1 + np.exp(-net_))
        return net_

    def backward(self, dz, lr_):
        dz = dz * 0.5 * (1 - self.a2 ** 2)
        dz = self.d2.backward(dz, lr_)
        dz = dz * 0.5 * (1 - self.a1 ** 2)
        dz = self.d1.backward(dz, lr_)
        return dz


net = FullyConnectedNeuralNetwork()

lr = 1
epsilon = 1e-20
loss_train = []
loss_test = []
loss_mse = []

for i in range(100):
    y_predicted = net.forward(X_train)
    delta = Y - y_predicted
    net.backward(delta, lr)
    loss_train.append(delta.item())

    y_predicted = net.forward(X_test)
    delta = Y - y_predicted
    loss_test.append(delta.item())

    mse = np.square(np.sum((Y - y_predicted) ** 2))
    loss_mse.append(mse)

    print(f"epoch : {i}\t predict : {y_predicted[0]:.3f}\t MSE : {mse:.3}\t error : {delta[0]:.3}")
    if mse <= epsilon:
        break

# Строим график ошибки y_true - y_predicted на обучающей выборке
plt.plot(loss_train)
plt.grid()
plt.title('Dependence between error and epoch number on train')
plt.xlabel('epoch')
plt.ylabel('error')
plt.show()

# Строим график ошибки y_true - y_predicted на тестовой выборке
plt.plot(loss_test)
plt.grid()
plt.title('Dependence between error and epoch number on test')
plt.xlabel('epoch')
plt.ylabel('error')
plt.show()

# Строим график среднеквадратичной ошибки MSE
plt.plot(loss_mse)
plt.grid()
plt.title('Dependence between MSE and epoch number')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()

# Обученная модель имеет следующие веса
print(net.d1.W, net.d1.b)

print(net.d2.W, net.d2.b)
