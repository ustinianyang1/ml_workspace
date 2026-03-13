import numpy as np
import matplotlib.pyplot as plt
from perceptron_wrapper import Perceptron

X = np.array([[0.5, 0.5], [1.0, 1.5], [1.5, 0.8], [3.5, 4.5], [4.0, 3.5], [4.5, 5.0]])
y = np.array([1, 1, 1, -1, -1, -1])

model = Perceptron(learning_rate=0.1, max_iters=500)
print("开始训练...")
model.fit(X, y)

predictions = model.predict(X)
print(f"原始标签: {y}")
print(f"预测结果: {predictions}")

def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(8, 6))

    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='blue', label='Class 1')
    plt.scatter(X[y==-1][:, 0], X[y==-1][:, 1], color='red', label='Class -1')

    # 现在可以通过 C++ 核心拿到 w 了
    w = [model._core.w[0], model._core.w[1]]
    b = model._core.b

    x_points = np.linspace(0, 5, 10)
    if w[1] != 0:
        y_points = -(w[0] * x_points + b) / w[1]
        plt.plot(x_points, y_points, 'g--', label='Decision Boundary')

    plt.title("Perceptron Decision Boundary (C++ Core)")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_decision_boundary(model, X, y)