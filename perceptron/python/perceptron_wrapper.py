import numpy as np
import perceptron_cpp

class Perceptron:
    def __init__(self, learning_rate=0.01, max_iters=1000):
        self._core = perceptron_cpp.Perceptron()
        self._core.learning_rate = learning_rate
        self._core.max_iters = max_iters

    def _convert(self, X, y=None):
        # 核心逻辑：NumPy (M, N) -> C++ Matrix (N, M)
        # N: 特征数 (rows), M: 样本数 (cols)
        X_t = X.T
        rows, cols = X_t.shape
        cpp_X = perceptron_cpp.Matrix(rows, cols)
        for r in range(rows):
            for c in range(cols):
                cpp_X.set(r, c, float(X_t[r, c]))
        
        if y is None: return cpp_X
        
        cpp_y = perceptron_cpp.IntVector(len(y))
        for i, val in enumerate(y):
            cpp_y.set(i, int(val))
        return cpp_X, cpp_y

    def fit(self, X, y):
        cpp_X, cpp_y = self._convert(X, y)
        self._core.train(cpp_X, cpp_y)

    def predict(self, X):
        cpp_X = self._convert(X)
        return np.array([self._core.sign(cpp_X, i) for i in range(X.shape[0])])