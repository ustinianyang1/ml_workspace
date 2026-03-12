#include "perceptron.hpp"

int Perceptron::sign(const Matrix<double>& X, int idx) const
{
    return (X.dot_col(w, idx) + b >= 0.0) ? 1 : -1;
}

double Perceptron::loss(const Matrix<double>& X, const ColumnVector<int>& y) const
{
    double total_loss = 0.0;
    for (int idx : misclassified_indices)
        total_loss -= y[idx] * (X.dot_col(w, idx) + b);
    return total_loss;
}

void Perceptron::SGD(const Matrix<double>& X, const ColumnVector<int>& y)
{
    int idx = random_index();
    double step = y[idx] * learning_rate;

    for (int i = 0; i < w.size(); ++i)
        w[i] += X(i, idx) * step;
    b += step;
}

bool Perceptron::is_misclassified(const Matrix<double>& X, const ColumnVector<int>& y, int idx) const
{
    return y[idx] * (X.dot_col(w, idx) + b) <= 0.0;
}

int Perceptron::random_index()
{
    std::uniform_int_distribution<int> dis(0, misclassified_indices.size() - 1);
    return misclassified_indices[dis(rng_)];
}

void Perceptron::train(const Matrix<double>& X, const ColumnVector<int>& y)
{
    int n = y.size();
    for (int iter = 0; iter < max_iters; iter++) {
        misclassified_indices.clear();
        for (int i = 0; i < n; i++)
            if (is_misclassified(X, y, i))
                misclassified_indices.push_back(i);
        if (misclassified_indices.empty()) break;
        SGD(X, y);
    }
}