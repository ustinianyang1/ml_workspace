#include "perceptron.hpp"
#include <random>

int Perceptron::sign(Matrix<double>& X, int idx)
{
    double dot = w.dot(X.col(idx));
    return (dot + b >= 0.0) ? 1 : -1;
}

double Perceptron::loss(Matrix<double>& X, std::vector<int>& Y)
{
    double total_loss = 0.0;
    for(int idx : misclassified_indices)
        total_loss -= Y[idx] * (w.dot(X.col(idx)) + b);
    return total_loss;
}

void Perceptron::SGD(Matrix<double>& X, std::vector<int>& Y)
{
    int idx = random_index();
    w = w + X.col(idx) * Y[idx] * learning_rate;
    b += Y[idx] * learning_rate;
}

bool Perceptron::is_misclassified(Matrix<double>& X, ColumnVector<int>& Y, int idx)
{
    return Y[idx] * (w.dot(X.col(idx)) + b) <= 0.0;
}

int Perceptron::random_index()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, misclassified_indices.size() - 1);
    return misclassified_indices[dis(gen)];
}