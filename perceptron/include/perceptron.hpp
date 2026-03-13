#pragma once
#include <vector>
#include <initializer_list>
#include <random>
#include <stdexcept>

template<typename T>
class ColumnVector {
public:
    ColumnVector() : n_(0) {}
    explicit ColumnVector(int n, const T& val = T()) : n_(n), data_(n, val) {}
    ColumnVector(std::initializer_list<T> init) : n_(0), data_(init.begin(), init.end()) {
        n_ = data_.size();
    }

    T& operator[](int i) { return data_.at(i); }
    const T& operator[](int i) const { return data_.at(i); }

    int size() const noexcept { return n_; }

    void resize(int n, const T& val = T()) {
        n_ = n;
        data_.assign(n, val);
    }

    T dot(const ColumnVector& other) const {
        if (n_ != other.n_) throw std::invalid_argument("dot product: vector sizes must match");
        T s = T();
        for (int i = 0; i < n_; ++i) s += data_[i] * other.data_[i];
        return s;
    }

    ColumnVector operator+(const ColumnVector& other) const {
        if (n_ != other.n_) throw std::invalid_argument("addition: vector sizes must match");
        ColumnVector out(n_);
        for (int i = 0; i < n_; ++i) out.data_[i] = data_[i] + other.data_[i];
        return out;
    }
    
    // scalar multiplication (vector * scalar)
    ColumnVector operator*(const T& scalar) const {
        ColumnVector out(n_);
        for (int i = 0; i < n_; ++i) out.data_[i] = data_[i] * scalar;
        return out;
    }
    
    // in-place scalar multiply
    ColumnVector& operator*=(const T& scalar) {
        for (int i = 0; i < n_; ++i) data_[i] *= scalar;
        return *this;
    }
    
    // friend scalar * vector
    friend ColumnVector operator*(const T& scalar, const ColumnVector& v) {
        return v * scalar;
    }

private:
    int n_;
    std::vector<T> data_;
};

template<typename T>
class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(int r, int c, const T& val = T()) : rows_(r), cols_(c), data_(r * c, val) {}

    T& operator()(int i, int j) { return data_.at(i * cols_ + j); }
    const T& operator()(int i, int j) const { return data_.at(i * cols_ + j); }

    ColumnVector<T> col(int j) const {
        if (j >= cols_) throw std::out_of_range("column index out of range");
        ColumnVector<T> out(rows_);
        for (int i = 0; i < rows_; ++i) out[i] = this->operator()(i, j);
        return out;
    }

    int rows() const noexcept { return rows_; }
    int cols() const noexcept { return cols_; }

    T dot_col(const ColumnVector<T>& v, int j) const {
        if (v.size() != rows_) throw std::invalid_argument("dot_col: vector size must match row count");
        if (j >= cols_) throw std::out_of_range("dot_col: column index out of range");

        T s = T();
        for (int i = 0; i < rows_; ++i) s += v[i] * this->operator()(i, j);
        return s;
    }

private:
    int rows_, cols_;
    std::vector<T> data_;
};

class Perceptron
{
    public:
        double b, learning_rate;
        int max_iters;
        ColumnVector<double> w;
        Perceptron()
            : b(0.0), learning_rate(0.01), max_iters(5000), w(), rng_(std::random_device{}()) {}

        void train(const Matrix<double>& X, const ColumnVector<int>& y);
        int sign(const Matrix<double>& X, int idx) const;
    private:
        std::mt19937 rng_;
        std::vector<int> misclassified_indices;

        double loss(const Matrix<double>& X, const ColumnVector<int>& y) const;
        void SGD(const Matrix<double>& X, const ColumnVector<int>& y);
        bool is_misclassified(const Matrix<double>& X, const ColumnVector<int>& y, int idx) const;
        int random_index();
};
