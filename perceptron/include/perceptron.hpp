#pragma once
#include <vector>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>

template<typename T>
class ColumnVector {
public:
    ColumnVector() : n_(0) {}
    explicit ColumnVector(std::size_t n, const T& val = T()) : n_(n), data_(n, val) {}
    ColumnVector(std::initializer_list<T> init) : n_(init.size()), data_(init.begin(), init.end()) {}

    T& operator[](std::size_t i) { return data_.at(i); }
    const T& operator[](std::size_t i) const { return data_.at(i); }

    std::size_t size() const noexcept { return n_; }

    T dot(const ColumnVector& other) const {
        if (n_ != other.n_) throw std::invalid_argument("dot product: vector sizes must match");
        T s = T();
        for (std::size_t i = 0; i < n_; ++i) s += data_[i] * other.data_[i];
        return s;
    }

    ColumnVector operator+(const ColumnVector& other) const {
        if (n_ != other.n_) throw std::invalid_argument("addition: vector sizes must match");
        ColumnVector out(n_);
        for (std::size_t i = 0; i < n_; ++i) out.data_[i] = data_[i] + other.data_[i];
        return out;
    }
    
    // scalar multiplication (vector * scalar)
    ColumnVector operator*(const T& scalar) const {
        ColumnVector out(n_);
        for (std::size_t i = 0; i < n_; ++i) out.data_[i] = data_[i] * scalar;
        return out;
    }
    
    // in-place scalar multiply
    ColumnVector& operator*=(const T& scalar) {
        for (std::size_t i = 0; i < n_; ++i) data_[i] *= scalar;
        return *this;
    }
    
    // friend scalar * vector
    friend ColumnVector operator*(const T& scalar, const ColumnVector& v) {
        return v * scalar;
    }

private:
    std::size_t n_;
    std::vector<T> data_;
};

template<typename T>
class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(std::size_t r, std::size_t c, const T& val = T()) : rows_(r), cols_(c), data_(r * c, val) {}

    T& operator()(std::size_t i, std::size_t j) { return data_.at(i * cols_ + j); }
    const T& operator()(std::size_t i, std::size_t j) const { return data_.at(i * cols_ + j); }

    ColumnVector<T> col(std::size_t j) const {
        if (j >= cols_) throw std::out_of_range("column index out of range");
        ColumnVector<T> out(rows_);
        for (std::size_t i = 0; i < rows_; ++i) out[i] = this->operator()(i, j);
        return out;
    }

private:
    std::size_t rows_, cols_;
    std::vector<T> data_;
};

class Perceptron
{
    public:
        Perceptron(){
            w = ColumnVector<double>();
            b = 0.0;
            learning_rate = 0.01;
        };
        void fit(const Matrix<double>& X, const std::vector<int>& y);
        int predict(const ColumnVector<double>& x);
    private:
        double b, learning_rate;
        ColumnVector<double> w;
        std::vector<int> misclassified_indices;
        int sign(Matrix<double>& X, int idx);
        double loss(Matrix<double>& X, std::vector<int>& Y);
        void SGD(Matrix<double>& X, std::vector<int>& Y);
        bool is_misclassified(Matrix<double>& X, ColumnVector<int>& y, int idx);
        int random_index();
};
