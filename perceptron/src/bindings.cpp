#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "perceptron.hpp"

namespace py = pybind11;

PYBIND11_MODULE(perceptron_cpp, m) {
    py::class_<Matrix<double>>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def("__setitem__", [](Matrix<double> &m, std::pair<size_t, size_t> i, double v) {
            m(i.first, i.second) = v;
        });

    py::class_<Perceptron>(m, "Perceptron")
        .def(py::init<double, int>())
        .def("train", &Perceptron::train)
        .def("predict", [](Perceptron &self, std::vector<double> x_raw) {
            ColumnVector<double> x(x_raw.size());
            for(size_t i=0; i<x_raw.size(); ++i) x[i] = x_raw[i];
            return self.predict(x);
        });
}