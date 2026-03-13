#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "perceptron.hpp"

namespace py = pybind11;

PYBIND11_MODULE(perceptron_cpp, m) {
    py::class_<ColumnVector<int>>(m, "IntVector")
        .def(py::init<int>())
        .def("set", [](ColumnVector<int> &v, int i, int val) { v[i] = val; })
        .def("__getitem__", [](const ColumnVector<int> &v, int i) { return v[i]; })
        .def("__len__", &ColumnVector<int>::size);

    py::class_<ColumnVector<double>>(m, "DoubleVector")
        .def(py::init<int>())
        .def("set", [](ColumnVector<double> &v, int i, double val) { v[i] = val; })
        .def("__getitem__", [](const ColumnVector<double> &v, int i) { return v[i]; })
        .def("__len__", &ColumnVector<double>::size);

    py::class_<Matrix<double>>(m, "Matrix")
        .def(py::init<int, int>())
        .def("set", [](Matrix<double> &mat, int r, int c, double val) { mat(r, c) = val; });

    py::class_<Perceptron>(m, "Perceptron")
        .def(py::init<>())
        .def_readwrite("b", &Perceptron::b)
        .def_readwrite("w", &Perceptron::w)
        .def_readwrite("learning_rate", &Perceptron::learning_rate)
        .def_readwrite("max_iters", &Perceptron::max_iters)
        .def("train", &Perceptron::train)
        .def("sign", &Perceptron::sign);
}