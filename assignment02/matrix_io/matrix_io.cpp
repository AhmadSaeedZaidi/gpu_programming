#include "matrix_io.hpp"
#include <stdexcept>
#include <string>

Matrix readMatrix(std::istream& is) {
    Matrix mat;
    if (!(is >> mat.rows >> mat.cols)) {
        throw std::runtime_error("Failed to read matrix dimensions or EOF reached.");
    }
    
    mat.data.resize(mat.rows * mat.cols);
    for (int i = 0; i < mat.rows * mat.cols; ++i) {
        if (!(is >> mat.data[i])) {
            throw std::runtime_error("Failed to read matrix data elements.");
        }
    }
    return mat;
}

void writeMatrix(std::ostream& os, const Matrix& mat) {
    os << mat.rows << " " << mat.cols << "\n";
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            os << mat.data[i * mat.cols + j];
            if (j < mat.cols - 1) os << " ";
        }
        os << "\n";
    }
}
