
#ifndef MATRIX_IO_HPP
#define MATRIX_IO_HPP

#include <vector>
#include <iostream>

struct Matrix {
    int rows;
    int cols;
    std::vector<double> data; 
};

Matrix readMatrix(std::istream& is);
void writeMatrix(std::ostream& os, const Matrix& mat);

#endif
