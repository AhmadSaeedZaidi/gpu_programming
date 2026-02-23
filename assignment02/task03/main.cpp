#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include "matrix_io.hpp"
#include "gpu_wrapper.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [output_file]\n";
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = (argc == 3) ? argv[2] : "";

    try {
        std::ifstream inFile(inputFile);
        if (!inFile.is_open()) throw std::runtime_error("Could not open input file.");
        
        Matrix A = readMatrix(inFile);
        Matrix B = readMatrix(inFile);
        inFile.close();

        Matrix C = multiplyMatricesGPU(A, B);

        if (!outputFile.empty()) {
            std::ofstream outFile(outputFile);
            writeMatrix(outFile, C);
            outFile.close();
        } else {
            writeMatrix(std::cout, C);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
