
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>

#include "gpu_wrapper.hpp"
#include "matrix_io.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [output_file]\n";
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = (argc == 3) ? argv[2] : "";

    try {
        std::ifstream inFile(inputFile);
        if (!inFile.is_open()) {
            throw std::runtime_error("Could not open input file: " + inputFile);
        }
        
        Matrix A = readMatrix(inFile);
        Matrix B = readMatrix(inFile);
        inFile.close();

        // start timer
        auto start = std::chrono::high_resolution_clock::now();

        Matrix C = multiplyMatricesGPU(A, B);

        // stop timer
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        if (!outputFile.empty()) {
            std::ofstream outFile(outputFile);
            if (!outFile.is_open()) {
                throw std::runtime_error("Could not open output file: " + outputFile);
            }
            writeMatrix(outFile, C);
            outFile.close();
        } else {
            writeMatrix(std::cout, C);
        }
        
        // print duration
        std::cout << duration.count() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}