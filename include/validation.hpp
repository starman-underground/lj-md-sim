#pragma once

#include <iostream>
#include <stdexcept>

// CUDA error checking macro
#define checkCuda(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Exception class for CUDA errors
class CudaException : public std::runtime_error {
public:
    CudaException(const std::string& message) : std::runtime_error(message) {}
};

// Validation function for CUDA calls
inline void validateCudaCall(cudaError_t result, const char* const func, const char* const file, int const line) {
    if (result != cudaSuccess) {
        std::string error_msg = std::string("CUDA Runtime Error: ") + cudaGetErrorString(result) + 
                               " at " + file + ":" + std::to_string(line) + " in function " + func;
        throw CudaException(error_msg);
    }
}

#define CUDA_CHECK(val) validateCudaCall((val), #val, __FILE__, __LINE__)