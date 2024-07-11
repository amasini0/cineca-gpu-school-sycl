#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

void print_metrics(const std::vector<float>& timings)
{
    // compute and print metrics (skip first 5 entries)
    const auto fifth{ std::next(timings.begin(), 5) };
    const float avg{  std::accumulate( fifth, timings.end(), 0.0f) / (timings.size() - 5) };
    const float min{ *std::min_element(fifth, timings.end()) };
    const float max{ *std::max_element(fifth, timings.end()) };
    std::cout << "Host Metrics:" << std::endl;
    std::cout << "AVERAGE: " << avg << " ms" << std::endl;
    std::cout << "MINIMUM: " << min << " ms" << std::endl;
    std::cout << "MAXIMUM: " << max << " ms" << std::endl;
    std::cout << std::endl;
}

// overload for matmul cases
void check_results(const float *R, const float *C, const size_t size)
{
    float totErr{};
    float maxErr{};
    #pragma omp parallel for shared(maxErr) reduction(+:totErr)
    for (size_t i = 0; i < size * size; ++i) {
        const float diff = fabs(C[i] - R[i]);
        totErr += diff;
        #pragma omp critical
        {
            if (diff > maxErr) maxErr = diff;
        }
    }
    std::cout << "Results check:" << std::endl;
    std::cout << "ACC. ERROR: " << totErr << std::endl;
    std::cout << "MAX. ERROR: " << maxErr << std::endl;
    std::cout << std::endl;   
}

// overload for stencil case
void check_results(const float* hmap, const size_t size, const float min, const float max) 
{
    bool mismatch{ false };
    #pragma omp parallel for
    for (size_t i = 0; i < size * size; ++i) {
        float val{ hmap[i] };
        if (val < min || val > max) {
            #pragma omp critical
            {
                mismatch = true;
            }
        }
    } 
    std::cout << "Results check:" << std::endl;
    std::cout << "MISMATCH FOUND: " << std::boolalpha << mismatch << std::endl;
    std::cout << std::endl;
}

// overload for prefix sum
void check_results(const float* out, const size_t size) {
    bool mismatch{ false };
    for (size_t i{}; i < size * size; ++i) {
        if (out[i] != i) {
            mismatch = true;
            std::cout << std::endl;
            std::cout << "Expected: " << i      << std::endl;
            std::cout << "Obtained: " << out[i] << std::endl;
            break;
        }
    }
    std::cout << "Results check:" << std::endl;
    std::cout << "MISMATCH FOUND: " << std::boolalpha << mismatch << std::endl;
    std::cout << std::endl;
}

#endif // COMMON_UTILS_H
