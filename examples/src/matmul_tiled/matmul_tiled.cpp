#include "common_utils.h"
#include "sycl_utils.h"
#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

constexpr size_t RUNS{   25 };
constexpr size_t SIZE{ 1024 };
constexpr size_t TILE{   16 };
constexpr size_t VECT{    4 };

namespace chr = std::chrono;

///////////////////////////////////////
// user functions

void tiledMatMul(const float* A, const float* B, float* C, 
                 float* As, const sycl::nd_item<2> item)
{
    // workgroup index
    const sycl::id<2> group_id{ item.get_group().get_group_id() };
    const size_t bx{ group_id[0] };
    const size_t by{ group_id[1] };

    // workitem index in group
    const sycl::id<2> local_id{ item.get_local_id() };
    const size_t tx{ local_id[0] };
    const size_t ty{ local_id[1] };

    // vector to accumulate results
    float cv[TILE]{};

    const size_t aBegin{ SIZE * TILE * by };
    const size_t aEnd{ aBegin + SIZE - 1 };
    const size_t aStep{  TILE };

    const size_t bBegin = TILE * VECT * bx;
    const size_t bStep  = TILE * SIZE;

    for (size_t a{ aBegin }, b{ bBegin }; a <= aEnd; a+=aStep, b+=bStep) {
        for (size_t i{}; i < TILE/VECT; ++i) {
            As[(i * VECT + ty) + TILE * tx] = A[a + SIZE * (i * VECT + ty) + tx];
        }
        item.barrier(sycl::access::fence_space::local_space);

        const float* ap{ &As[0] };
        const float* bp{ &B[b + TILE * ty + tx] };

        for (size_t i{}; i < TILE; ++i) {
            float bv{ bp[0] };
            for (size_t j{}; j < TILE; ++j) {
                cv[j] += ap[j] * bv;
            }
            ap += TILE;
            bp += SIZE;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    size_t c{ (SIZE * TILE * by) + (TILE * VECT * bx) };
    c += TILE * ty + tx;
    for (size_t i{}; i < TILE; ++i) {
        C[c] = cv[i];
        c += SIZE;
    }
}

///////////////////////////////////////
// main

int main() {

    constexpr size_t nelems{ SIZE * SIZE };

    // define work group size and grid
    const sycl::range grid{ SIZE / (TILE * VECT), SIZE / TILE };
    const sycl::range block{ TILE, VECT };

    // init sycl queue with default device
    sycl::queue Q{};

    // print kernel and launch info
    print_kernel_info("Tiled Matrix Multiplication SYCL", SIZE, Q);
    print_launch_info(grid, block, Q);

    // timers
    chr::time_point<chr::steady_clock> start{}, stop{};
    std::vector<float> timings{};
    timings.reserve(RUNS);

    // allocate host matrices
    float* hstA{ new float[nelems] };
    float* hstB{ new float[nelems] };
    float* hstC{ new float[nelems] };

    // initialize host matrices
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            hstA[i * SIZE + j] =  10.0 * i + j;
            hstB[i * SIZE + j] = (i == j) ? 1.0 : 0.0;
            hstC[i * SIZE + j] = -42.0;
        }
    }

    // allocate device matrices
    float* devA{ sycl::malloc_device<float>(nelems, Q) };
    float* devB{ sycl::malloc_device<float>(nelems, Q) };
    float* devC{ sycl::malloc_device<float>(nelems, Q) };

    // copy A and B to device
    Q.copy<float>(hstA, devA, nelems).wait();
    Q.copy<float>(hstB, devB, nelems).wait();

    // benchmark kernel runs, 5 for warmup
    for (size_t i{}; i < RUNS; ++i) {
        start = chr::steady_clock::now();

        Q.submit([&](sycl::handler &h) {
            sycl::local_accessor<float> As{ sycl::range{ TILE*TILE }, h};

            h.parallel_for(
                sycl::nd_range<2>(grid * block, block),
                [=](sycl::nd_item<2> item) {
                    tiledMatMul(devA, devB, devC, &As[0], item);
                });
        }).wait();

        stop = chr::steady_clock::now();
        timings.push_back(chr::duration<double, std::milli>(stop - start).count());
    }

    // copy back C and check results
    Q.copy<float>(devC, hstC, nelems).wait();

    // check result and print metrics
    check_results(hstA, hstC, SIZE);
    print_metrics(timings);

    // free resources
    free(hstA);
    free(hstB);
    free(hstC);

    sycl::free(devA, Q);
    sycl::free(devB, Q);
    sycl::free(devC, Q);

    return 0;
}
