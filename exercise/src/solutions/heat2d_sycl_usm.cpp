#include <algorithm>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

#include <sycl/sycl.hpp>

// Takes 2d indices and returns linear index
inline int get_linear_index(int x, int y, int width) {
    return x * width + y;
}

// Main code
int main(const int argc, const char** argv) {
    // Define simulation parameters 
    constexpr float a = 0.5; // Diffusion coefficient

    constexpr int nx = 2048; // Grid points on x
    constexpr int ny = 2048; // Grid points on y

    constexpr float dx = 0.01; // Grid spacing on x
    constexpr float dy = 0.01; // Grid spacing on y

    // Compute largest stable timestep
    constexpr float dx2 = dx * dx;
    constexpr float dy2 = dy * dy;
    constexpr float dt = dx2 * dy2 / (2. * a * (dx2 + dy2));

    constexpr int num_steps = 5000;

    // Initialize SYCL queue
    sycl::queue device_queue;
    
    // Print start message
    std::cout << "2D heat equation solver (sycl version)\n";
    std::cout << "Binary name: " << argv[0] << "\n";
    std::cout << "Device name: " << device_queue.get_device()
                                                .get_info<sycl::info::device::name>() << "\n";
    std::cout << "Grid size:    [" << nx << ", " << ny << "]\n";
    std::cout << "Grid spacing: [" << dx << ", " << dy << "]\n";
    std::cout << "Total steps:  " << num_steps << "\n";
    std::cout << "Timestep:     " << dt << "\n";

    // Create field vectors on CPU (current and updated)
    constexpr int num_elements = nx * ny;
    std::vector<float> field_n  (num_elements);
    std::vector<float> field_np1(num_elements);
    
    // Set initial conditions: disk of radius width/6
    constexpr float radius2 = (nx/6.0) * (nx/6.0);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const int linear_idx = get_linear_index(i, j, nx);

            // Distance of point i,j from center of domain
            const float ds2 = (i - nx/2)*(i - nx/2) + (j - ny/2)*(j - ny/2);

            // If distance is less then radius assign higher value
            if (ds2 < radius2) {
                field_n[linear_idx] = 65.0;
            } else {
                field_n[linear_idx] = 5.0;
            }
        }
    }
    
    // Copy field_n to field_np1
    std::copy(field_n.cbegin(), field_n.cend(), field_np1.begin());

    // Start timer
    std::chrono::high_resolution_clock clock;
    const auto start = clock.now();

    // Allocate memory on device
    float* d_field_n   = sycl::malloc_device<float>(num_elements, device_queue);
    float* d_field_np1 = sycl::malloc_device<float>(num_elements, device_queue);

    // Copy data to device
    device_queue.copy<float>(field_n.data(), d_field_n, num_elements);
    device_queue.copy<float>(field_np1.data(), d_field_np1, num_elements);
    device_queue.wait(); // Wait for end of memory transfers

    // Define kernel launch parameters
    sycl::range<2> total(nx, ny);
    sycl::range<2> group(16, 16);
    
    // Main loop
    for (int n = 0; n < num_steps; ++n) {
        // Update field (in kernel)
        device_queue.parallel_for(
            sycl::nd_range(total, group), 
            [=](auto nd_item) {
                int i = nd_item.get_global_id(0);
                int j = nd_item.get_global_id(1);
                    
                if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
                    const int linear_index = get_linear_index(i, j, nx);
                        
                    // Get field values required for update
                    const float field_ij = d_field_n[linear_index];
                    const float field_ijm1 = d_field_n[get_linear_index(i, j-1, nx)];
                    const float field_ijp1 = d_field_n[get_linear_index(i, j+1, nx)];
                    const float field_im1j = d_field_n[get_linear_index(i-1, j, nx)];
                    const float field_ip1j = d_field_n[get_linear_index(i+1, j, nx)];

                    // Compute field second derivatives
                    const float field_xx_ij = (field_im1j - 2.0*field_ij + field_ip1j) / dx2;
                    const float field_yy_ij = (field_ijm1 - 2.0*field_ij + field_ijp1) / dy2;

                    // Compute updated field
                    d_field_np1[linear_index] = field_ij + a * dt * (field_xx_ij + field_yy_ij); 
                }
            }
        ).wait(); // Wait for kernel to finish
        
        // Swap field with updated field for next iteration
        std::swap(d_field_n, d_field_np1);
    }

    // Copy data back to host
    device_queue.copy<float>(d_field_n, field_n.data(), num_elements).wait();

    // Compute elapsed time
    const auto stop = clock.now();
    const auto elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    std::cout << "\nSimulation completed in " << elapsed_time_ms << "ms\n";

    // Output results to binary file (placed in same folder as binary)
    std::string bin_name(argv[0]);
    std::string out_name = bin_name.substr(0, bin_name.find_last_of('/')+1)
                                   .append("output_sycl.bin");
    std::ofstream out_file(out_name, std::ios::out | std::ios::binary | std::ios::trunc);

    // Write grid size
    out_file.write((char*) &nx, sizeof(int));
    out_file.write((char*) &ny, sizeof(int)); 
    // Write grid spacings
    out_file.write((char*) &dx, sizeof(float));
    out_file.write((char*) &dy, sizeof(float));
    // Write number of steps
    out_file.write((char*) &num_steps, sizeof(int));
    // Write timestep
    out_file.write((char*) &dt, sizeof(int));
    // Write field values at last timestep
    out_file.write((char*) field_n.data(), field_n.size() * sizeof(float));

    // Print output file path
    std::cout << "Output written at " << out_name << "\n";

    // Free device memory
    sycl::free(d_field_n);
    sycl::free(d_field_np1);

    return 0;
}
