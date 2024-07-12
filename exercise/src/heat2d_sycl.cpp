#include <algorithm>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

// TODO: include sycl header


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

    // TODO: initialize SYCL queue
    //
    
    // Print start message
    std::cout << "2D heat equation solver (sycl version)\n";
    std::cout << "Binary name: " << argv[0] << "\n";

    // TODO: get associated device name, and print it out
    std::cout << "Device name: " << /* device name here << */ "\n";

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

    // TODO: allocate memory on device
    // Remember that you can choose between C++-style templated function
    // or C-style memory allocation returning void* 
    // If you prefer, you can also try to use buffers instead

    // TODO: copy data to device (if required)
    // If you are using a memory API that does not provide implicit memory
    // managemet, you also have to mode data to device memory
    
    // HINT: pay attention to synchronization

    // TODO: define sycl ranges for kernel launch
    // Remember that a sycl::nd_range takes two sycl::ranges
    // (total amount of threads, and threads in workgroup)
    
    // Main loop
    for (int n = 0; n < num_steps; ++n) {
        // Update field (in kernel)
        //
        // TODO: submit a kernel to the device queue to perform the field update
        // Remeber that we need to do a "double buffering" to update the field, by 
        // writing the updated value and later using the it as input for the next
        // iteration

    }

    // TODO: copy last updated field back to host
    // Same considerations as for the memory copy to device also apply here

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
    
    // TODO: write field values at last timestep
    // If you decided to use USM you can leave this as is
    // If you used buffers you should adapt it to use an host_accessor to safely read buffer's data
    out_file.write((char*) field_n.data(), field_n.size() * sizeof(float));

    // Print output file path
    std::cout << "Output written at " << out_name << "\n";

    // TODO: release resources if required
    // e.g. if you allocated memory on device

    return 0;
}
