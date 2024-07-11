#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>

constexpr float tolerance = 1.0e-3;

int main(const int argc, const char** argv){
    // Read output binary files
    std::string bin_name(argv[0]);
    std::string out_folder = bin_name.substr(0, bin_name.find_last_of('/')+1);

    // Full output file paths
    std::string file_serial = out_folder + "output_serial.bin";
    std::string file_sycl   = out_folder + "output_sycl.bin";
    
    // Create input stream from files
    std::ifstream ifs1(file_serial, std::ios::in | std::ios::binary);
    std::ifstream ifs2(file_sycl,   std::ios::in | std::ios::binary);

    // Check grid sizes
    int nx1, nx2, ny1, ny2;
    ifs1.read((char*) &nx1, sizeof(int));
    ifs1.read((char*) &ny1, sizeof(int));
    ifs2.read((char*) &nx2, sizeof(int));
    ifs2.read((char*) &ny2, sizeof(int)); 
    
    if (nx1 != nx2 || ny1 != ny2) {
        std::cerr << "Outputs have incompatible grid sizes.\n";
        std::exit(0);
    }

    // Check grid spacings
    float dx1, dx2, dy1, dy2;
    ifs1.read((char*) &dx1, sizeof(float));
    ifs1.read((char*) &dy1, sizeof(float));
    ifs2.read((char*) &dx2, sizeof(float));
    ifs2.read((char*) &dy2, sizeof(float)); 

    if (dx1 != dx2 || dy1 != dy2) {
        std::cerr << "Outputs have incompatible grid spacings.\n";
        std::exit(0);
    }

    // Read num steps
    int num_steps1, num_steps2;
    float dt1, dt2;
    ifs1.read((char*) &num_steps1, sizeof(int));
    ifs1.read((char*) &dt1, sizeof(float));
    ifs2.read((char*) &num_steps2, sizeof(int));
    ifs2.read((char*) &dt2, sizeof(float));
    
    if (num_steps1 != num_steps2 || dt1 != dt2) {
        std::cerr << "Outputs correspond to different evolution steps.\n";
        std::exit(0);
    }
    
    // Read value arrays
    const int num_elements = nx1 * ny1;
    std::vector<float> field_1(num_elements);
    std::vector<float> field_2(num_elements);
    ifs1.read((char*) field_1.data(), num_elements * sizeof(float));
    ifs2.read((char*) field_2.data(), num_elements * sizeof(float));
    
    // Compute L2 norm of the difference
    float err = 0;
    int diff_count = 0;
    for (int i = 0; i < num_elements; ++i) {
        float diff = std::abs(field_1[i] - field_2[i]);
        if (diff > tolerance) {
            std::cout << field_1[i] << " " << field_2[i] << "\n";
            diff_count++;
        }
        err += diff * diff;  
    }
    
    std::cout << "Values with difference larger than " << tolerance << ": " << diff_count << "\n";
    std::cout << "Total L2 norm of difference: " << std::sqrt(err) << "\n";

    return 0;
}
