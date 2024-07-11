# SYCL Examples
This folder contains some example programs in SYCL and CUDA, in order to allow you to compare the two programmin models (syntax and APIs), and let you check some performance comparisons.

Remember however, that these results are not a precise measure of the performance, and actual results may be quite different from the ones obtained with these simple benchmarks.

## Compiling the examples
The only prerequisites to build the examples are a working SYCL compiler (possibly capable of generating device code for NVIDIA GPUs), and a working CUDA compiler.

The main choices for the SYCL compiler are either the OneAPI LLVM-based compiler `icpx` together with the plugin for NVIDIA GPUs provided by codeplay, or the AdaptiveCpp infrastructure. The first is the most easily obtainable betweem the two, since AdaptiveCpp requires compiling it from source.

Once you have a working SYCL and CUDA compilers, you can build the examples using the following commands

```bash
cd /path/to/examples
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=icpx
cmake --build . --parallel
```

The generated binaries are placed inside the `build/bin` folder. Technically, SYCL binaries should be ale to run also without a device, but will probably be really slow. However, this may be useful for debuggin purposes.

## Other resources
Here you can find some useful links to get more info on SYCL implementations

- [Intel OneAPI DPC++](https://www.intel.com/content/www/us/en/developer/tools/oneapi/data-parallel-c-plus-plus.html)
- [Codeplay NVIDIA/AMD plugins](https://codeplay.com/solutions/oneapi/plugins/)
- [AdaptiveCpp](https://adaptivecpp.github.io/)
