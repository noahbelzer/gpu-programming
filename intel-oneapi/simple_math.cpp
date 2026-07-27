/*
icx -fsycl simple_math.cpp -o simple_math.exe
simple_math.exe
*/

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main() {
    // Create SYCL queue (will pick CPU or GPU automatically)
    sycl::queue q;
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // USM lets you allocate memory that both CPU and GPU kernels can see directly, much like cudaMallocManaged in CUDA.
    // sycl::malloc_shared<int>(N, q)
    // This is a function call to allocate memory that is shared between host (CPU) and device (GPU).
    // ---- Breakdown: ----
    // Part	                    Meaning
    // sycl::malloc_shared	    SYCL function to allocate USM shared memory. Both host and device can access it.
    // <int>	                Template parameter — we are allocating memory for int elements.
    // (N, q)	                Function arguments:
    // N                        Number of elements to allocate (array size).
    // q	                    The SYCL queue associated with a device (CPU, GPU, FPGA). SYCL uses it to know which device this memory is “visible” to.

    // Allocate USM shared memory
    int* A = sycl::malloc_shared<int>(3, q);
    int* B = sycl::malloc_shared<int>(3, q);
    int* C = sycl::malloc_shared<int>(3, q);

    // Initialize input arrays directly in USM
    A[0] = 1; A[1] = 2; A[2] = 3;
    B[0] = 0; B[1] = 5; B[2] = 10;

    // Kernel: add arrays
    q.parallel_for(sycl::range<1>(3), [=](sycl::id<1> i) {
        C[i] = A[i] + B[i];  // result goes directly into USM array C
    }).wait(); // wait for kernel to finish

    // Print results directly from USM array
    for (int i = 0; i < 3; i++) {
        std::cout << "C[" << i << "] = " << C[i] << std::endl;
    }

    // Free USM memory
    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);

    return 0;
}
