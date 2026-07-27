/*
SYCL is a royalty-free, open-standard C++ programming model
designed for heterogeneous parallel computing, enabling developers
to write single-source code that runs on diverse hardware accelerators,
including CPUs, GPUs, and FPGAs. 

SYCL is a GPU programming environment that behaves like CUDA, but runs anywhere and is fully standard C++.

Intel oneAPI is an implementation of SYCL.

Explanation:
“I develop GPU programs in C++ using Intel oneAPI, which provides a SYCL
implementation. SYCL is a cross-platform programming model for heterogeneous
computing — meaning the same code can run on CPUs, GPUs, and other accelerators.
It’s conceptually similar to CUDA, but portable across different hardware.”
---------------------------------------------------------------
Compile: icx -fsycl vector_add.cpp -o vector_add.exe
Run: vector_add.exe
*/

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main() {
    constexpr int N = 16;

    std::vector<int> A(N), B(N), C(N);

    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    // Create a SYCL queue (CPU fallback if no GPU)
    sycl::queue q;

    std::cout << "Running on: "
              << q.get_device().get_info<sycl::info::device::name>()
              << "\n";

    {
        sycl::buffer<int> bufA(A.data(), sycl::range<1>(N));
        sycl::buffer<int> bufB(B.data(), sycl::range<1>(N));
        sycl::buffer<int> bufC(C.data(), sycl::range<1>(N));

        q.submit([&](sycl::handler& h) {
            auto a = bufA.get_access<sycl::access::mode::read>(h);
            auto b = bufB.get_access<sycl::access::mode::read>(h);
            auto c = bufC.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                int i = idx[0];
                c[i] = a[i] + b[i];
            });
        });
    } // buffers sync here

    for (int i = 0; i < N; i++) {
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << "\n";
    }

    return 0;
}
