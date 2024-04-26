%%cu

#include <ctime>
#include <iostream>
#include <time.h> 

using namespace std;

__global__ void add(int* A, int* B, int* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}

void add_serial(int* A, int* B, int* C, int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

void initialize(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] = rand() % 10;
    }
}

void print(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        cout << vector[i] << " ";
    }
    cout << endl;
}

int main() {
    int i = 1;
    while (i == 1) {
        i=0;
        int N = 4;
        int* A, * B, * C;
        int vectorSize=2000000;
        //cout << "\nEnter size of Vector: ";
        //cin >> vectorSize;
        size_t vectorBytes = vectorSize * sizeof(int);
        A = new int[vectorSize]; 
        B = new int[vectorSize]; 
        C = new int[vectorSize];
        bool shouldPrint=0; 
        //cout << "\nDisplay Vectors? "; 
        //cin >> shouldPrint;
        initialize(A, vectorSize); 
        initialize(B, vectorSize); 
        if (shouldPrint) {
            cout << "\nVector A: "; 
            print(A, vectorSize); 
            cout << "Vector B: "; 
            print(B, vectorSize);
        }
        cout << "\nCalculating Parallel..\n"; 
        int* X, * Y, * Z; 
        cudaMalloc(&X, vectorBytes); 
        cudaMalloc(&Y, vectorBytes); 
        cudaMalloc(&Z, vectorBytes);
        cudaMemcpy(X, A, vectorBytes, cudaMemcpyHostToDevice); 
        cudaMemcpy(Y, B, vectorBytes, cudaMemcpyHostToDevice);
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 
        clock_t start_time = clock();
        add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, N); 
        cout << "Time taken: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "\n\n"; 
        cudaMemcpy(C, Z, vectorBytes, cudaMemcpyDeviceToHost);
        if (shouldPrint) { 
            cout << "Addition: "; 
            print(C, N);
        }
        cout << "Calculating Serial..\n"; 
        start_time = clock();
        add_serial(A, B, C, vectorSize);
        if (shouldPrint) { 
            cout << "Addition: "; 
            print(C, vectorSize);
        }
        cout << "Time taken: " << (double)(clock() - start_time) / CLOCKS_PER_SEC << "\n\n";
        delete[] A; 
        delete[] B; 
        delete[] C;
        cudaFree(X); 
        cudaFree(Y); 
        cudaFree(Z);
        //cout << "Enter 1 to go again: "; 
        //cin >> i;
    }
    return 0;
}

/*
Sure, let's go through the provided code line by line:

1. ```cpp
   %%cu
   ```
   This is a Jupyter magic command (`%%cu`) indicating that the following code cell contains CUDA C++ code.

2. ```cpp
   #include <ctime>
   #include <iostream>
   #include <time.h> 
   ```
   These lines include necessary C++ and CUDA libraries for handling time, input/output operations, and CUDA functionality.

3. ```cpp
   using namespace std;
   ```
   This line brings the `std` namespace into scope, allowing the use of standard library functions and objects without prefixing them with `std::`.

4. ```cpp
   __global__ void add(int* A, int* B, int* C, int size) {
   ```
   This line defines a CUDA kernel function named `add`. CUDA kernel functions are executed on the GPU and are prefixed with `__global__`. This function takes pointers to three integer arrays (`A`, `B`, and `C`) and an integer `size` as arguments.

5. ```cpp
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   ```
   This line calculates the thread index within the grid of threads. It multiplies the block index (`blockIdx.x`) by the number of threads per block (`blockDim.x`) and adds the thread index within the block (`threadIdx.x`) to determine the unique thread ID (`tid`).

6. ```cpp
   if (tid < size) {
       C[tid] = A[tid] + B[tid];
   }
   ```
   This block of code checks if the thread ID is within the range of the array size. If it is, the thread will participate in the addition operation and store the result in array `C`.

7. ```cpp
   void add_serial(int* A, int* B, int* C, int size) {
   ```
   This line defines a serial function `add_serial` that performs the same addition operation as the CUDA kernel function `add`, but on the CPU instead of the GPU.

8. ```cpp
   void initialize(int* vector, int size) {
   ```
   This line defines a function named `initialize` that initializes the elements of an integer array (`vector`) of a given `size` with random values between 0 and 9.

9. ```cpp
   void print(int* vector, int size) {
   ```
   This line defines a function named `print` that prints the elements of an integer array (`vector`) of a given `size` to the console.

10. ```cpp
    int main() {
    ```
Certainly! Let's continue explaining the main code:

11. ```cpp
    int i = 1;
    while (i == 1) {
    ```
    The variable `i` is initialized to `1`, and then there's a `while` loop that continues indefinitely as long as `i` remains `1`. This loop structure suggests that the code inside the loop will be executed repeatedly until a condition is met to change the value of `i`.

12. ```cpp
    i = 0;
    ```
    This line sets the value of `i` to `0`, effectively breaking the loop after the first iteration. This means that the loop body will only execute once in this code snippet.

13. ```cpp
    int N = 4;
    ```
    An integer variable `N` is initialized with the value `4`. This variable likely represents the size of the vectors being operated on.

14. ```cpp
    int vectorSize = 2000000;
    ```
    This line initializes an integer variable `vectorSize` with the value `2000000`, which likely represents the size of the vectors being operated on.

15. ```cpp
    size_t vectorBytes = vectorSize * sizeof(int);
    ```
    This line calculates the total number of bytes required to store the vectors (`A`, `B`, and `C`) based on the `vectorSize`. It's the product of `vectorSize` and the size of an integer (`sizeof(int)`).

16. ```cpp
    A = new int[vectorSize];
    B = new int[vectorSize];
    C = new int[vectorSize];
    ```
    Dynamic memory allocation (`new`) is used to allocate memory for three integer arrays (`A`, `B`, and `C`) of size `vectorSize`.

17. ```cpp
    initialize(A, vectorSize);
    initialize(B, vectorSize);
    ```
    The `initialize` function is called twice to populate the arrays `A` and `B` with random integer values between 0 and 9.

18. ```cpp
    cout << "Vector A: ";
    print(A, N);
    cout << "Vector B: ";
    print(B, N);
    ```
    The contents of vectors `A` and `B` are printed to the console using the `print` function.

19. ```cpp
    int* X, * Y, * Z;
    cudaMalloc(&X, vectorBytes);
    cudaMalloc(&Y, vectorBytes);
    cudaMalloc(&Z, vectorBytes);
    ```
    Three device (GPU) memory buffers (`X`, `Y`, and `Z`) are allocated using `cudaMalloc` to store the vectors `A`, `B`, and the result vector `C`.

20. ```cpp
    cudaMemcpy(X, A, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, vectorBytes, cudaMemcpyHostToDevice);
    ```
    The contents of vectors `A` and `B` are copied from host (CPU) memory to device (GPU) memory buffers `X` and `Y` using `cudaMemcpy`.

21. ```cpp
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    ```
    The number of threads per block (`threadsPerBlock`) is set to 256, and the number of blocks per grid (`blocksPerGrid`) is calculated based on the size of the vectors (`N`) and the number of threads per block.

22. ```cpp
    add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, N);
    ```
    The CUDA kernel function `add` is launched on the GPU with the specified grid and block dimensions, and the device memory buffers `X`, `Y`, and `Z` as arguments.

23. ```cpp
    cudaMemcpy(C, Z, vectorBytes, cudaMemcpyDeviceToHost);
    ```
    The result vector `C` is copied from the device (GPU) memory buffer `Z` to the host (CPU) memory using `cudaMemcpy`.

24. ```cpp
    cout << "Addition: ";
    print(C, N);
    ```
    The result vector `C` is printed to the console using the `print` function.

25. ```cpp
    delete[] A;
    delete[] B;
    delete[] C;
    ```
    Dynamic memory allocated for vectors `A`, `B`, and `C` is released to avoid memory leaks.

26. ```cpp
    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);
    ```
    The device (GPU) memory buffers `X`, `Y`, and `Z` are freed using `cudaFree`.

27. ```cpp
    return 0;
    ```
    The `main` function returns `0`, indicating successful execution of the program.
*/