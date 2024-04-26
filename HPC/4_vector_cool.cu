#include <iostream>
using namespace std;

__global__
void add(int* A, int* B, int* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        C[tid] = A[tid] + B[tid];
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
    int N = 4;
    int* A, * B, * C;

    int vectorSize = N;
    size_t vectorBytes = vectorSize * sizeof(int);

    A = new int[vectorSize];
    B = new int[vectorSize];
    C = new int[vectorSize];

    initialize(A, vectorSize);
    initialize(B, vectorSize);

    cout << "Vector A: ";
    print(A, N);
    cout << "Vector B: ";
    print(B, N);

    int* X, * Y, * Z;
    cudaMalloc(&X, vectorBytes);
    cudaMalloc(&Y, vectorBytes);
    cudaMalloc(&Z, vectorBytes);

    cudaMemcpy(X, A, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, vectorBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, N);

    cudaMemcpy(C, Z, vectorBytes, cudaMemcpyDeviceToHost);

    cout << "Addition: ";
    print(C, N);

    delete[] A;
    delete[] B;
    delete[] C;

    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);

    return 0;
}

/*
Let's break down the provided code line by line:

1. ```cpp
   #include <iostream>
   using namespace std;
   ```
   This includes the `iostream` standard library, allowing for input and output operations, and brings the `std` namespace into scope, enabling the use of standard library functions and objects without prefixing them with `std::`.

2. ```cpp
   __global__ void add(int* A, int* B, int* C, int size) {
   ```
   This line defines a CUDA kernel function named `add`. CUDA kernel functions are executed on the GPU and are prefixed with `__global__`. This function takes pointers to three integer arrays (`A`, `B`, and `C`) and an integer `size` as arguments.

3. ```cpp
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   ```
   This line calculates the thread index within the grid of threads. It multiplies the block index (`blockIdx.x`) by the number of threads per block (`blockDim.x`) and adds the thread index within the block (`threadIdx.x`) to determine the unique thread ID (`tid`).

4. ```cpp
   if (tid < size) {
   ```
   This line checks if the thread ID is within the range of the array size. If it is, the thread will participate in the computation.

5. ```cpp
   C[tid] = A[tid] + B[tid];
   ```
   This line performs the addition operation element-wise between corresponding elements of arrays `A` and `B` and stores the result in array `C`. Each thread handles one element of the arrays based on its thread ID.

6. ```cpp
   void initialize(int* vector, int size) {
   ```
   This line defines a function named `initialize`, which takes a pointer to an integer array (`vector`) and an integer `size` as arguments. This function initializes the elements of the array with random values between 0 and 9.

7. ```cpp
   void print(int* vector, int size) {
   ```
   This line defines a function named `print`, which takes a pointer to an integer array (`vector`) and an integer `size` as arguments. This function prints the elements of the array to the console.

8. ```cpp
   int main() {
   ```
   This line marks the beginning of the `main` function, which is the entry point of the program.

9. ```cpp
   int N = 4;
   ```
   This line defines an integer variable `N` and initializes it to 4, representing the size of the arrays.

10. ```cpp
    int* A, * B, * C;
    ```
    This line declares pointers `A`, `B`, and `C` to integer arrays.

11. ```cpp
    int vectorSize = N;
    size_t vectorBytes = vectorSize * sizeof(int);
    ```
    These lines calculate the total number of elements (`vectorSize`) and the total number of bytes (`vectorBytes`) needed to store the arrays.

12. ```cpp
    A = new int[vectorSize];
    B = new int[vectorSize];
    C = new int[vectorSize];
    ```
    These lines dynamically allocate memory on the heap for arrays `A`, `B`, and `C` with `vectorSize` elements each.

13. ```cpp
    initialize(A, vectorSize);
    initialize(B, vectorSize);
    ```
    These lines call the `initialize` function to populate arrays `A` and `B` with random values.

14. ```cpp
    cout << "Vector A: ";
    print(A, N);
    cout << "Vector B: ";
    print(B, N);
    ```
    These lines print the contents of arrays `A` and `B` to the console using the `print` function.

15. ```cpp
    int* X, * Y, * Z;
    cudaMalloc(&X, vectorBytes);
    cudaMalloc(&Y, vectorBytes);
    cudaMalloc(&Z, vectorBytes);
    ```
    These lines allocate memory on the GPU for arrays `X`, `Y`, and `Z` using `cudaMalloc`.

16. ```cpp
    cudaMemcpy(X, A, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, vectorBytes, cudaMemcpyHostToDevice);
    ```
    These lines copy the data from arrays `A` and `B` on the host to arrays `X` and `Y` on the GPU using `cudaMemcpy`.

17. ```cpp
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    ```
    These lines calculate the number of blocks needed to cover the entire arrays, ensuring that there are enough threads to handle all elements.

18. ```cpp
    add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, N);
    ```
    This line launches the CUDA kernel `add` with the specified grid and block dimensions, passing arrays `X`, `Y`, and `Z`, along with the size `N`.

19. ```cpp
    cudaMemcpy(C, Z, vectorBytes, cudaMemcpyDeviceToHost);
    ```
    This line copies the result array `Z` from the GPU back to the host into array `C` using `cudaMemcpy`.

20. ```cpp
    cout << "Addition: ";
    print(C, N);
    ```
    This line prints the result array `C` containing the element-wise addition of arrays `A` and `B`.

21. ```cpp
    delete[] A;
    delete[] B;
    delete[] C;
    ```
    These lines free the dynamically allocated memory for arrays `A`, `B`, and `C` on the host.

22. ```cpp
    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);
    ```
    These lines free the memory allocated for arrays `X`, `Y`, and `Z` on the GPU.

23. ```cpp
    return 0;
    ```
    This line indicates successful termination of the `main` function and the program.

*/