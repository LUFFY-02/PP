#include <iostream>
using namespace std;


// CUDA code to multiply matrices
__global__ void multiply(int* A, int* B, int* C, int size) {
    // Uses thread idices and block indices to compute each element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int sum = 0;
        for (int i = 0; i < size; i++) {
            sum += A[row * size + i] * B[i * size + col];
        }
        C[row * size + col] = sum;
    }
}


void initialize(int* matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 10;
    }
}


void print(int* matrix, int size) {
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            cout << matrix[row * size + col] << " ";
        }
        cout << '\n';
    }
    cout << '\n';
}


int main() {
    int* A, * B, * C;

    int N = 2;
    int blockSize =  16;

    int matrixSize = N * N;
    size_t matrixBytes = matrixSize * sizeof(int);

    A = new int[matrixSize];
    B = new int[matrixSize];
    C = new int[matrixSize];

    initialize(A, N);
    initialize(B, N);
    cout << "Matrix A: \n";
    print(A, N);

    cout << "Matrix B: \n";
    print(B, N);

    
    int* X, * Y, * Z;
    // Allocate space
    cudaMalloc(&X, matrixBytes);
    cudaMalloc(&Y, matrixBytes);
    cudaMalloc(&Z, matrixBytes);

    // Copy values from A to X
    cudaMemcpy(X, A, matrixBytes, cudaMemcpyHostToDevice);
    
    // Copy values from A to X and B to Y
    cudaMemcpy(Y, B, matrixBytes, cudaMemcpyHostToDevice);

    // Threads per CTA dimension
    int THREADS = 2;

    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = N / THREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // Launch kernel
    multiply<<<blocks, threads>>>(X, Y, Z, N);

    cudaMemcpy(C, Z, matrixBytes, cudaMemcpyDeviceToHost);
    cout << "Multiplication of matrix A and B: \n";
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
Sure, let's go through the code line by line:

1. `#include <iostream>`: This line includes the standard input/output stream library, which is necessary for input/output operations.
   
2. `using namespace std;`: This line brings the entire `std` namespace into the current scope, allowing us to use functions and objects from the standard library without prefixing them with `std::`.

3. `__global__ void multiply(int* A, int* B, int* C, int size) { ... }`: This is a CUDA kernel function declaration. It will be executed on the GPU and is responsible for performing matrix multiplication. The `__global__` keyword specifies that this function is a CUDA kernel, and `void` indicates that it doesn't return any value. The function takes pointers to matrices `A`, `B`, and `C`, along with the size of the matrices.

4. `int row = blockIdx.y * blockDim.y + threadIdx.y;`: This line calculates the row index of the current thread in the grid.

5. `int col = blockIdx.x * blockDim.x + threadIdx.x;`: This line calculates the column index of the current thread in the grid.

6. `if (row < size && col < size) { ... }`: This condition ensures that the thread operates within the bounds of the matrices.

7. `int sum = 0;`: This initializes the sum variable to zero, which will accumulate the result of matrix multiplication.

8. `for (int i = 0; i < size; i++) { ... }`: This loop iterates over the elements of the row of matrix A and the column of matrix B, multiplying corresponding elements and accumulating the result in `sum`.

9. `C[row * size + col] = sum;`: This line assigns the computed sum to the corresponding element in the result matrix `C`.

10. `void initialize(int* matrix, int size) { ... }`: This function initializes a matrix of given size with random values.

11. `void print(int* matrix, int size) { ... }`: This function prints the elements of a matrix.

12. `int main() { ... }`: This is the main function where the program execution begins.

13. `int* A, * B, * C;`: These are pointers to matrices A, B, and C.

14. `int N = 2;`: This defines the size of the matrices as 2x2.

15. `int blockSize = 16;`: This defines the block size for the CUDA grid.

16. `int matrixSize = N * N;`: This calculates the total number of elements in the matrices.

17. `size_t matrixBytes = matrixSize * sizeof(int);`: This calculates the total memory required to store the matrices in bytes.

18. `A = new int[matrixSize]; B = new int[matrixSize]; C = new int[matrixSize];`: This dynamically allocates memory for matrices A, B, and C on the host (CPU).

19. `initialize(A, N); initialize(B, N);`: These function calls initialize matrices A and B with random values.

20. `cout << "Matrix A: \n"; print(A, N);`: This prints matrix A.

21. `cudaMalloc(&X, matrixBytes); cudaMalloc(&Y, matrixBytes); cudaMalloc(&Z, matrixBytes);`: This allocates memory for matrices X, Y, and Z on the GPU.

22. `cudaMemcpy(X, A, matrixBytes, cudaMemcpyHostToDevice); cudaMemcpy(Y, B, matrixBytes, cudaMemcpyHostToDevice);`: These lines copy the contents of matrices A and B from the host (CPU) to the device (GPU).

23. `int THREADS = 2; int BLOCKS = N / THREADS;`: These lines determine the number of threads per block and the number of blocks in each dimension of the grid.

24. `dim3 threads(THREADS, THREADS); dim3 blocks(BLOCKS, BLOCKS);`: These lines define the dimensions of the CUDA grid and blocks using the `dim3` structure.

25. `multiply<<<blocks, threads>>>(X, Y, Z, N);`: This line launches the CUDA kernel function `multiply` with the specified grid and block dimensions.

26. `cudaMemcpy(C, Z, matrixBytes, cudaMemcpyDeviceToHost);`: This copies the result matrix Z from the GPU back to the host.

27. `cout << "Multiplication of matrix A and B: \n"; print(C, N);`: This prints the result matrix C.

28. `delete[] A; delete[] B; delete[] C;`: This deallocates memory for matrices A, B, and C on the host.

29. `cudaFree(X); cudaFree(Y); cudaFree(Z);`: This deallocates memory for matrices X, Y, and Z on the GPU.

30. `return 0;`: This indicates successful completion of the program.
*/