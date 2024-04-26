%%cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Kernel code
__global__ void calc_prod_cuda(int* A, int* B, int* C, int rows_a, int cols_a, int rows_b, int cols_b) {
    // get row, column from block and thread index 
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int col = g / rows_a, row = g % rows_a;
 
    // calculate product for a cell 
    C[row * cols_b + col] = 0; 
    for (int i = 0; i < cols_b; i++) {
        C[row * cols_b + col] += A[row * cols_a + i] * B[i * cols_b + col];
    }
}

// serial product method
void calc_prod_serial(int* A, int* B, int* C, int rows_a, int cols_a, int rows_b, int cols_b) {
    // traverse rows 
    for (int i = 0; i < rows_a; i++) { 
        // traverse columns 
        for (int j = 0; j < cols_b; j++) {
            // calculate product for a cell 
            C[i * cols_b + j] = 0; 
            for (int k = 0; k < cols_b; k++) {
                C[i * cols_b + j] += A[i * cols_a + k] * B[k * cols_b + j];
            }
        }
    }
}

void initialize_matrix(
    int* host_a, int* host_b, int* host_prod, // Host matrices
    int rows_a, int cols_a, // dimensions of A
    int rows_b, int cols_b // dimensions of B
) {
    printf("Initializing matrix..\n");
    // initialize A, B 
    for (int i = 0; i < rows_a * cols_a; i++) {
        host_a[i] = i;
    }
    for (int i = 0; i < rows_b * cols_b; i++) { 
        host_b[i] = i + i;
    }
 
    printf("Matrix initialized\n"); 
    fflush(stdout); 
}

// function to print matrix 
void display_matrix(int* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) { 
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

// GPU matrix multiplication function
void calculate_cuda(
    int* host_a, int* host_b, int* host_prod, // Host matrices 
    int rows_a, int cols_a, // dimensions of A 
    int rows_b, int cols_b, // dimensions of B 
    int rows_prod, int cols_prod, // dimensions of product 
    bool show_product
) {
    // initialize matrix on device 
    int* device_a, * device_b, * device_prod; 
    printf("\nCalculating PARALLEL..\n");
    // Allocate on device 
    cudaMalloc((void**)&device_a, rows_a * cols_a * sizeof(int)); 
    cudaMalloc((void**)&device_b, rows_b * cols_b * sizeof(int)); 
    cudaMalloc((void**)&device_prod, rows_prod * cols_prod * sizeof(int));
    // Copy host to device 
    cudaMemcpy(device_a, host_a, rows_a * rows_b * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, rows_b * cols_b * sizeof(int), cudaMemcpyHostToDevice);
    // Define grid and block dimensions 
    dim3 blockDim(cols_b); 
    dim3 gridDim(rows_a);
    clock_t start_time = clock(); 
    // multiply 
    calc_prod_cuda <<<gridDim, blockDim>>> (device_a, device_b, device_prod, rows_a, cols_a, rows_b, cols_b);
    // Copy the result back to the host 
    printf("\nProduct calculated in %f seconds\n", (double)(clock() - start_time) / CLOCKS_PER_SEC); 
    cudaMemcpy(host_prod, device_prod, rows_prod * cols_prod * sizeof(int), cudaMemcpyDeviceToHost);
    if (show_product) { 
        printf("\nProduct is:\n"); 
        display_matrix(host_prod, rows_prod, cols_prod);
    }
    fflush(stdout);
    cudaFree(device_a); 
    cudaFree(device_b); 
    cudaFree(device_prod);
}

// serial matrix multiplication function
void calculate_serial(
    int* host_a, int* host_b, int* host_prod, // Host matrices 
    int rows_a, int cols_a, // dimensions of A 
    int rows_b, int cols_b, // dimensions of B 
    int rows_prod, int cols_prod, // dimensions of product 
    bool show_product
) {
    clock_t start_time = clock(); 
    printf("\nCalculating Serial..\n"); 
    calc_prod_serial(host_a, host_b, host_prod, rows_a, rows_b, rows_b, cols_b);
    if (show_product) { 
        printf("\nProduct is:\n"); 
        display_matrix(host_prod, rows_prod, cols_prod);
    }
    printf("\nProduct calculated in %f seconds\n", (double)(clock() - start_time) / CLOCKS_PER_SEC); 
    fflush(stdout); 
}

void free_matrix(int* host_a, int* host_b, int* host_prod) {
    // free memory 
    free(host_a); 
    free(host_b); 
    free(host_prod);
}

int main() { 
    int i = 1; 
    while (true) {
        if (i == 1) { 
            int rows_a=10, cols_a, rows_b, cols_b, see_prod;
            //printf("\nEnter dimensions of Matrix: "); 
            //scanf("%d", &rows_a); 
            cols_a = cols_b = rows_b = rows_a;
            //printf("\nDo you want to see product? "); 
            //scanf("%d", &see_prod); 
            printf("\n"); 
            int* A, * B, * prod;
            // matrix size 
            int rows_prod = rows_a; 
            int cols_prod = cols_b;
            // allocate on host 
            A = (int*)malloc(rows_a * cols_a * sizeof(int)); 
            B = (int*)malloc(rows_b * cols_b * sizeof(int)); 
            prod = (int*)malloc(rows_prod * cols_prod * sizeof(int));
            initialize_matrix(A, B, prod, rows_a, cols_a, rows_b, cols_b);
            calculate_cuda(A, B, prod, rows_a, cols_a, rows_b, cols_b, rows_prod, cols_prod, see_prod);
            calculate_serial(A, B, prod, rows_a, cols_a, rows_b, cols_b, rows_prod, cols_prod, see_prod);
            free_matrix(A, B, prod);
            i=0;
        } 
        else { 
            break; 
        } 
        //printf("Enter 1 to calculate again? "); 
        scanf("%d", &i); 
    } 
    return 0;
}

/*

Sure, let's go through the provided CUDA code line by line:

1. `#include "cuda_runtime.h"`: This line includes the CUDA runtime header file, which contains declarations needed for CUDA runtime API functions.

2. `#include "device_launch_parameters.h"`: This line includes the CUDA device launch parameters header file, which defines structures and macros used for configuring kernel launches.

3. Standard C and C++ library headers are included:
   - `<cstdio>` and `<stdio.h>`: These headers provide functions for input and output operations.
   - `<ctime>` and `<time.h>`: These headers provide functions for manipulating time.
   - `<cstdlib>` and `<stdlib.h>`: These headers provide general utility functions, such as memory allocation and random number generation.

4. Kernel function declaration:
   ```cpp
   __global__ void calc_prod_cuda(int* A, int* B, int* C, int rows_a, int cols_a, int rows_b, int cols_b)
   ```
   This declares a CUDA kernel function `calc_prod_cuda`, which performs matrix multiplication on the GPU. It takes pointers to matrices `A` and `B`, along with their dimensions, and computes the product matrix `C`.

5. Serial product method declaration:
   ```cpp
   void calc_prod_serial(int* A, int* B, int* C, int rows_a, int cols_a, int rows_b, int cols_b)
   ```
   This function performs matrix multiplication on the CPU in a serial manner. It takes pointers to matrices `A` and `B`, along with their dimensions, and computes the product matrix `C`.

6. Function to initialize matrices:
   ```cpp
   void initialize_matrix(int* host_a, int* host_b, int* host_prod, int rows_a, int cols_a, int rows_b, int cols_b)
   ```
   This function initializes matrices `A` and `B` with sequential values and prints a message indicating the initialization.

7. Function to display matrices:
   ```cpp
   void display_matrix(int* matrix, int rows, int cols)
   ```
   This function prints the contents of a matrix to the console.

8. GPU matrix multiplication function:
   ```cpp
   void calculate_cuda(int* host_a, int* host_b, int* host_prod, int rows_a, int cols_a, int rows_b, int cols_b, int rows_prod, int cols_prod, bool show_product)
   ```
   This function performs matrix multiplication on the GPU. It allocates memory for matrices `A`, `B`, and `C` on the device, copies data from the host to the device, launches the CUDA kernel `calc_prod_cuda`, and copies the result back to the host.

9. Serial matrix multiplication function:
   ```cpp
   void calculate_serial(int* host_a, int* host_b, int* host_prod, int rows_a, int cols_a, int rows_b, int cols_b, int rows_prod, int cols_prod, bool show_product)
   ```
   This function performs matrix multiplication on the CPU. It calls the serial product method `calc_prod_serial` to compute the product matrix.

10. Function to free matrix memory:
    ```cpp
    void free_matrix(int* host_a, int* host_b, int* host_prod)
    ```
    This function deallocates memory for matrices `A`, `B`, and `C`.

11. `main()` function:
    - It initializes loop variable `i` to `1`.
    - It enters an infinite loop.
    - Inside the loop, it initializes matrix dimensions and allocates memory for matrices `A`, `B`, and `prod`.
    - It calls the `initialize_matrix` function to initialize matrices `A` and `B`.
    - It calls the `calculate_cuda` and `calculate_serial` functions to perform matrix multiplication on the GPU and CPU, respectively.
    - It frees memory allocated for matrices.
    - It reads user input to decide whether to continue the loop.
    - Finally, it returns `0` to indicate successful completion of the program.
*/