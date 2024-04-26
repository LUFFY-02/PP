#include <iostream>
using namespace std;

__global__ void CudaMultiplication(int* x, int* y, int* z, int N1, int N2, int M1, int M2){
      int row = blockIdx.x * blockDim.x + threadIdx.x;
      int col = blockIdx.y * blockDim.y + threadIdx.y;
      int val = 0;


      for(int k=0; k<N2; k++){
          val += x[row*N2+k]*y[k*M2+col] ;
      }

      z[row*M2+col]=val;
}

int main(){

    int N1 = 3;
    int N2 = 4;
    int M1 = 4;
    int M2 = 1;

    int *x = (int *)malloc(sizeof(int)*N1*N2);
    int *y = (int *)malloc(sizeof(int)*M1*M2);
    int *z = (int *)malloc(sizeof(int)*N1*M2);

    for(int i=0;i<N1;i++){
      for(int j=0;j<N2;j++) {
          x[i*N2+j]=1;
      }
    }

    for(int i=0;i<M1;i++){
      for(int j=0;j<M2;j++) {
          y[i*M2+j]=1;
      }
    }

    int *a , *b , *c;
    cudaMalloc(&a , sizeof(int)*N1*N2);
    cudaMalloc(&b , sizeof(int)*M1*M2);
    cudaMalloc(&c , sizeof(int)*N1*M2);

    cudaMemcpy(a, x, sizeof(int)*N1*N2, cudaMemcpyHostToDevice);
    cudaMemcpy(b, y, sizeof(int)*M1*M2, cudaMemcpyHostToDevice);

    dim3 th(N1,M2);
    CudaMultiplication<<<1, th>>>(a, b, c, N1, N2, M1, M2);

    cudaMemcpy(z, c, sizeof(int)*N1*M2, cudaMemcpyDeviceToHost);

    for(int i=0;i<N1;i++){
        for(int j=0;j<M2;j++){
            cout<<z[i*M2+j]<<" ";
        }
        cout<<endl;
    }
    return 0;
}
/*
Let's break down the code step by step:

1. `#include <iostream>`: This line includes the standard input/output stream library, allowing for input/output operations.

2. `using namespace std;`: This line brings the entire `std` namespace into the current scope, allowing us to use functions and objects from the standard library without prefixing them with `std::`.

3. `__global__ void CudaMultiplication(int* x, int* y, int* z, int N1, int N2, int M1, int M2) { ... }`: This is a CUDA kernel function declaration. It will be executed on the GPU and is responsible for performing matrix multiplication. The function takes pointers to matrices `x`, `y`, and `z`, along with their dimensions `N1`, `N2`, `M1`, and `M2`.

4. `int row = blockIdx.x * blockDim.x + threadIdx.x;`: This line calculates the row index of the current thread in the grid.

5. `int col = blockIdx.y * blockDim.y + threadIdx.y;`: This line calculates the column index of the current thread in the grid.

6. `int val = 0;`: This initializes the variable `val` to zero, which will accumulate the result of matrix multiplication.

7. `for(int k=0; k<N2; k++) { ... }`: This loop iterates over the elements of the row of matrix `x` and the column of matrix `y`, multiplying corresponding elements and accumulating the result in `val`.

8. `z[row*M2+col]=val;`: This line assigns the computed value to the corresponding element in the result matrix `z`.

9. `int main() { ... }`: This is the main function where the program execution begins.

10. `int N1 = 3; int N2 = 4; int M1 = 4; int M2 = 1;`: These variables define the dimensions of the matrices `x`, `y`, and `z`.

11. Memory allocation for matrices `x`, `y`, and `z` using `malloc` and initialization of their values to 1.

12. Memory allocation for matrices `a`, `b`, and `c` on the GPU using `cudaMalloc`.

13. Copying the contents of matrices `x` and `y` from the host (CPU) to the device (GPU) using `cudaMemcpy`.

14. `dim3 th(N1,M2);`: This line defines the dimensions of the CUDA grid and blocks using the `dim3` structure.

15. `CudaMultiplication<<<1, th>>>(a, b, c, N1, N2, M1, M2);`: This line launches the CUDA kernel function `CudaMultiplication` with the specified grid and block dimensions.

16. `cudaMemcpy(z, c, sizeof(int)*N1*M2, cudaMemcpyDeviceToHost);`: This line copies the result matrix `c` from the GPU back to the host.

17. Printing the result matrix `z` to the console.

18. `return 0;`: This indicates successful completion of the program.
*/