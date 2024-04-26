#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Function to perform linear regression using gradient descent
void linear_regression_parallel(vector<double>& X, vector<double>& y, double& theta0, double& theta1, double alpha, int num_iterations) {
    int m = X.size();
    double cost;

    for (int iter = 0; iter < num_iterations; ++iter) {
        double sum_error0 = 0.0, sum_error1 = 0.0;

        // Parallelize the computation of gradient updates
        #pragma omp parallel for reduction(+:sum_error0, sum_error1)
        for (int i = 0; i < m; ++i) {
            double prediction = theta0 + theta1 * X[i];
            double error = prediction - y[i];

            sum_error0 += error;
            sum_error1 += error * X[i];
        }

        // Update parameters theta0 and theta1
        theta0 -= (alpha / m) * sum_error0;
        theta1 -= (alpha / m) * sum_error1;

        // Calculate cost (optional)
        cost = 0.0;
        for (int i = 0; i < m; ++i) {
            double prediction = theta0 + theta1 * X[i];
            cost += (prediction - y[i]) * (prediction - y[i]);
        }
        cost /= (2 * m);

        // Output cost for monitoring convergence (optional)
        cout << "Iteration " << iter + 1 << ", Cost: " << cost <<" theta0 "<<theta0<<" theta1 "<<theta1<< endl;
    }
}

int main() {
    // Example dataset
    vector<double> X = {1, 2, 3, 4, 5};
    vector<double> y = {3, 5, 7, 9, 11};

    // Initial parameters
    double theta0 = 0.0;
    double theta1 = 0.0;

    // Learning rate and number of iterations
    double alpha = 0.01;
    int num_iterations = 1000;

    // Perform linear regression using gradient descent
    linear_regression_parallel(X, y, theta0, theta1, alpha, num_iterations);

    // Output final parameters
    cout << "Final parameters: theta0 = " << theta0 << ", theta1 = " << theta1 << endl;

    return 0;
}


/*
Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data. It is one of the simplest and most commonly used techniques in statistical modeling and machine learning for predictive analysis.

In simple linear regression, there is only one independent variable, \(x\), and one dependent variable, \(y\). The relationship between \(x\) and \(y\) is modeled as a straight line:

\[ y = mx + b \]

Where:
- \(y\) is the dependent variable (the variable we want to predict),
- \(x\) is the independent variable (the variable used for prediction),
- \(m\) is the slope of the line (the change in \(y\) per unit change in \(x\)),
- \(b\) is the y-intercept (the value of \(y\) when \(x = 0\)).

The goal of linear regression is to find the best-fitting line through the data points that minimizes the sum of the squared differences between the observed \(y\) values and the predicted \(y\) values (\(mx + b\)).

In multiple linear regression, there are multiple independent variables, and the relationship between the dependent variable and the independent variables is modeled as a linear combination:

\[ y = b_0 + b_1x_1 + b_2x_2 + ... + b_px_p \]

Where:
- \(y\) is the dependent variable,
- \(b_0\) is the intercept,
- \(b_1, b_2, ..., b_p\) are the coefficients for the independent variables \(x_1, x_2, ..., x_p\).

The coefficients are estimated using methods such as least squares, which minimizes the sum of the squared differences between the observed and predicted values.
Gradient descent linear regression is a method used to optimize the parameters of a linear regression model by iteratively adjusting them in the direction that minimizes a loss function. In the context of linear regression, the loss function typically represents the difference between the predicted values of the model and the actual observed values in the training data.

Here's how gradient descent linear regression works:

1. **Initialization**: Start with initial values for the parameters of the linear regression model (e.g., coefficients and intercept).

2. **Calculate Predictions**: Use the current parameter values to make predictions for the dependent variable based on the independent variables in the training data.

3. **Compute Loss**: Calculate the loss, which is a measure of how well the model's predictions match the actual observed values. The loss function is typically a measure of the difference between the predicted and actual values, such as the mean squared error (MSE) or the mean absolute error (MAE).

4. **Compute Gradient**: Calculate the gradient of the loss function with respect to each parameter. The gradient indicates the direction and magnitude of the steepest increase in the loss function. This step involves taking the derivative of the loss function with respect to each parameter.

5. **Update Parameters**: Adjust the parameters of the model by moving them in the opposite direction of the gradient, scaled by a factor known as the learning rate. This step aims to minimize the loss function by iteratively updating the parameters in the direction that decreases the loss.

6. **Iterate**: Repeat steps 2-5 until a stopping criterion is met, such as reaching a maximum number of iterations or achieving a desired level of convergence.

Gradient descent is an iterative optimization algorithm, and its effectiveness depends on the choice of learning rate, the convergence criterion, and the initialization of parameters. It is a widely used method for training linear regression models and other machine learning algorithms.

Certainly! Let's break down the code and explain each part:

1. **Header Files**: 
   ```cpp
   #include <iostream>
   #include <vector>
   #include <omp.h>
   ```
   These lines include the necessary header files for input/output operations (`iostream`), working with dynamic arrays (`vector`), and OpenMP for parallel computing (`omp.h`).

2. **Namespace**: 
   ```cpp
   using namespace std;
   ```
   This line brings the entire `std` namespace into the current scope, allowing us to use standard library functions without prefixing them with `std::`.

3. **Function: linear_regression_parallel**: 
   ```cpp
   void linear_regression_parallel(vector<double>& X, vector<double>& y, double& theta0, double& theta1, double alpha, int num_iterations)
   ```
   This function performs linear regression using gradient descent in parallel. It takes input vectors `X` and `y`, initial parameters `theta0` and `theta1`, learning rate `alpha`, and the number of iterations `num_iterations`.

4. **Variable Initialization**: 
   ```cpp
   int m = X.size();
   double cost;
   ```
   `m` stores the number of data points in the dataset. `cost` will be used to track the cost (loss) function during iterations.

5. **Gradient Descent Loop**: 
   ```cpp
   for (int iter = 0; iter < num_iterations; ++iter)
   ```
   This loop iterates over the specified number of iterations for gradient descent optimization.

6. **Parallelization Directive**: 
   ```cpp
   #pragma omp parallel for reduction(+:sum_error0, sum_error1)
   ```
   This directive parallelizes the computation of gradient updates. The `reduction` clause ensures that the variables `sum_error0` and `sum_error1` are correctly updated across threads.

7. **Gradient Calculation**: 
   ```cpp
   double prediction = theta0 + theta1 * X[i];
   double error = prediction - y[i];
   sum_error0 += error;
   sum_error1 += error * X[i];
   ```
   These lines calculate the prediction error and accumulate the error gradients for `theta0` and `theta1`.

8. **Parameter Update**: 
   ```cpp
   theta0 -= (alpha / m) * sum_error0;
   theta1 -= (alpha / m) * sum_error1;
   ```
   The parameters `theta0` and `theta1` are updated based on the computed gradients and the learning rate.

9. **Cost Calculation**: 
   ```cpp
   cost = 0.0;
   for (int i = 0; i < m; ++i) {
       double prediction = theta0 + theta1 * X[i];
       cost += (prediction - y[i]) * (prediction - y[i]);
   }
   cost /= (2 * m);
   ```
   This section calculates the cost function (mean squared error) for monitoring convergence. It's optional and can be removed if not needed.

10. **Cost Output**: 
    ```cpp
    cout << "Iteration " << iter + 1 << ", Cost: " << cost << endl;
    ```
    This line outputs the current iteration number and cost for monitoring the convergence of the gradient descent algorithm.

11. **Main Function**: 
    ```cpp
    int main()
    ```
    This is the entry point of the program.

12. **Example Dataset**: 
    ```cpp
    vector<double> X = {1, 2, 3, 4, 5};
    vector<double> y = {3, 5, 7, 9, 11};
    ```
    These vectors represent the input features `X` and the corresponding target values `y`.

13. **Initial Parameters and Hyperparameters**: 
    ```cpp
    double theta0 = 0.0;
    double theta1 = 0.0;
    double alpha = 0.01;
    int num_iterations = 1000;
    ```
    These variables define the initial parameters `theta0` and `theta1`, the learning rate `alpha`, and the number of iterations for gradient descent.

14. **Function Call**: 
    ```cpp
    linear_regression_parallel(X, y, theta0, theta1, alpha, num_iterations);
    ```
    This line calls the `linear_regression_parallel` function to perform linear regression using gradient descent with the specified parameters.

15. **Output**: 
    ```cpp
    cout << "Final parameters: theta0 = " << theta0 << ", theta1 = " << theta1 << endl;
    ```
    This line outputs the final parameters `theta0` and `theta1` after the optimization process.

In summary, this code performs linear regression using gradient descent in parallel. It iteratively updates the parameters `theta0` and `theta1` to minimize the mean squared error between the predicted and actual target values. The OpenMP directive is used to parallelize the computation of gradient updates, improving efficiency by utilizing multiple threads.

