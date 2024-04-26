#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Function to perform linear regression
void linearRegression(const vector<double>& x, const vector<double>& y, double& slope, double& intercept) {
    int n = x.size();
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;

    for (int i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xx += x[i] * x[i];
        sum_xy += x[i] * y[i];
    }

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    intercept = (sum_y - slope * sum_x) / n;
}

// Parallel implementation of linear regression
void linearRegressionParallel(const vector<double>& x, const vector<double>& y, double& slope, double& intercept) {
    int n = x.size();
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;

    #pragma omp parallel for reduction(+:sum_x,sum_y,sum_xx,sum_xy)
    for (int i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xx += x[i] * x[i];
        sum_xy += x[i] * y[i];
    }

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    intercept = (sum_y - slope * sum_x) / n;
}

int main() {
    // Generate some sample data
    int n = 1000000;
    vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i;
        y[i] = 2 * i + 3 + (rand() % 100) / 100.0; // adding some noise
    }

    // Sequential execution
    auto start = high_resolution_clock::now();
    double slope_seq, intercept_seq;
    linearRegression(x, y, slope_seq, intercept_seq);
    auto stop = high_resolution_clock::now();
    auto duration_seq = duration_cast<milliseconds>(stop - start);
    cout << "Sequential Execution Time: " << duration_seq.count() << " milliseconds" << endl;

    // Parallel execution
    start = high_resolution_clock::now();
    double slope_par, intercept_par;
    linearRegressionParallel(x, y, slope_par, intercept_par);
    stop = high_resolution_clock::now();
    auto duration_par = duration_cast<milliseconds>(stop - start);
    cout << "Parallel Execution Time: " << duration_par.count() << " milliseconds" << endl;

    // Comparing results
    cout << "Slope (Sequential): " << slope_seq << ", Intercept (Sequential): " << intercept_seq << endl;
    cout << "Slope (Parallel): " << slope_par << ", Intercept (Parallel): " << intercept_par << endl;

    return 0;
}
