#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Point structure representing a data point in a multidimensional space
struct Point {
    vector<double> coordinates;
    int label;

    Point(vector<double> coords, int lbl) : coordinates(coords), label(lbl) {}
};

// Function to calculate Euclidean distance between two points
double euclideanDistance(const vector<double>& p1, const vector<double>& p2) {
    double distance = 0.0;
    int dim = p1.size();
    for (int i = 0; i < dim; ++i) {
        distance += pow(p1[i] - p2[i], 2);
    }
    return sqrt(distance);
}

// Sequential KNN function
vector<int> knnSequential(const vector<Point>& dataset, const vector<double>& query, int k) {
    vector<pair<double, int>> distances; // Pair of distance and index
    for (size_t i = 0; i < dataset.size(); ++i) {
        double distance = euclideanDistance(dataset[i].coordinates, query);
        distances.push_back({distance, i});
    }

    sort(distances.begin(), distances.end()); // Sort distances in ascending order

    vector<int> nearest_labels;
    for (int i = 0; i < k; ++i) {
        nearest_labels.push_back(dataset[distances[i].second].label);
    }

    return nearest_labels;
}

// Parallel KNN function
vector<int> knnParallel(const vector<Point>& dataset, const vector<double>& query, int k) {
    vector<pair<double, int>> distances; // Pair of distance and index

    #pragma omp parallel for
    for (size_t i = 0; i < dataset.size(); ++i) {
        double distance = euclideanDistance(dataset[i].coordinates, query);
        #pragma omp critical
        distances.push_back({distance, i});
    }

    sort(distances.begin(), distances.end()); // Sort distances in ascending order

    vector<int> nearest_labels;
    for (int i = 0; i < k; ++i) {
        nearest_labels.push_back(dataset[distances[i].second].label);
    }

    return nearest_labels;
}

int main() {
    // Generate sample dataset with 10000 points in 2-dimensional space
    vector<Point> dataset;
    srand(time(NULL)); // Seed for random number generation
    for (int i = 0; i < 1000000; ++i) {
        vector<double> coordinates = {double(rand() % 1000), double(rand() % 100)};
        int label = rand() % 2; // Random label 0 or 1
        dataset.push_back(Point(coordinates, label));
    }

    // Set a query point in the same 2-dimensional space
    vector<double> query = {50, 50}; // Query point
    int k = 5; // Number of nearest neighbors to find

    // Sequential execution
    auto start_seq = high_resolution_clock::now();
    vector<int> nearest_labels_seq = knnSequential(dataset, query, k);
    auto stop_seq = high_resolution_clock::now();
    auto duration_seq = duration_cast<milliseconds>(stop_seq - start_seq);
    cout << "Sequential Execution Time: " << duration_seq.count() << " milliseconds" << endl;

    // Parallel execution
    auto start_par = high_resolution_clock::now();
    vector<int> nearest_labels_par = knnParallel(dataset, query, k);
    auto stop_par = high_resolution_clock::now();
    auto duration_par = duration_cast<milliseconds>(stop_par - start_par);
    cout << "Parallel Execution Time: " << duration_par.count() << " milliseconds" << endl;

    // Output the results
    cout << "Nearest Labels (Sequential): ";
    for (int label : nearest_labels_seq) {
        cout << label << " ";
    }
    cout << endl;

    cout << "Nearest Labels (Parallel): ";
    for (int label : nearest_labels_par) {
        cout << label << " ";
    }
    cout << endl;

    return 0;
}
