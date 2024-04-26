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
    int cluster;

    Point(vector<double> coords) : coordinates(coords), cluster(0) {}
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

// Function to find the index of the nearest centroid for a point
int findNearestCentroid(const Point& point, const vector<vector<double>>& centroids) {
    int nearestCentroid = 0;
    double minDistance = euclideanDistance(point.coordinates, centroids[0]);

    for (size_t i = 1; i < centroids.size(); ++i) {
        double distance = euclideanDistance(point.coordinates, centroids[i]);
        if (distance < minDistance) {
            minDistance = distance;
            nearestCentroid = i;
        }
    }

    return nearestCentroid;
}

// Function to update the centroids based on the current clusters
void updateCentroids(vector<Point>& points, vector<vector<double>>& centroids) {
    for (size_t i = 0; i < centroids.size(); ++i) {
        vector<double> newCentroid(centroids[i].size(), 0.0);
        int count = 0;

        for (const Point& point : points) {
            if (point.cluster == i) {
                for (size_t j = 0; j < point.coordinates.size(); ++j) {
                    newCentroid[j] += point.coordinates[j];
                }
                count++;
            }
        }

        if (count != 0) {
            for (size_t j = 0; j < newCentroid.size(); ++j) {
                newCentroid[j] /= count;
            }
            centroids[i] = newCentroid;
        }
    }
}

// Function to perform K-means clustering sequentially
void kmeansSequential(vector<Point>& points, int k) {
    // Randomly initialize centroids
    vector<vector<double>> centroids(k, vector<double>(points[0].coordinates.size()));
    for (size_t i = 0; i < centroids.size(); ++i) {
        centroids[i] = points[i].coordinates;
    }

    bool centroidsChanged = true;
    while (centroidsChanged) {
        centroidsChanged = false;

        // Assign each point to the nearest centroid
        for (Point& point : points) {
            int nearestCentroid = findNearestCentroid(point, centroids);
            if (point.cluster != nearestCentroid) {
                point.cluster = nearestCentroid;
                centroidsChanged = true;
            }
        }

        // Update centroids based on the current clusters
        updateCentroids(points, centroids);
    }
}

// Function to perform K-means clustering in parallel
void kmeansParallel(vector<Point>& points, int k) {
    // Randomly initialize centroids
    vector<vector<double>> centroids(k, vector<double>(points[0].coordinates.size()));
    for (size_t i = 0; i < centroids.size(); ++i) {
        centroids[i] = points[i].coordinates;
    }

    bool centroidsChanged = true;
    while (centroidsChanged) {
        centroidsChanged = false;

        // Assign each point to the nearest centroid in parallel
        #pragma omp parallel for
        for (size_t i = 0; i < points.size(); ++i) {
            int nearestCentroid = findNearestCentroid(points[i], centroids);
            #pragma omp critical
            if (points[i].cluster != nearestCentroid) {
                points[i].cluster = nearestCentroid;
                centroidsChanged = true;
            }
        }

        // Update centroids based on the current clusters
        updateCentroids(points, centroids);
    }
}

int main() {
    // Generate sample dataset with 1000 points in 2-dimensional space
    vector<Point> points;
    srand(time(NULL)); // Seed for random number generation
    for (int i = 0; i < 100000; ++i) {
        vector<double> coordinates = {double(rand() % 1000), double(rand() % 100)};
        points.push_back(Point(coordinates));
    }

    int k = 5; // Number of clusters

    auto start = high_resolution_clock::now();

    // Sequential execution
    kmeansSequential(points, k);
    auto stop_seq = high_resolution_clock::now();
    auto duration_seq = duration_cast<milliseconds>(stop_seq - start);
    cout << "Sequential Execution Time: " << duration_seq.count() << " milliseconds" << endl;

    // Reset points for parallel execution
    for (Point& p : points) {
        p.cluster = 0; // Reset cluster assignments
    }

    start = high_resolution_clock::now();

    // Parallel execution
    #pragma omp parallel
    {
        kmeansParallel(points, k);
    }

    auto stop_par = high_resolution_clock::now();
    auto duration_par = duration_cast<milliseconds>(stop_par - start);
    cout << "Parallel Execution Time: " << duration_par.count() << " milliseconds" << endl;

    return 0;
}
