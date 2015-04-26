#ifndef __RBM_HH__
#define __RBM_HH__

#include <armadillo>
#include <cmath>
#include <functional>

#include <singlealgorithm.hh>
#include <netflix.hh>

#define HIDDEN 100
#define EPSILON 0.001
#define MOMENTUM 0.9

using namespace arma;
using namespace netflix;

typedef float data_t;
typedef unsigned char ind_t;

template <typename T, typename K>
int binary_search(const T &data, K key, std::function<int(T, K, int)> probe, 
                  int min, int max) {
    // Midpoint of the range
    int mid;
    // Direction returned by the probe; must be one of -1, 0, or 1
    int direction;
    // While the range [min, max] is not empty
    while ( max >= min ) {
        // Calculate the middle index
        mid = (max + min) / 2;
        // Probe for direction
        direction = probe(data, key, mid);
        // Probe the data to determine which half of the range the key is in
        switch ( direction ) {
            // The key is in (mid, max]
            case 1:
                min = mid + 1;
                break;
            // The key is at the current index
            case 0:
                return mid;
            // The key is in [min, mid)
            case -1:
                max = mid - 1;
                break;
            default:
                throw std::logic_error("Probe returned invalid direction");
        }
    }
    // The item was not found; return an invalid index
    return -1;
}

// Convert a probability matrix to a random Bernoulli matrix
template<typename T>
Mat<ind_t> randomBernoulli(Mat<T> probabilities) {
    // Generate a uniform random matrix w/ values in the range [0, 1]
    Mat<T> uniform = randu<Mat<T>>(probabilities.n_rows, probabilities.n_cols);
    // Bernoulli matrix
    Mat<ind_t> bernoulli(probabilities.n_rows, probabilities.n_cols, 
                           fill::zeros);
    // Set randomly selected elements to one in the Bernoulli matrix
    bernoulli.elem(find(probabilities > uniform)).ones();

    return bernoulli;
}

template <typename T>
T sigmoid(const T &x) {
    return 1 / (1 + exp(-1 * x));
}

class RBM : public SingleAlgorithm {
private:
    // Number of users in the data set
    int users;

    // Number of movies in the data set
    int movies;

    // Number of hidden units
    int hidden;

    // Learning rate
    float rate;

    // Momentum
    float momentum;

    // Weights between the hidden & visible units (W)
    Cube<data_t> weights;

    // Shared biases of visible units
    Mat<data_t> visibleBias;

    // Shared biases of the hidden units
    Col<data_t> hiddenBias;

public:
    RBM(int users, int movies, int hidden, float rate, float momentum);
    ~RBM();

    void train(const Mat<data_t> &data);
    float predict(int user, int item, int date);
};

#endif // __RBM_HH__