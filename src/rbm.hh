#ifndef __RBM_HH__
#define __RBM_HH__

#include <armadillo>
#include <cmath>
#include <functional>

#include <singlealgorithm.hh>
#include <netflix.hh>

#define HIDDEN 32
#define EPSILON 0.001
#define MOMENTUM 0.9
#define DELTA 0.00002

using namespace arma;
using namespace netflix;

typedef float data_t;

// Type for storing rating information about a movie
struct rating_t {
    unsigned movie : 32;
    unsigned score : 8;
    unsigned softmax : 8;
};

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

template <typename T>
inline T sigmoid(const T &x) {
    return 1.0 / (1.0 + exp(-1.0 * x));
}

class RBM : public SingleAlgorithm {
private:

    // TODO: different learning rates for weights & biases, dynamic momentum

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
    data_t **weights;

    // Shared biases of visible units
    data_t *visibleBias;

    // Shared biases of the hidden units
    data_t *hiddenBias;

public:
    RBM(int users, int movies, int hidden, float rate, float momentum);
    ~RBM();

    void train(const Mat<data_t> &data);
    // void train(const Mat<data_t> &data, const Mat<data_t> &probe);
    float predict(int user, int item, int date) { return 0.0; }
    Mat<data_t> predict(const Mat<data_t> &targets);
    Mat<data_t> predict(const std::string &targetPath);
};

#endif // __RBM_HH__