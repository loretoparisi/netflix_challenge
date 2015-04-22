#ifndef RBM_HH
#define RBM_HH

#include <armadillo>
#include <functional>

#include <singlealgorithm.hh>
#include <netflix.hh>

#define HIDDEN 100
#define EPSILON 0.001
#define MOMENTUM 0.9

using namespace arma;
using namespace netflix;

template <typename T, typename K>
inline static int binary_search(
    const T &data, K key, std::function<int(T, K, int)> probe, int min, 
    int max) {
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
    fcube weights;

    // Shared biases of visible units
    fmat visibleBias;

    // Shared biases of the hidden units
    frowvec hiddenBias;

public:
    RBM(int users, int movies, int hidden, float rate, float momentum);
    ~RBM();

    void train(const fmat &data);
    // void train(const std::string &dataPath);
    float predict(int user, int item, int date);
};

#endif