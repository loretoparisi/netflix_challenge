#ifndef RBM_HH
#define RBM_HH

#include <armadillo>

#include "singlealgorithm.hh"
#include "netflix.hh"

#define EPSILON 0.001
#define HIDDEN 100

using namespace arma;
using namespace netflix;

// TODO: add momentum

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

    // Weights between the hidden & visible units (W)
    fcube weights;

    // Shared biases of visible units
    fmat visibleBias;

    // Shared biases of the hidden units
    frowvec hiddenBias;

public:
    RBM(int users, int movies, int hidden, float rate);
    ~RBM();

    void train(const imat &data);
    float predict(int user, int item, int date);
};

#endif