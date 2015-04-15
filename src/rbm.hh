#ifndef RBM_HH
#define RBM_HH

#include <armadillo>

#include "mlalgorithm.hh"
#include "netflix.hh"

#define EPSILON 0.001
#define HIDDEN 100

using namespace arma;
using namespace netflix;

// TODO: add momentum, sparse cube class (operators?)

class RBM : public MLAlgorithm {
private:
    // Weights between the hidden & visible units (W)
    fcube weights;

    // Shared biases of visible units
    fmat visibleBias;

    // Shared biases of the hidden units
    frowvec hiddenBias;

    // Number of users in the data set
    int users;

    // Number of movies in the data set
    int movies;

    // Number of hidden units
    int hidden;

    // Learning rate
    float rate;

public:
    RBM(int users, int movies, int hidden, float rate);
    ~RBM();

    void train();
    void train(const imat &data);
    float predict(int user, int item);
};

#endif