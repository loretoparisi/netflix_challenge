#ifndef RBM_HH
#define RBM_HH

#include <armadillo>

#include "mlalgorithm.hh"
#include "netflix.hh"

#define HIDDEN 100

using namespace arma;
using namespace netflix;

class RBM : public MLAlgorithm {
private:
    fcube::fixed<NUM_MOVIES, HIDDEN, MAX_RATING> weights;
    fmat::fixed<NUM_MOVIES, MAX_RATING> visibleBias;
    frowvec::fixed<HIDDEN> hiddenBias;
    float rate;

public:
    RBM(float rate);
    ~RBM();

    void train(const imat &data);
    float predict(int user, int item);
};

#endif