#ifndef MLALGORITHM_HH
#define MLALGORITHM_HH

#include <armadillo>

using namespace arma;

class MLAlgorithm 
{
public:
    virtual void train(const imat &data) = 0;
    virtual float predict(int user, int item) = 0;
    
    virtual ~MLAlgorithm() {}
};

#endif // MLALGORITHM_HH
