#ifndef MLALGORITHM_HH
#define MLALGORITHM_HH

#include <armadillo>

using namespace arma;

class MLAlgorithm 
{
public:
    virtual void train(const imat &data);
    virtual float predict(int user, int item);
    
    virtual ~MLAlgorithm() {}
};

#endif // MLALGORITHM_HH
