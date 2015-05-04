#ifndef COMBOALGORITHM_HH
#define COMBOALGORITHM_HH

#include <armadillo>

using namespace arma;

class ComboAlgorithm 
{
    public:
        /* Train on the first model regularly. */
        virtual void trainFirst(fmat &data) = 0;

        /* Output residuals of the first model on training set.
         * This should save the output to the computer to
         * conserve space.
         */
        virtual void firstResiduals(fmat &data) = 0;

        /* Train on the second model with residuals. */
        virtual void trainSecond(const fmat &data) = 0;

        /* Output predicted residuals to qual. */
        virtual void outputQual(const std::string originalQual,
            const std::string newQual) = 0;

        virtual ~ComboAlgorithm() {}
};

#endif // COMBOALGORITHM_HH
