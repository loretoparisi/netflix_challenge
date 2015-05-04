#ifndef COMBOALGORITHM_HH
#define COMBOALGORITHM_HH

#include <armadillo>
#include <singlealgorithm.hh>

using namespace arma;

class ComboAlgorithm 
{
    public:
        /* Train on the first model regularly. */
        virtual void trainFirst(SingleAlgorithm &predAlgo) = 0;

        /* Output residuals of the first model on training set.
         * This should save the output to the computer to
         * conserve space.
         */
        virtual void firstResiduals(SingleAlgorithm &predAlgo) = 0;

        /* Return the current average for the training set. */
        virtual float getAverage() = 0;

        /* Save the residuals to a file. */
        virtual void saveResiduals(const std::string residualsFile) = 0;

        /* Train on the second model with residuals. */
        virtual void trainSecond(SingleAlgorithm &predAlgo) = 0;

        /* Output predicted residuals to qual. */
        virtual void outputQual(SingleAlgorithm &predAlgo,
            const std::string &testFileName,
            const std::string &previousOutputName,
            const std::string &newOutputFileName) = 0;

        virtual ~ComboAlgorithm() {}
};

#endif // COMBOALGORITHM_HH
