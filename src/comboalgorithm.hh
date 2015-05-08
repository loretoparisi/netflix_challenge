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

        /* Compute the residuals of the first model on training set. */
        virtual void computeFirstResiduals(SingleAlgorithm &firstAlgo) = 0;

        /* Return the current average for the training set. */
        virtual float getAverage() = 0;
        
        /* 
         * Save the residuals to a file. This must be called after
         * computeFirstResiduals() so that the appropriate residuals are
         * saved to "currentTrain".
         */
        virtual void saveResiduals(const std::string residualsFile) = 0;

        /* Train on the second model with residuals. */
        virtual void trainSecond(SingleAlgorithm &secondAlgo) = 0;

        /* 
         * Save the first algorithm's qual predictions to 
         * an intermediate file.
         */
        virtual void saveFirstQualPredictions(SingleAlgorithm &firstAlgo,
                const std::string &qualFileName) = 0;

        /* Output the combined algorithms' predictions on qual. */
        virtual void saveSecondQualPredictions(SingleAlgorithm &secondAlgo,
                const std::string &qualFileName,
                const std::string &outputFileName) = 0;
        
        virtual ~ComboAlgorithm() {}
};

#endif // COMBOALGORITHM_HH
