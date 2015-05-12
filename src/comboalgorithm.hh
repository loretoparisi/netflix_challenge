#ifndef COMBOALGORITHM_HH
#define COMBOALGORITHM_HH

#include <armadillo>
#include <basealgorithm.hh>

using namespace arma;

class ComboAlgorithm 
{
    public:
        /* Train on the first model regularly. */
        virtual void trainFirst(BaseAlgorithm &predAlgo) = 0;

        /* Compute the residuals of the first model on training set. */
        virtual void computeFirstResiduals(BaseAlgorithm &firstAlgo) = 0;

        /* Return the current average for the training set. */
        virtual float getAverage() = 0;
        
        /* 
         * Save the residuals to a file. This must be called after
         * computeFirstResiduals() so that the appropriate residuals are
         * saved to "currentTrain".
         */
        virtual void saveResiduals(const std::string residualsFile) = 0;

        /* Train on the second model with residuals. */
        virtual void trainSecond(BaseAlgorithm &secondAlgo) = 0;

        /* 
         * Save the first algorithm's qual predictions to 
         * an intermediate file.
         */
        virtual void saveFirstQualPredictions(BaseAlgorithm &firstAlgo,
                const std::string &qualFileName) = 0;

        /* Output the combined algorithms' predictions on qual. */
        virtual void saveSecondQualPredictions(BaseAlgorithm &secondAlgo,
                const std::string &qualFileName,
                const std::string &outputFileName) = 0;
        
        virtual ~ComboAlgorithm() {}
};

#endif // COMBOALGORITHM_HH
