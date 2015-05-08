/*
 * This class allows us to carry out combination algorithms
 * by combining TWO single algorithms.
 */

#ifndef TWO_ALGO_HH
#define TWO_ALGO_HH

#include <algorithm>
#include <armadillo>
#include <array>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iomanip>
#include <netflix.hh>
#include <comboalgorithm.hh>

using namespace arma;
using namespace netflix; // challenge-related constants/functions.
using std::cout;
using std::endl;

class Two_Algo : public ComboAlgorithm
{
    private:
        fmat currentTrain;
        
        /*
         * The file name where intermediate qual predictions will be stored
         * (i.e. the unbounded predictions made by the first algorithm).
         */
        std::string intermediatePredFileName;

        int ratingSigFig;

    public:
        Two_Algo(const std::string &trainingSet,
                 const std::string &intermediatePredFileName,
                 const int ratingSigFig);

        /* Train on the first model normally. */
        void trainFirst(SingleAlgorithm &firstAlgo);
        
        /* 
         * Save the first algorithm's qual predictions to 
         * intermediatePredFileName.
         */
        void saveFirstQualPredictions(SingleAlgorithm &firstAlgo,
                const std::string &qualFileName);
         
        /* Update the fmat for the residuals of training set. */
        void computeFirstResiduals(SingleAlgorithm &firstAlgo);

        /* Return the current average for the training set. */
        float getAverage();

        /* Save the residuals to a file. */
        void saveResiduals(const std::string residualsFile);

        /* Train on the second model with residuals. */
        void trainSecond(SingleAlgorithm &secondAlgo);

        /* Output the combined algorithms' predictions on qual. */
        void saveSecondQualPredictions(SingleAlgorithm &secondAlgo,
                const std::string &qualFileName,
                const std::string &outputFileName);
        
        ~Two_Algo();
};

#endif // TWO_ALGO_HH

