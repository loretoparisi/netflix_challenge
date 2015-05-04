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
        int ratingSigFig;

    public:
        Two_Algo(const std::string &trainingSet,
            const int ratingSigFig);

        /* Train on the first model normally. */
        void trainFirst(SingleAlgorithm &predAlgo);

        /* Update the fmat for the residuals of training set. */
        void firstResiduals(SingleAlgorithm &predAlgo);

        /* Return the current average for the training set. */
        float getAverage();

        /* Save the residuals to a file. */
        void saveResiduals(const std::string residualsFile);

        /* Train on the second model with residuals. */
        void trainSecond(SingleAlgorithm &predAlgo);

        /* Output predicted residuals to qual. */
        void outputQual(SingleAlgorithm &predAlgo,
            const std::string &testFileName,
            const std::string &previousOutputName,
            const std::string &newOutputFileName);

        ~Two_Algo();
};

#endif // TWO_ALGO_HH

