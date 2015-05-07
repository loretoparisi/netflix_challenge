/** 
 * A simple test to check if Global Effect is working. Note that this class
 * assumes that data is in the (user, movie) order.
 *
 * Note: The main method does not expect any arguments in this case.
 *
 */

#include <armadillo>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <netflix.hh>
#include <two_algo.hh>
#include <globals.hh>
#include <knn.hh>

using namespace std;
using namespace arma;
using namespace netflix; // challenge-related constants/functions.

// The Armadillo binary file to use for training.
// Make sure that both UM and MU data are using the same
// "type" of training resource (eg. probe, hidden, base, etc.)
const string TRAIN_UM = VALID_BIN;

// MU train matrix for global effect. Note that
// the index type should match that of TRAIN_UM.
const string TRAIN_MU = "data/valid-mu.mat";

// The "level" of global effect we want to train on.
// (See globals_README in "data" dir for more detail)
const int level = 10;

// Sig-figs for output file.
const int RATING_SIG_FIGS = 4;

// Minimum common neighbors required for decent prediction.
const int MIN_COMMON = 24;

// Max weight elements to consider when predicting.
const unsigned int MAX_WEIGHT = 30;

// Original output qual file for the KNN.
const string KNN_QUAL = "data/knn_cached/knn_test_predictions.dta";

// Pearson value file path.
const string P_FN = "data/knn_cached/knn-p.dta";

// Final output file for this combo algorithm.
const string OUTPUT_FILENAME = "data/globals_knn_test.dta";

int main(void)
{
    Two_Algo combine(TRAIN_UM, RATING_SIG_FIGS);
    cout << "Loaded training data from " << TRAIN_UM << "." << endl;

    Globals predAlgoGE(NUM_USERS, NUM_MOVIES, level, TRAIN_MU);

    combine.trainFirst(predAlgoGE);

    combine.firstResiduals(predAlgoGE);

    float newAverage = combine.getAverage();

    cout << "New average is: " << newAverage << endl;

    combine.saveResiduals("data/residualStore.dta");

    KNN predAlgoKNN(NUM_USERS, NUM_MOVIES, MIN_COMMON, MAX_WEIGHT, P_FN);

    combine.trainSecond(predAlgoKNN);
    combine.outputQual(predAlgoKNN,
        QUAL_DATA_FN, KNN_QUAL, OUTPUT_FILENAME);
    cout << "Output is in " << OUTPUT_FILENAME << " ." << endl;
}

