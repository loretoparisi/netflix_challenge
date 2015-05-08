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
// const string TRAIN_UM = VALID_BIN;
const string TRAIN_UM = ALL_TRAIN_BIN;
// const string TRAIN_UM = BASE_BIN;

// MU train matrix for global effect. Note that
// the index type should match that of TRAIN_UM.
// const string TRAIN_MU = "data/valid-mu.mat";
const string TRAIN_MU = MU_ALL_TRAIN_BIN;
// const string TRAIN_MU = MU_BASE_BIN;

// The "level" of global effect we want to train on.
// (See globals_README in "data" dir for more detail)
const int LEVEL = 10;

// Sig-figs for output file.
const int RATING_SIG_FIGS = 4;

// Minimum common neighbors required for decent prediction.
const int MIN_COMMON = 24;

// Max weight elements to consider when predicting.
const unsigned int MAX_WEIGHT = 30;

// A temporary file name where intermediate qual predictions (made by the
// unbounded first algorithm) will be saved.
const string INTERMED_PRED_FILE = "data/intermed_pred_temp.dta";

// Whether we want to load P. If this is false, then P will be recomputed.
const bool LOAD_P = true;

// Whether we want to save P after computing it.
const bool SAVE_P = false;

// Pearson value file path.
const string P_FN = "data/knn_cached/knn-p.dta";

// Final output file for this combo algorithm.
const string OUTPUT_FILENAME = "data/globals_knn_combo_predictions.dta";


int main(void)
{
    Two_Algo combine(TRAIN_UM, INTERMED_PRED_FILE, RATING_SIG_FIGS);

    {
        Globals predAlgoGE(NUM_USERS, NUM_MOVIES, LEVEL, TRAIN_MU);
        
        combine.trainFirst(predAlgoGE);
        combine.saveFirstQualPredictions(predAlgoGE, QUAL_DATA_FN);
        combine.computeFirstResiduals(predAlgoGE);

        /*
        float newAverage = combine.getAverage();

        cout << "New average is: " << newAverage << endl;
        */
    }

    {
        KNN predAlgoKNN(NUM_USERS, NUM_MOVIES, MIN_COMMON, MAX_WEIGHT, 
                        LOAD_P, SAVE_P, P_FN);

        combine.trainSecond(predAlgoKNN);
        combine.saveSecondQualPredictions(predAlgoKNN, QUAL_DATA_FN,
                                          OUTPUT_FILENAME);
    }
}
