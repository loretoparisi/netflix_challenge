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

// A temporary file where intermediate qual predictions (made by the
// unbounded first algorithm) will be saved. These are stored in plain-text
// format.
const string INTERMED_PRED_FILE = "data/knn_ge_intermed_pred_temp.dta";

// If this is true, then we will delete the intermediate prediction file
// mentioned above.
const bool DELETE_INTERMED_PRED_FILE = false;

// The file where we'll store the residuals of the first model on the
// training set, in Armadillo's binary format. If this is uninitialized
// (i.e. the string is empty()), then the residuals will not be saved.
const string RESIDUALS_FILE = "data/knn_ge_resid.mat";

// Whether we've cached the residuals of the first model, as well as the
// intermediate qual predictions it generated (at the above-mentioned
// locations). If so, we can avoid training the first model again.
const bool CACHED_FIRST_MODEL = true;

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
    Two_Algo *combine;
    
    // Setting up the first model.
    if (CACHED_FIRST_MODEL)
    {
        // We really shouldn't delete the intermediate prediction file
        // afterwards in this case...
        if (DELETE_INTERMED_PRED_FILE)
        {
            throw logic_error("You shouldn't delete intermediate "
                    "predictions if the first model is cached.");
        }

        // Just construct the Two_Algo with the given residuals.
        combine = new Two_Algo(RESIDUALS_FILE, INTERMED_PRED_FILE,
                               RATING_SIG_FIGS, DELETE_INTERMED_PRED_FILE);
        cout << "\nTwo_Algo is using cached residuals of the first model."
            << endl;
    }
    else
    {
        // Construct the Two_Algo to work with the training set.
        combine = new Two_Algo(TRAIN_UM, INTERMED_PRED_FILE,
                               RATING_SIG_FIGS, DELETE_INTERMED_PRED_FILE);

        Globals predAlgoGE(NUM_USERS, NUM_MOVIES, LEVEL, TRAIN_MU);
    
        combine->trainFirst(predAlgoGE);
        combine->saveFirstQualPredictions(predAlgoGE, QUAL_DATA_FN);
        combine->computeAndSaveFirstResiduals(predAlgoGE, RESIDUALS_FILE);
    }
    
    // Setting up the second model and outputting predictions.
    {
        KNN predAlgoKNN(NUM_USERS, NUM_MOVIES, MIN_COMMON, MAX_WEIGHT, 
                        LOAD_P, SAVE_P, P_FN);

        combine->trainSecond(predAlgoKNN);
        combine->saveSecondQualPredictions(predAlgoKNN, QUAL_DATA_FN,
                                           OUTPUT_FILENAME);
    }

    delete combine;
}
