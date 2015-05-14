/** 
 * kNN on the residuals of Time-SVD++. Note that we're assuming that data
 * is in the (user, movie) order. The main method to this script does not
 * expect any arguments.
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
#include <timesvdpp.hh>
#include <knn.hh>

using namespace std;
using namespace arma;
using namespace netflix; // challenge-related constants/functions.

// The UM-ordered Armadillo binary file to use for training.
// const string TRAIN_UM = HIDDEN_BIN;
// const string TRAIN_UM = BASE_HIDDEN_VALID_BIN;
const string TRAIN_UM = ALL_TRAIN_BIN;

// The number of factors to use for Time-SVD++.
const int NUM_FACTORS = 60;

// The number of iterations of Time-SVD++ to carry out.
const int NUM_ITERATIONS = 40;

// The number of time bins to use for movies in Time-SVD++. BellKor used 30
// so we will too for now.
const int NUM_TIME_BINS = 30;

// Sig-figs for output file.
const int RATING_SIG_FIGS = 4;

// Whether we'll use userFacMatTime. Setting this to false saves a lot of
// RAM utilization, but might slightly impact accuracy.
const bool INCLUDE_USER_FAC_MAT_TIME = true;

// Minimum common neighbors required for decent prediction.
const int MIN_COMMON = 24;

// Max weight elements to consider when predicting.
const unsigned int MAX_WEIGHT = 400;

// A temporary file where intermediate qual predictions (made by the
// unbounded first algorithm) will be saved. These are stored in plain-text
// format.
const string INTERMED_PRED_FILE = "data/knn_timesvdpp_intermed_pred_"
                                  "temp.dta";

// If this is true, then we will delete the intermediate prediction file
// mentioned above.
const bool DELETE_INTERMED_PRED_FILE = false;

// The file where we'll store the residuals of the first model on the
// training set, in Armadillo's binary format. If this is uninitialized
// (i.e. the string is empty()), then the residuals will not be saved.
const string RESIDUALS_FILE = "data/knn_timesvdpp_resid.mat";

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

// The name of the output file to use (for predictions on "qual").
const string OUTPUT_FN = "data/knn_on_timesvdpp_predictions.dta";


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

        TimeSVDPP predAlgoTimeSVDPP(NUM_USERS, NUM_MOVIES, NUM_DATES,
                                    MEAN_RATING_TRAINING_SET, NUM_FACTORS,
                                    NUM_ITERATIONS, NUM_TIME_BINS,
                                    INCLUDE_USER_FAC_MAT_TIME,
                                    N_FN, HAT_DEV_U_T_FN, F_U_T_FN);
        
        combine->trainFirst(predAlgoTimeSVDPP);
        combine->saveFirstQualPredictions(predAlgoTimeSVDPP, QUAL_DATA_FN);
        combine->computeAndSaveFirstResiduals(predAlgoTimeSVDPP,
                                              RESIDUALS_FILE);
    }

    // Setting up the second model and outputting predictions.
    {
        KNN predAlgoKNN(NUM_USERS, NUM_MOVIES, MIN_COMMON, MAX_WEIGHT, 
                        LOAD_P, SAVE_P, P_FN);

        combine->trainSecond(predAlgoKNN);
        combine->saveSecondQualPredictions(predAlgoKNN, QUAL_DATA_FN,
                                           OUTPUT_FN);
    }

    delete combine;
}
