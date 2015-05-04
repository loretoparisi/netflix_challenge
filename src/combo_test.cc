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
#include <timesvdpp.hh>

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

// The number of factors to use for Time-SVD++.
const int NUM_FACTORS = 10;

// The number of iterations of Time-SVD++ to carry out.
const int NUM_ITERATIONS = 5;

// The number of time bins to use for movies in Time-SVD++.
// BellKor used 30 so we will too for now.
const int NUM_TIME_BINS = 30;

// Original output qual file for the TimeSVD++.
const string TIMESVD_QUAL = "data/good_predictions/TIMESVDPP_QUAL_6.963.dta";

// Final output file for this combo algorithm.
const string OUTPUT_FILENAME = "data/combine_test.dta";

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

    TimeSVDPP predAlgoTimeSVD(NUM_USERS, NUM_MOVIES, NUM_DATES,
                              newAverage, NUM_FACTORS,
                              NUM_ITERATIONS, NUM_TIME_BINS,
                              N_FN, HAT_DEV_U_T_FN);

    combine.trainSecond(predAlgoTimeSVD);
    combine.outputQual(predAlgoTimeSVD,
        QUAL_DATA_FN, TIMESVD_QUAL, OUTPUT_FILENAME);
    cout << "Output is in " << OUTPUT_FILENAME << " ." << endl;
}

