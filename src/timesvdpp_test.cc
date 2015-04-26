/** 
 * A simple test to check if Time-SVD++ is working. Note that this class
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
#include <timesvdpp.hh>

using namespace std;
using namespace arma;
using namespace netflix; // challenge-related constants/functions.


/* Constants */

// The Armadillo binary file to use for training.
//const string TIMESVDPP_TRAIN_FILE = HIDDEN_BIN;
const string TIMESVDPP_TRAIN_FILE = BASE_HIDDEN_VALID_BIN;

// The number of factors to use for Time-SVD++.
const int NUM_FACTORS = 200;

// The number of iterations of Time-SVD++ to carry out.
const int NUM_ITERATIONS = 30;

// The number of time bins to use for movies in Time-SVD++. BellKor used 30
// so we will too for now.
const int NUM_TIME_BINS = 30;

// The name of the output file to use (for predictions on "qual").
const string OUTPUT_FN = "data/timesvdpp_predictions.dta";

// Sig-figs for output file.
const int RATING_SIG_FIGS = 4;

// Whether the data will be cached after training.
const bool WILL_CACHE_DATA = false;

// Whether we're using cached data **instead of** training.
const bool USING_CACHED_DATA = true;

// The locations of the files we'll use for caching (and read from if we're
// using cached data). These must be in Armadillo binary format!
const string B_USER_CONST_FN =          "data/timesvdpp_cached/"
                                        "b_user_const.mat";
const string B_USER_ALPHA_FN =          "data/timesvdpp_cached/"
                                        "b_user_alpha.mat";
const string B_USER_TIME_FN =           "data/timesvdpp_cached/"
                                        "b_user_time.mat";
const string B_ITEM_CONST_FN =          "data/timesvdpp_cached/"
                                        "b_item_const.mat";
const string B_ITEM_TIMEWISE_FN =       "data/timesvdpp_cached/"
                                        "b_item_timewise.mat";
const string USER_FAC_MAT_FN =          "data/timesvdpp_cached/"
                                        "user_fac.mat";
const string USER_FAC_MAT_ALPHA_FN =    "data/timesvdpp_cached/"
                                        "user_fac_alpha.mat";
const string ITEM_FAC_MAT_FN =          "data/timesvdpp_cached/"
                                        "item_fac.mat";
const string Y_MAT_FN =                 "data/timesvdpp_cached/y.mat";
const string SUM_MOVIE_WEIGHTS_FN =     "data/timesvdpp_cached/"
                                        "user_sum_y.mat";

// Helper function that carries out "predAlgo" on the test file specified
// by testFileName, and then puts the prediction results (for each (user,
// item, time) in testFileName) in outputFileName.
void testOnDataFile(TimeSVDPP &predAlgo, const string &testFileName,
                    const string &outputFileName);

// Helper function that carries out "predAlgo" on the test file specified
// by testFileName (most likely the "probe" dataset), and calculates the
// RMSE based on those results. Note that testFileName must refer to an
// Armadillo binary in this case.
float computeRMSE(TimeSVDPP &predAlgo, const string &testFileName);


int main(void)
{
    // Sanity check on arguments.
    if (USING_CACHED_DATA && WILL_CACHE_DATA)
    {
        throw logic_error("It doesn't make sense to set the \"will cache "
                          "data\" flag if you're using cached data!");
    }
    
    // If we've cached the matrices produced by Time-SVD++, use those to
    // skip the training step.
    if (USING_CACHED_DATA)
    {
        // Construct a TimeSVDPP object and pass in the cached data file
        // names.
        TimeSVDPP predAlgo(NUM_USERS, NUM_MOVIES, NUM_DATES,
                           MEAN_RATING_TRAINING_SET, NUM_FACTORS,
                           NUM_ITERATIONS, NUM_TIME_BINS,
                           N_FN, HAT_DEV_U_T_FN,
                           B_USER_CONST_FN, B_USER_ALPHA_FN,
                           B_USER_TIME_FN,
                           B_ITEM_CONST_FN, B_ITEM_TIMEWISE_FN,
                           USER_FAC_MAT_FN, USER_FAC_MAT_ALPHA_FN,
                           ITEM_FAC_MAT_FN, Y_MAT_FN,
                           SUM_MOVIE_WEIGHTS_FN);
         
        // Go through qual.dta and produce a prediction file.
        testOnDataFile(predAlgo, QUAL_DATA_FN, OUTPUT_FN);

        // Get probe RMSE.
        float probeRMSE = computeRMSE(predAlgo, PROBE_BIN);

        cout << "\nProbe RMSE: " << probeRMSE << endl;
    }
    else // If not using cached data, we need to train.
    {
        // Load from binary.
        fmat trainingSet;
        trainingSet.load(TIMESVDPP_TRAIN_FILE, arma_binary);
        
        cout << "Loaded training data from " << TIMESVDPP_TRAIN_FILE << "."
            << endl;

        TimeSVDPP predAlgo(NUM_USERS, NUM_MOVIES, NUM_DATES,
                           MEAN_RATING_TRAINING_SET, NUM_FACTORS,
                           NUM_ITERATIONS, NUM_TIME_BINS,
                           N_FN, HAT_DEV_U_T_FN);
        
        // Check if we want to cache.
        if (WILL_CACHE_DATA)
        {
            // If so, train and then cache the results after training.
            cout << "\nTraining Time-SVD++. The resulting matrices will be"
                    " cached." << endl;
            
            predAlgo.trainAndCache(trainingSet,
                                   B_USER_CONST_FN, B_USER_ALPHA_FN,
                                   B_USER_TIME_FN,
                                   B_ITEM_CONST_FN, B_ITEM_TIMEWISE_FN,
                                   USER_FAC_MAT_FN, USER_FAC_MAT_ALPHA_FN,
                                   ITEM_FAC_MAT_FN, Y_MAT_FN,
                                   SUM_MOVIE_WEIGHTS_FN);
        }
        else
        {
            // If not, just train.
            cout << "\nTraining Time-SVD++. The resulting matrices won't "
                    "be cached." << endl;
            predAlgo.train(trainingSet);
        }        
        
        // Go through qual.dta to produce a prediction file.
        testOnDataFile(predAlgo, QUAL_DATA_FN, OUTPUT_FN);

        // Get probe RMSE.
        float probeRMSE = computeRMSE(predAlgo, PROBE_BIN);
        
        cout << "\nProbe RMSE: " << probeRMSE << endl;
    }

}


/**
 * Compute the RMSE of a given prediction algorithm on a certain set of
 * data. Note that testFileName must refer to an **Armadillo binary** in
 * this case. This binary must represent a 4 x N matrix, where N is the
 * number of test points.
 *
 */
float computeRMSE(TimeSVDPP &predAlgo, const string &testFileName)
{
    // Load from binary.
    fmat testSet;
    testSet.load(testFileName, arma_binary);

    // The test set should have four rows.
    if (testSet.n_rows != 4)
    {
        throw invalid_argument("File " + testFileName + " did not have " +
                               "four rows!");
    }

    // The number we divide by in computing the RMSE.
    int nMinusOne = testSet.n_cols - 1;

    // Accumulator for RMSE (take square root at the end)
    float rmse = 0.0;

    for (unsigned int i = 0; i < testSet.n_cols; i ++)
    {
        int user = roundToInt(testSet(USER_ROW, i));
        int item = roundToInt(testSet(MOVIE_ROW, i));
        int date = roundToInt(testSet(DATE_ROW, i));
        float actualRating = testSet(RATING_ROW, i);
        
        float prediction = predAlgo.predict(user, item, date);
        
        rmse += pow(actualRating - prediction, 2.0)/nMinusOne;
    }

    return sqrt(rmse);
}


/**
 * Goes through the test file and saves the prediction results to the
 * specified output file.
 *
 */
void testOnDataFile(TimeSVDPP &predAlgo, const string &testFileName,
                    const string &outputFileName)
{
    ifstream testDataFile(testFileName);
    ofstream outputFile(outputFileName); 
    
    if (testDataFile.fail())
    {
        throw runtime_error("Couldn't find test file at " + testFileName);
    }

    if (outputFile.fail())
    {
        throw runtime_error("Couldn't open output file at " 
                            + outputFileName);
    }

    string testDataLine;
    
    cout << "\nTesting on data in " << testFileName << "..." << endl;

    while (getline(testDataFile, testDataLine))
    {
        // Read the line and split it.
        vector<int> thisLineVec;
        splitIntoInts(testDataLine, DELIMITER, thisLineVec);

        if (thisLineVec.size() != 3)
        {
            throw logic_error("The line \"" + testDataLine + "\" did not "
                              "contain three delimiter-separated "
                              "entries!");
        }
        
        // The first entry is the user ID, the second entry is the item ID,
        // and the third entry is the date. All of these should be
        // zero-indexed!
        int user = thisLineVec[0];
        int item = thisLineVec[1];
        int date = thisLineVec[2];
        
        // Output the prediction to file.
        float prediction = predAlgo.predict(user, item, date);
        outputFile << setprecision(RATING_SIG_FIGS) << prediction << endl;
    }
    
    outputFile.close();

    cout << "\nOutputted predictions on " << testFileName << " to the " 
        "output file " << outputFileName << endl;
}
