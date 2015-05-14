/** 
 * A simple test to check if KNN is working. This is carried out on the
 * Netflix challenge data set. Note that this class assumes that data is in
 * the (user, movie) order.
 *
 * Note: The main method does not expect any arguments in this case.
 *
 */

#include <string>
#include <vector>
#include <set>
#include <armadillo>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>

#include <netflix.hh>
#include <knn.hh>

using namespace std;
using namespace arma;
using namespace netflix; // challenge-related constants/functions.


// The Armadillo binary file to use for training.
// Make sure that both UM and MU data are using the same
// "type" of training resource (eg. probe, hidden, base, etc.)
// const string TRAIN_UM = VALID_BIN;
const string TRAIN_UM = ALL_TRAIN_BIN;

// Minimum common neighbors required for decent prediction.
const int MIN_COMMON = 24;

// Max weight elements to consider when predicting.
const unsigned int MAX_WEIGHT = 30;

// If P is already precomputed, we only need to load, so set this to true.
// If P hasn't been computed, set this to false.
const bool LOAD_P = false;

// Whether we want to save the output of P.
const bool SAVE_P = true;

// Sig-figs for output file.
const int RATING_SIG_FIGS = 4;

// If we do want to load or save, the location of P file.
const string P_PATH = "data/knn_cached/knn-p.dta";

// The name of the output file to use (for predictions on "qual").
const string OUTPUT_FN = "data/knn_cached/knn_qual_predictions.dta";

// Test on qual file for KNN and output the file to store.
void testOnDataFile(KNN &predAlgo, const string &testFileName,
                    const string &outputFileName);

// Compute the probe RMSE of our KNN model.
float computeRMSE(KNN &predAlgo, const string &testFileName);

int main(void)
{
    cout << "Start KNN..." << endl;
    cout << "Load UM matrix..." << endl;
    // Load from binary.
    fmat trainingSetUM;
    trainingSetUM.load(TRAIN_UM, arma_binary);
    cout << "Finished loading UM matrix." << endl;

    // Initializing the KNN.
    KNN knn(NUM_USERS, NUM_MOVIES, MIN_COMMON, MAX_WEIGHT,
            LOAD_P, SAVE_P, P_PATH);
    knn.train(trainingSetUM);

    // Go through qual.dta to produce a prediction file.
    testOnDataFile(knn, QUAL_DATA_FN, OUTPUT_FN);

    // Get probe RMSE.
    float probeRMSE = computeRMSE(knn, PROBE_BIN);

    cout << "Probe RMSE: " << probeRMSE << endl;
    cout << "KNN completed.\n";
}

/**
 * Compute the RMSE of a given prediction algorithm on a certain set of
 * data. Note that testFileName must refer to an **Armadillo binary** in
 * this case. This binary must represent a 4 x N matrix, where N is the
 * number of test points.
 *
 */
float computeRMSE(KNN &predAlgo, const string &testFileName)
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
        
        float prediction = predAlgo.predict(user, item, date, true);
        
        rmse += pow(actualRating - prediction, 2.0)/nMinusOne;
    }

    return sqrt(rmse);
}


/**
 * Goes through the test file and saves the prediction results to the
 * specified output file.
 *
 */
void testOnDataFile(KNN &predAlgo, const string &testFileName,
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
        float prediction = predAlgo.predict(user, item, date, true);
        outputFile << setprecision(RATING_SIG_FIGS) << prediction << endl;
    }
    
    outputFile.close();

    cout << "\nOutputted predictions on " << testFileName << " to the " 
        "output file " << outputFileName << endl;
}

