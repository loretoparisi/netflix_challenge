/** 
 * A simple test to check if RBM (new) is working. Note that this class
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
#include <rbm_new.hh>

using namespace std;
using namespace arma;
using namespace netflix; // challenge-related constants/functions.

// The Armadillo binary file to use for training.
// Make sure that both UM and MU data are using the same
// "type" of training resource (eg. probe, hidden, base, etc.)
const string TRAIN_UM = BASE_BIN;

const int NUM_FACTORS = 100;

const float LEARNING_RATE = 0.001;

const int NUM_ITERS = 3;

// The name of the output file to use (for predictions on "qual").
const string OUTPUT_FN = "data/rbm_cached/VALID_f100_i2_predictions.dta";

// Sig-figs for output file.
const int RATING_SIG_FIGS = 4;

fcolvec temp;

// Helper function that carries out "predAlgo" on the test file specified
// by testFileName, and then puts the prediction results (for each (user,
// item, time) in testFileName) in outputFileName.
void testOnDataFile(RBM_New &predAlgo, const string &testFileName,
                    const string &outputFileName);

// Helper function that carries out "predAlgo" on the test file specified
// by testFileName (most likely the "probe" dataset), and calculates the
// RMSE based on those results. Note that testFileName must refer to an
// Armadillo binary in this case.
float computeRMSE(RBM_New &predAlgo, const string &testFileName);


int main(void)
{
    // Load from binary.
    fmat trainingSetUM;
    trainingSetUM.load(TRAIN_UM, arma_binary);
    cout << "Loaded training data from " << TRAIN_UM << "."
        << endl;
        cout << trainingSetUM.n_cols << endl;
    RBM_New predAlgo(NUM_USERS, NUM_MOVIES,
        MEAN_RATING_TRAINING_SET, MAX_RATING, NUM_FACTORS,
        LEARNING_RATE, NUM_ITERS);
    //predAlgo.train(trainingSetUM);
    predAlgo.new_train(trainingSetUM);
    int iter = 20;
    for (int j = 0; j < iter; j++)
    {
        std::cout << "At iteration " << j << std::endl;
        predAlgo.update(j);
            // Get probe RMSE.
        float probeRMSE = computeRMSE(predAlgo, PROBE_BIN);
        // Go through qual.dta to produce a prediction file.
        //testOnDataFile(predAlgo, QUAL_DATA_FN, OUTPUT_FN);
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
float computeRMSE(RBM_New &predAlgo, const string &testFileName)
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
    temp.resize(testSet.n_cols);
    temp.zeros();

    // Accumulator for RMSE (take square root at the end)
    float rmse = 0.0;

    for (unsigned int i = 0; i < testSet.n_cols; i ++)
    {
        int user = roundToInt(testSet(USER_ROW, i));
        int item = roundToInt(testSet(MOVIE_ROW, i));
        int date = roundToInt(testSet(DATE_ROW, i));
        float actualRating = testSet(RATING_ROW, i);
        
        //float prediction = predAlgo.predict(user, item, date, true);
        float prediction;
        if (temp[i] != 0)
            prediction = temp[i];
        else
            prediction = -1;
        prediction = predAlgo.new_predict(user, item, 0);
        
        rmse += pow(actualRating - prediction, 2.0)/nMinusOne;
    }

    return sqrt(rmse);
}


/**
 * Goes through the test file and saves the prediction results to the
 * specified output file.
 *
 */
void testOnDataFile(RBM_New &predAlgo, const string &testFileName,
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
        //float prediction = predAlgo.predict(user, item, date, true);
        float prediction = predAlgo.new_predict(user, item, -1);
        outputFile << setprecision(RATING_SIG_FIGS) << prediction << endl;
    }
    
    outputFile.close();

    cout << "\nOutputted predictions on " << testFileName << " to the " 
        "output file " << outputFileName << endl;
}

