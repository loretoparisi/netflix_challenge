/** 
 * A simple test to check if SVD++ is working. This is carried out on the
 * Netflix challenge data set. Note that this class assumes that data is in
 * the (user, movie) order.
 *
 * Note: The main method does not expect any arguments in this case.
 *
 */

#include "svdpp.hh"
#include "netflix.hh"
#include <string>
#include <vector>
#include <set>
#include <armadillo>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>

using namespace std;
using namespace arma;
using namespace netflix; // challenge-related constants/functions.


/* Constants */

// The indices of the dataset to use for training.
const set<int> TRAINING_SET_INDICES = {BASE_SET, HIDDEN_SET, VALID_SET};

// The number of factors to use for SVD++.
const int NUM_FACTORS = 200;

// The number of iterations of SVD++ to carry out.
const int NUM_ITERATIONS = 30;

// The name of the output file to use (for predictions on "qual").
const string OUTPUT_FN = "../data/svdpppredictions.dta";

// Sig-figs for output file.
const int RATING_SIG_FIGS = 4;

// Whether the data will be cached after training.
const bool WILL_CACHE_DATA = true;

// Whether we're using cached data **instead of** training.
const bool USING_CACHED_DATA = false;

// The locations of the files we'll use for caching (and read from if we're
// using cached data). These must be in Armadillo binary format!
const string B_USER_FN =            "../data/svdppcached/b_user.mat";
const string B_ITEM_FN =            "../data/svdppcached/b_item.mat";
const string USER_FAC_MAT_FN =      "../data/svdppcached/user_fac.mat";
const string ITEM_FAC_MAT_FN =      "../data/svdppcached/item_fac.mat";
const string Y_MAT_FN =             "../data/svdppcached/y.mat";
const string SUM_MOVIE_WEIGHTS_FN = "../data/svdppcached/user_sum_y.mat";


// Helper function that carries out "predAlgo" on the test file specified
// by testFileName, and then puts the prediction results (for each (user,
// item) pair in testFileName) in outputFileName.
void testOnDataFile(SVDPP &predAlgo, const string &testFileName,
                    const string &outputFileName);

// Function that shuffles rows of data in a vector in place. The vector is
// treated as a 2D array with numRows rows and numCols columns.
void shuffleVector(vector<int> &data, int numRows, int numCols);


int main(void)
{
    // Sanity check on arguments.
    if (USING_CACHED_DATA && WILL_CACHE_DATA)
    {
        throw logic_error("It doesn't make sense to set the \"will cache "
                          "data\" flag if you're using cached data!");
    }
    
    // If we've cached the matrices produced by SVD++, use those to skip
    // the training step.
    if (USING_CACHED_DATA)
    {
        // Construct an SVDPP object and pass in the cached data file
        // names.
        SVDPP predAlgo(NUM_USERS, NUM_MOVIES, MEAN_RATING_TRAINING_SET,
                       NUM_FACTORS, NUM_ITERATIONS, N_FN,
                       B_USER_FN, B_ITEM_FN, USER_FAC_MAT_FN,
                       ITEM_FAC_MAT_FN, Y_MAT_FN, SUM_MOVIE_WEIGHTS_FN);
        
        // Go through qual.dta and produce a prediction file.
        testOnDataFile(predAlgo, QUAL_DATA_FN, OUTPUT_FN);
    }
    else // If not using cached data, we need to train.
    {
        imat trainingSet = parseData(INDEX_PATH, DATA_PATH, 
                                     TRAINING_SET_INDICES);

        SVDPP predAlgo(NUM_USERS, NUM_MOVIES, MEAN_RATING_TRAINING_SET,
                       NUM_FACTORS, NUM_ITERATIONS, N_FN);
        
        // Check if we want to cache.
        if (WILL_CACHE_DATA)
        {
            // If so, train and then cache the results after training.
            cout << "\nTraining SVD++. The resulting matrices will be "
                    "cached." << endl;
            
            predAlgo.trainAndCache(trainingSet, B_USER_FN, B_ITEM_FN,
                                   USER_FAC_MAT_FN, ITEM_FAC_MAT_FN,
                                   Y_MAT_FN, SUM_MOVIE_WEIGHTS_FN);
        }
        else
        {
            // If not, just train.
            cout << "\nTraining SVD++. The resulting matrices won't be "
                    "cached." << endl;
            predAlgo.train(trainingSet);
        }        
        
        // Go through qual.dta to produce a prediction file.
        testOnDataFile(predAlgo, QUAL_DATA_FN, OUTPUT_FN);
    }

}


/**
 * Use the Fisher-Yates algorithm to randomly shuffle rows of data in a
 * vector (in place). We're treating the vector as a 2D array.
 *
 * @param data: The input vector.
 * @param numRows: The number of rows in the vector.
 * @param numCols: The number of columns in the vector.
 *
 */
void shuffleVector(vector<int> &data, int numRows, int numCols)
{
    // Seed is dependent on program run-time.
    srand(time(NULL));
 
    // Consider rows 0 through i on each iteration, and swap row i with row
    // j, where j is a randomly chosen index between 0 and i.  Decrement
    // the index on each iteration so that the "end" of the vector is the
    // shuffled part.
    for (int i = numRows - 1; i > 0; i --)
    {
        // The random index we'll use for our swap. Ranges from 0 to i.
        int j = rand() % (i + 1);

        // Swap rows i and j. We'll treat our vector like a 2D array, so
        // that the element at the mth row and nth column is given by 
        // data[m * numCols + n].
        for (int colNumber = 0; colNumber < numCols; colNumber ++)
        {
            int temp = data[i * numCols + colNumber];
            data[i * numCols + colNumber] = data[j * numCols + colNumber];
            data[j * numCols + colNumber] = temp;
        }
    }
}


/**
 * Goes through the test file and saves the prediction results to the
 * specified output file.
 *
 */
void testOnDataFile(SVDPP &predAlgo, const string &testFileName,
                    const string &outputFileName)
{
    ifstream qualDataFile(testFileName);
    ofstream outputFile(outputFileName); 
    
    if (qualDataFile.fail())
    {
        throw runtime_error("Couldn't find test file at " + testFileName);
    }

    if (outputFile.fail())
    {
        throw runtime_error("Couldn't open output file at " 
                            + outputFileName);
    }

    string qualDataLine;
    
    cout << "\nTesting on data in " << testFileName << "..." << endl;

    while (getline(qualDataFile, qualDataLine))
    {
        // Read the line and split it.
        vector<int> thisLineVec;
        splitIntoInts(qualDataLine, DELIMITER,
                      thisLineVec);

        if (thisLineVec.size() != 3)
        {
            throw logic_error("The line \"" + qualDataLine + "\" did not "
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
