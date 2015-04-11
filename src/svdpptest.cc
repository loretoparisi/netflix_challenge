/** 
 * A simple test to check if SVD++ is working. This is carried out on the
 * Netflix challenge data set. Note that this class assumes that data is in
 * the (user, movie) order.
 *
 * Note: The main method does not expect any arguments in this case.
 *
 */

#include "svdpp.hh"
#include "netflix_namespace.hh"
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
const set<int> TRAINING_SET_INDICES = {BASE_SET};

// The number of factors to use for SVD++.
const int NUM_FACTORS = 60;

// The number of iterations of SVD++ to carry out.
const int NUM_ITERATIONS = 30;

// The name of the output file to use.
const string OUTPUT_FN = "../data/svdpppredictions.dta";

// Sig-figs for output file.
const int RATING_SIG_FIGS = 4;

// Whether the data will be cached after training.
const bool WILL_CACHE_DATA = true;

// Whether we're using cached data **instead of** training.
const bool USING_CACHED_DATA = false;

// The locations of the files we'll use for caching (and read from if we're
// using cached data). These must be in Armadillo binary format!
const string B_USER_FN =        "../data/svdppcached/b_user.mat";
const string B_ITEM_FN =        "../data/svdppcached/b_item.mat";
const string USER_FAC_MAT_FN =  "../data/svdppcached/user_fac.mat";
const string ITEM_FAC_MAT_FN =  "../data/svdppcached/item_fac.mat";
const string Y_MAT_FN =         "../data/svdppcached/y.mat";


// Helper function that carries out "predAlgo" on the test file specified
// by testFileName, and then puts the prediction results (for each (user,
// item) pair in testFileName) in outputFileName.
void testOnDataFile(SVDPP &predAlgo, const string &testFileName,
                    const string &outputFileName);


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
                       ITEM_FAC_MAT_FN, Y_MAT_FN, /* verbose */ true);
        
        // Go through qual.dta and produce a prediction file.
        testOnDataFile(predAlgo, QUAL_DATA_FN, OUTPUT_FN);
    }
    else // If not using cached data, we need to train.
    {
        // We'll temporarily read our data into a vector, and then convert
        // into an imat afterwards. The number of entries in this vector
        // will be a multiple of 3. Each triple corresponds to a (user,
        // movie, rating) data point.
        vector<int> tempData;
        
        // Read in from "all.dta" and create the training set by including
        // the entries with index "TRAINING_SET_IND" (this requires
        // simultaneously traversing "all.idx")
        ifstream allIdxFile(ALL_IDX_FN);
        ifstream allDataFile(ALL_DATA_FN);
        
        // Lines from the above two files will be stored in these.
        string allDataLine;
        int thisLineInd;
       
        // Keep track of how many inputs have been read in.
        int inputsReadIn = 0; 

        // Get the index for the first line and so on.
        while (allIdxFile >> thisLineInd)
        {
            // Should be able to read the next line in "all.dta" too...
            if (!getline(allDataFile, allDataLine))
            {
                throw logic_error("Couldn't simultaneously traverse" +
                                  ALL_IDX_FN + " and " + ALL_DATA_FN +
                                  "!");
            }

            // Check the set index of this line. We will ignore this line if
            // the corresponding data isn't in the training set.
            bool notInTrainingSet = TRAINING_SET_INDICES.find(thisLineInd) == 
                TRAINING_SET_INDICES.end();
            
            if (notInTrainingSet)
            {
                continue;
            }
     
            // This line should be in the training set. Split the string from
            // all.dta around the specified delimiter.
            vector<int> thisDataLineVec;
            splitIntoInts(allDataLine, NETFLIX_FILES_DELIMITER,
                          thisDataLineVec);
            
            if (thisDataLineVec.size() != 4)
            {
                throw logic_error("The line \"" + allDataLine + 
                                  "\" did not contain four "
                                  "delimiter-separated entries!");
            }
            
            // Insert the (user, movie, rating) information for this entry
            // into trainingSet. The user IDs and movie IDs should be
            // zero-indexed in the data file!
            int thisEntry[3] = {thisDataLineVec[0],     // user
                                thisDataLineVec[1],     // movie
                                thisDataLineVec[3]};    // rating
            
            tempData.insert(tempData.end(), thisEntry, thisEntry + 3);
            
            inputsReadIn ++;

            if (inputsReadIn % 1000000 == 0)
            {
                cout << "Stored " << inputsReadIn << " data points so far" 
                    " (from the specified training set)" << endl;
            }
        }

        cout << "\nRead in a total of " << inputsReadIn << " points" << endl;
        
        // This is the matrix where we'll store our training set. The three
        // columns in this will correspond to (user, movie, rating).
        imat trainingSet = conv_to<imat>::from(tempData);

        // Armadillo works in column-major order so we'll have to reshape
        // the matrix to 3 x inputsReadIn before transposing it...
        trainingSet.reshape(3, inputsReadIn);
        inplace_trans(trainingSet);
        
        SVDPP predAlgo(NUM_USERS, NUM_MOVIES, MEAN_RATING_TRAINING_SET,
                       NUM_FACTORS, NUM_ITERATIONS, N_FN,
                       /* verbose */ true);

        // Check if we want to cache.
        if (WILL_CACHE_DATA)
        {
            // If so, train and then cache the results after training.
            cout << "\nTraining SVD++. The resulting matrices will be "
                    "cached." << endl;

            predAlgo.trainAndCache(trainingSet, B_USER_FN, B_ITEM_FN,
                                   USER_FAC_MAT_FN, ITEM_FAC_MAT_FN,
                                   Y_MAT_FN);
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
 * Goes through the test file and saves the prediction results to the
 * specified output file.
 *
 */
void testOnDataFile(SVDPP &predAlgo, const string &testFileName,
                    const string &outputFileName)
{
    ifstream qualDataFile(testFileName);
    ofstream outputFile(outputFileName, ios::out); 
    
    string qualDataLine;
    
    cout << "\nTesting on data in " << testFileName << "..." << endl;

    while (getline(qualDataFile, qualDataLine))
    {
        // Read the line and split it.
        vector<int> thisLineVec;
        splitIntoInts(qualDataLine, NETFLIX_FILES_DELIMITER,
                      thisLineVec);

        if (thisLineVec.size() != 3)
        {
            throw logic_error("The line \"" + qualDataLine + "\" did not "
                              "contain three delimiter-separated "
                              "entries!");
        }
        
        // The first entry is the user ID, and the second entry is the item
        // ID. Both of these should be zero-indexed!
        int user = thisLineVec[0];
        int item = thisLineVec[1];

        // Output the prediction to file.
        float prediction = predAlgo.predict(user, item);
        outputFile << setprecision(RATING_SIG_FIGS) << prediction << endl;
    }
    
    outputFile.close();
    
    cout << "\nOutputted predictions on " << testFileName << " to the " 
        "output file " << outputFileName << endl;
}
