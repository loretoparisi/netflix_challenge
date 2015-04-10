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
#include <armadillo>
#include <iostream>
#include <fstream>
#include <stdexcept>

using namespace std;
using namespace arma;
using namespace netflix; // challenge-related constants/functions.


/* Constants */

// The index of the dataset to use for training.
const int TRAINING_SET_IND = VALID_SET;

// The number of factors to use for SVD++.
const int NUM_FACTORS = 60;

// The number of iterations of SVD++ to carry out.
const int NUM_ITERATIONS = 30;

// The name of the output file to use.


int main(void)
{
    // We'll temporarily read our data into a vector, and then convert into
    // an imat afterwards. The number of entries in this vector will be a
    // multiple of 3. Each triple corresponds to a (user, movie, rating)
    // data point.
    vector<int> tempData;

    // Read in from "all.dta" and create the training set by including the
    // entries with index "TRAINING_SET_IND" (this requires simultaneously
    // traversing "all.idx")
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
                              ALL_IDX_FN + " and " + ALL_DATA_FN + "!");
        }

        // Check the set index of this line. We will ignore this line if
        // the corresponding data isn't in the training set.
        if (thisLineInd != TRAINING_SET_IND)
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
            throw logic_error("The line \"" + allDataLine + "\" did not "
                              "contain four delimiter-separated "
                              "entries!");
        }
        
        // Insert the (user, movie, rating) information for this entry into
        // trainingSet. Note that user IDs and movie IDs are
        // **one-indexed**, so we subtract 1 here. This makes things easier
        // in terms of array access later on.
        int thisEntry[3] = {thisDataLineVec[0] - 1, // user
                            thisDataLineVec[1] - 1, // movie
                            thisDataLineVec[3]  // rating
                           };

        tempData.insert(tempData.end(), thisEntry, thisEntry + 3);

        inputsReadIn ++;

        if (inputsReadIn % 100000 == 0)
        {
            cout << "Read in " << inputsReadIn << " data points so far" <<
                endl;
        }
    }

    cout << endl;

    // The matrix where we'll store our training set. The three columns in
    // this will correspond to (user, movie, rating).
    imat trainingSet = conv_to<imat>::from(tempData);

    // Armadillo works in column-major order so we'll have to reshape the
    // matrix to 3 x inputsReadIn before transposing it...
    trainingSet.reshape(3, inputsReadIn);
    inplace_trans(trainingSet);

    // Set up an SVDPP object with verbose output, and train.
    SVDPP predAlgo(NUM_USERS, NUM_MOVIES, MEAN_RATING_TRAINING_SET,
                   NUM_FACTORS, NUM_ITERATIONS, N_FN, true);
    predAlgo.train(trainingSet);

    // Go through qual.dta to produce a prediction file.


    cout << predAlgo.predict(0, 79) << endl;
    cout << predAlgo.predict(0, 13833) << endl;

    
}
