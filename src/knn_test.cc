/** 
 * A simple test to check if SVD++ is working. This is carried out on the
 * Netflix challenge data set. Note that this class assumes that data is in
 * the (user, movie) order.
 *
 * Note: The main method does not expect any arguments in this case.
 *
 */

#include "knn.hh"
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

// The file path we use to train, to test, and to output.
const string TRAIN_PATH =    "data/small_test.dta";
const string QUAL_PATH =     "data/probe.dta";
const string OUTPUT_PATH =   "data/output_knn_sharon.dta";
const bool TEST = true;

int main(void)
{
    // Initializing the KNN.
    KNN *knn = new KNN(NUM_USERS, NUM_MOVIES, TRAIN_PATH, QUAL_PATH,
        OUTPUT_PATH, TEST);
    knn->run();
}
