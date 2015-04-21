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
const string TRAIN_PATH_UM =    "../../netflix_challenge/data/train.dta";
const string TRAIN_PATH_MU =    "../../netflix_challenge/data/train-mu.dta";
const string P_PATH =           "../../netflix_challenge/data/knn-p-35.dta";
const string QUAL_PATH =        "../../netflix_challenge/data/um/new_qual.dta";
const string OUTPUT_PATH =      "data/output_knn_sharon_q17770.dta";
const bool SAVE_P = false;
const bool TEST = true;

int main(void)
{
    // Initializing the KNN.
    KNN *knn = new KNN(TRAIN_PATH_UM, TRAIN_PATH_MU, P_PATH,
    	QUAL_PATH, OUTPUT_PATH, TEST);
    knn->loadData();
    if (SAVE_P)
    {
    	knn->calcP();
    	knn->saveP();
    }
    knn->loadP();
    knn->output();
    cout << "KNN completed.\n";
}
