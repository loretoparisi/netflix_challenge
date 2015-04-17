/*
 * This class allows us to carry out KNN on the Netflix dataset. 
 */

#ifndef KNN_HH
#define KNN_HH

#include "mlalgorithm.hh"
#include "netflix.hh"
#include <armadillo>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <random>
#include <functional>
#include <iterator>
#include <array>
#include <math.h>
#include <sstream>
#include <fstream>
#include <iostream>

using namespace std;
using namespace arma;
using namespace netflix; // challenge-related constants/functions.

class KNN : public MLAlgorithm
{
private:
    // The number of users and items in this dataset.
    const int numUsers, numItems;
    // Testing data set.
    const string &qualFilename;
    const string &outputFilename;

    // In test mode => test data has "answers."
    // For qual data, test mode is "false."
    bool test;

    // rateMatrix[user][movie] = rating
    unordered_map<int, unordered_map<int, int> > rateMatrix;
    // averageUser[user] = average_rating
    unordered_map<int, float > averageUser;
    // giant <# of user> by <# of user> matrix
    unordered_map<int, unordered_map<int, float> > userUser;

    void initInternalData(const string &trainFilename);
    float simPearson(int userId1, int userId2);
    void eachUser();
    float predict(int userId, int movieId);
    void beginKNN();

public:
    KNN(int numUsers, int numItems, const string &trainFilename,
        const string &qualFilename, const string &outputFilename,
        bool test);
    void train(const imat &data);
    ~KNN();
    void run();
};

#endif // KNN_HH
