/*
 * This class allows us to carry out SVD++ on the Netflix dataset. Our
 * procedure for SVD++ uses stochastic gradient descent, and is based on
 * the methods provided in the following paper:
 *      http://dl.acm.org/citation.cfm?id=1401944
 *
 * We will refer to this as "the Koren paper" throughout this header file
 * and its corresponding implementation. Note that some of our notation
 * differs from that used in the paper.
 *
 */

#ifndef SVDPP_HH
#define SVDPP_HH

#include "MLAlgorithm.hh"
#include <armadillo>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <stdexcept>
#include <cmath>

using namespace std;
using namespace arma;

// TODO: Put these in a namespace or something in a different "constants"
// file. Namespace could be called netflix_challenge.
const int MIN_RATING = 1;
const int MAX_RATING = 5;

class SVDPP : public MLAlgorithm
{
private:
    // Regularization constants. See the train() method for more on what these
    // mean.
    static const float SVDPP_REG_1 = 0.005;
    static const float SVDPP_REG_2 = 0.015;

    // Step sizes used for stochastic gradient descent. See the train() method
    // for more on which parameters these apply to.
    static const float SVDPP_GAMMA_1 = 0.007;
    static const float SVDPP_GAMMA_2 = 0.007;

    // The fraction by which the step sizes will be multiplied on each
    // iteration (as recommended in the Koren paper).
    static const float SVDPP_GAMMA_MULT_PER_ITER = 0.9;

    // The delimiter used in our data files (e.g. in the data file containing
    // N). TODO: Maybe make a file containing codebase-wide constants like
    // delimiters...
    static const string DELIMITER = " ";

    // The number of factors used in matrix factorization.
    const int numFactors;
    
    // The number of users and items in this dataset.
    const int numUsers, numItems;

    // The number of iterations for which SVD++ will be carried out.
    const int numIterations;
    
    // The mean rating assigned to all items in the dataset. This is "mu"
    // in the Koren paper.
    const float meanRating;

    // A mapping from a user's ID to the items that the user indicated an
    // implicit preference for. These are essentially just the items that
    // we know the user rated (i.e. they show up in the data file), but we
    // don't know their rating. This is called N(u) in the Koren paper.
    unordered_map<int, vector<int>> N;

    // The bias for each user. Referred to as "b_u" in the Koren paper. The
    // uth element in this is the bias for user u.
    frowvec bUser;

    // The bias for each item. Referred to as "b_i" in the Koren paper.
    frowvec bItem;

    // The user factor matrix. This is a numUsers x numFactors matrix. The
    // nth row represents the user factor array p_n, using the convention
    // of the Koren paper.
    fmat userFacMat;

    // The item factor matrix. This is a numItems x numFactors matrix. The
    // nth row represents the item factor array q_n, using the convention
    // of the Koren paper.
    fmat itemFacMat;

    // The "y" matrix. This is a numItems x numFactors matrix. The jth row
    // of this is "y_j" in the convention of the Koren paper; it is
    // supposed to weight the implicit preferences of the user (i.e. the
    // preferences in N(u)).
    fmat yMat;

    // Tells us whether the algorithm has been trained yet or not.
    bool trained = false;

    // This flag enables some cout statements.
    bool verbose = false;
    
    // TODO: refactor this so that we put our convenience functions in a
    // separate file...
    void splitIntoInts(const string &str, const string &delimiter,
                       vector<int> &output);

public:
    SVDPP(int numUsers, int numItems, float meanRating, int numFactors,
          int numIterations, const string &fileNameN, bool verbose);

    ~SVDPP();

    void train(const imat &data);
    float predict(int user, int item);
};

#endif // SVDPP_HH
