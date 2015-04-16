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

using namespace std;
using namespace arma;
using namespace netflix; // challenge-related constants/functions.

class SVDPP : public MLAlgorithm
{
private:
    // Regularization constants. See the train() method for more on what these
    // mean.
    static constexpr float SVDPP_REG_1 = 0.005;
    static constexpr float SVDPP_REG_2 = 0.015;

    // Step sizes used for stochastic gradient descent. See the train() method
    // for more on which parameters these apply to.
    float SVDPP_GAMMA_1 = 0.007;
    float SVDPP_GAMMA_2 = 0.007;

    // The fraction by which the step sizes will be multiplied on each
    // iteration (as recommended in the Koren paper).
    static constexpr float SVDPP_GAMMA_MULT_PER_ITER = 0.9;
    
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
    unordered_map<int, vector<int> > N;

    // The bias for each user. Referred to as "b_u" in the Koren paper. The
    // uth element in this is the bias for user u.
    fcolvec bUser;

    // The bias for each item. Referred to as "b_i" in the Koren paper.
    fcolvec bItem;

    // The user factor matrix. This is a numFactors x numUsers matrix. The
    // nth column represents the user factor array p_n, using the convention
    // of the Koren paper.
    fmat userFacMat;

    // The item factor matrix. This is a numFactors x numItems matrix. The
    // nth column represents the item factor array q_n, using the
    // convention of the Koren paper.
    fmat itemFacMat;

    // The "y" matrix. This is a numFactors x numItems matrix. The jth
    // column of this is "y_j" in the convention of the Koren paper; it is
    // supposed to weight the implicit preferences of the user (i.e. the
    // preferences in N(u)).
    fmat yMat;

    // Whether the algorithm has been trained yet or not.
    bool trained = false;

    // Whether we're using cached data or not.
    bool usingCachedData = false;

    void initInternalData();
    void populateN(const string &fileNameN);

public:
    SVDPP(int numUsers, int numItems, float meanRating, int numFactors,
          int numIterations, const string &fileNameN);
    SVDPP(int numUsers, int numItems, float meanRating, int numFactors,
          int numIterations, const string &fileNameN,
          const string &fileNameBUser, const string &fileNameBItem,
          const string &fileNameUserFacMat,
          const string &fileNameItemFacMat, const string &fileNameYMat);
    
    ~SVDPP();
    
    void train(const imat &data);
    void trainAndCache(const imat &data, const string &fileNameBUser,
                       const string &fileNameBItem,
                       const string &fileNameUserFacMat,
                       const string &fileNameItemFacMat,
                       const string &fileNameYMat);
    
    float predict(int user, int item);
};

#endif // SVDPP_HH
