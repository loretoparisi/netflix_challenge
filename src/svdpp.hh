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

#include "singlealgorithm.hh"
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

class SVDPP : public SingleAlgorithm
{
private:
    // Regularization constants for each internal variable. See the train()
    // method for more on what these mean. These values came from a
    // combination of the Koren paper and
    // http://www.netflixprize.com/community/viewtopic.php?id=1359 and
    // other tweaks.
    static constexpr float SVDPP_LAM_B_I = 0.005; // different
    static constexpr float SVDPP_LAM_B_U = 0.030;
    static constexpr float SVDPP_LAM_Q_I = 0.006;
    static constexpr float SVDPP_LAM_P_U = 0.080;
    static constexpr float SVDPP_LAM_Y_J = 0.030;
    
    // Step sizes used for stochastic gradient descent. See the train()
    // method for more on which parameters these apply to. These came from
    // a combination of the Koren paper, the abovementioned forum link, and
    // other tweaks.
    float SVDPP_GAMMA_B_I = 0.003;
    float SVDPP_GAMMA_B_U = 0.012;
    float SVDPP_GAMMA_Q_I = 0.011;
    float SVDPP_GAMMA_P_U = 0.006;
    float SVDPP_GAMMA_Y_J = 0.001;

    // The fraction by which the step sizes will be multiplied on each
    // iteration (as recommended in the Koren paper).
    static constexpr float SVDPP_GAMMA_MULT_PER_ITER = 0.9;

    // What the different rows in the data matrix are expected to
    // correspond to.
    static const int USER_ROW =     0;
    static const int ITEM_ROW =     1;
    static const int DATE_ROW =     2;
    static const int RATING_ROW =   3;
    
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

    // The number of items rated by each user in the training set. This is
    // a column vector with numUsers elements. The nth element corresponds
    // to the number of items rated by user n (in the training set).
    fcolvec numItemsTrainingSet;

    // This is sum_{j in N(u)} y_j for each user u in the dataset. So the
    // uth column of this corresponds to the uth user's value for 
    // sum_{j in N(u)} y_j. This will change between iterations, but it's
    // very useful to precompute it at the beginning of each iteration.
    // Note that this matrix is numFactors x numUsers in shape.
    fmat sumMovieWeights;

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
    void populateNumItemsTrainingSet(const imat &data);
    void updateSumMovieWeights(int lowUserNum, int highUserNum);
    inline void updateUserSumMovieWeights(int user);

public:
    SVDPP(int numUsers, int numItems, float meanRating, int numFactors,
          int numIterations, const string &fileNameN);
    SVDPP(int numUsers, int numItems, float meanRating, int numFactors,
          int numIterations, const string &fileNameN,
          const string &fileNameBUser, const string &fileNameBItem,
          const string &fileNameUserFacMat,
          const string &fileNameItemFacMat, const string &fileNameYMat,
          const string &fileNameSumMovieWeights);
     
    ~SVDPP();
    
    void train(const imat &data);
    void trainAndCache(const imat &data, const string &fileNameBUser,
                       const string &fileNameBItem,
                       const string &fileNameUserFacMat,
                       const string &fileNameItemFacMat,
                       const string &fileNameYMat,
                       const string &fileNameSumMovieWeights);
    
    float predict(int user, int item, int date);
};

#endif // SVDPP_HH
