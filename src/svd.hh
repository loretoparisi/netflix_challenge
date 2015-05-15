/*
 * This class allows us to carry out SVD via stochastic gradient descent.
 *
 */

#ifndef SVD_HH
#define SVD_HH

#include <algorithm>
#include <armadillo>
#include <array>
#include <cmath>
#include <ctime>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <netflix.hh>
#include <basealgorithm.hh>

using namespace std;
using namespace arma;
using namespace netflix; // challenge-related constants/functions.

class SVD : public BaseAlgorithm
{
private:
    // Regularization constants for each internal variable. See the train()
    // method for more on what these mean. These values came from a
    // combination of the Koren paper and
    // http://www.netflixprize.com/community/viewtopic.php?id=1359 and
    // other tweaks.
    static constexpr float SVD_LAM_B_I = 0.008;
    static constexpr float SVD_LAM_B_U = 0.008;
    static constexpr float SVD_LAM_Q_I = 0.014;
    static constexpr float SVD_LAM_P_U = 0.014;
    
    // Step sizes used for stochastic gradient descent. See the train()
    // method for more on which parameters these apply to. These came from
    // a combination of the Koren paper, the abovementioned forum link, and
    // other tweaks.
    float SVD_GAMMA_B_I = 0.007;
    float SVD_GAMMA_B_U = 0.007;
    float SVD_GAMMA_Q_I = 0.007;
    float SVD_GAMMA_P_U = 0.007;
    
    // The fraction by which the step sizes will be multiplied on each
    // iteration (as recommended in the Koren paper).
    static constexpr float SVD_GAMMA_MULT_PER_ITER = 0.90;
    
    // The number of factors used in matrix factorization.
    const int numFactors;
    
    // The number of users and items in this dataset.
    const int numUsers, numItems;
    
    // The number of iterations for which SVD++ will be carried out.
    const int numIterations;
    
    // The mean rating assigned to all items in the dataset. This is "mu"
    // in the Koren paper.
    const float meanRating;
    
    // The bias for each user. Referred to as "b_u" in the Koren paper. The
    // uth element in this is the bias for user u.
    fcolvec bUser;

    // The bias for each item. Referred to as "b_i" in the Koren paper.
    fcolvec bItem;

    // The number of items rated by each user in the training set. This is
    // a column vector with numUsers elements. The nth element corresponds
    // to the number of items rated by user n (in the training set).
    fcolvec numItemsTrainingSet;

    // The user factor matrix. This is a numFactors x numUsers matrix. The
    // nth column represents the user factor array p_n, using the convention
    // of the Koren paper.
    fmat userFacMat;

    // The item factor matrix. This is a numFactors x numItems matrix. The
    // nth column represents the item factor array q_n, using the
    // convention of the Koren paper.
    fmat itemFacMat;

    // Whether the algorithm has been trained yet or not.
    bool trained = false;

    // Whether we're using cached data or not.
    bool usingCachedData = false;

    void initInternalData();
    void populateNumItemsTrainingSet(const fmat &data);
    float computeRMSE(const string &testFileName);

public:
    SVD(int numUsers, int numItems, float meanRating, int numFactors,
        int numIterations);

    SVD(int numUsers, int numItems, float meanRating, int numFactors,
        int numIterations,
        const string &fileNameBUser, const string &fileNameBItem,
        const string &fileNameUserFacMat,
        const string &fileNameItemFacMat);
     
    ~SVD();
    
    void train(const fmat &data);

    void trainAndCache(const fmat &data, const string &fileNameBUser,
                       const string &fileNameBItem,
                       const string &fileNameUserFacMat,
                       const string &fileNameItemFacMat);
    
    void trainAndCache(const string &fileNameData,
                       const string &fileNameBUser,
                       const string &fileNameBItem,
                       const string &fileNameUserFacMat,
                       const string &fileNameItemFacMat);
    
    float predict(int user, int item, int date, bool bound);
};

#endif // SVD_HH
