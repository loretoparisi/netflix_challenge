/*
 * This class allows us to carry out time-dependent SVD++ on the Netflix
 * dataset. Our procedure for time-dependent SVD++ uses stochastic gradient
 * descent. Our formulas for the predicted rating come from SVD++^(1) as
 * mentioned on page 5 of the following paper:
 *
 *      http://www.netflixprize.com/assets/ProgressPrize2008_BellKor.pdf
 *
 * We will refer to this as "the BellKor paper" throughout this header file
 * and its corresponding implementation. Note that some of our notation
 * differs from that used in the paper.
 *
 */

#ifndef TIMESVDPP_HH
#define TIMESVDPP_HH

#include <algorithm>
#include <armadillo>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <netflix.hh>
#include <singlealgorithm.hh>

using namespace arma;
using namespace netflix; // challenge-related constants/functions.

using std::cout;
using std::endl;

// Used to store (user ID, date ID) tuples.
struct UserDate
{
    int userID;
    unsigned short dateID;

    // An equality function used for hashing.
    bool operator==(const UserDate &other) const
    { 
        return userID == other.userID && dateID == other.dateID;
    }
};


// Define a hash function for UserDates, so they can be keys in
// unordered_maps.
struct UserDateHasher
{
    std::size_t operator()(const UserDate &ud) const
    {
        using std::hash;
        using std::size_t;

        return ((hash<int>()(ud.userID) << 1)
               ^ (hash<int>()(ud.dateID)));
    }
};

// A random-number-generating struct used to populate a vector. Takes
// minimum and maximum values as arguments.
struct genRand 
{ 
    float range;    // (maxVal - minVal) / RAND_MAX
    float minVal;
public:
    genRand(float minVal, float maxVal) : 
        range((maxVal - minVal) / RAND_MAX),
        minVal(minVal)
    {
        std::srand(std::time(NULL));
    }
     
    double operator()() 
    { 
        return std::rand() * range + minVal;
    }
};


// This can technically be a subclass of SVDPP, but there's no real point
// in nesting the hierarchy that much. Especially since SVDPP's internals
// represent fairly different things.
class TimeSVDPP : public SingleAlgorithm
{
private:
    // Regularization constants for each internal variable. See the train()
    // method for more on what these mean. These values came from a
    // combination of posts #50 and #57 on
    // http://www.netflixprize.com/community/viewtopic.php?id=1342&p=3,
    // http://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf,
    // as well as some other tweaks.
    static constexpr float TIMESVDPP_LAM_B_U = 0.0065;          // Laksh
    static constexpr float TIMESVDPP_LAM_ALPHA_B_U = 0.0004;    // Laksh
    static constexpr float TIMESVDPP_LAM_B_U_T = 0.0050;        // Laksh
    static constexpr float TIMESVDPP_LAM_B_I = 0.005;           // Laksh
    static constexpr float TIMESVDPP_LAM_B_I_T = 0.0050;        // Laksh
    static constexpr float TIMESVDPP_LAM_B_I_F_U_T = 4.40e-3;   // Laksh
    static constexpr float TIMESVDPP_LAM_C_U = 0.010;           // Laksh
    static constexpr float TIMESVDPP_LAM_C_U_T = 0.0070;        // Laksh
    static constexpr float TIMESVDPP_LAM_Q_I = 0.0155;          // Laksh
    static constexpr float TIMESVDPP_LAM_Q_I_BIN = 0.022;       // Laksh
    static constexpr float TIMESVDPP_LAM_Q_I_F = 0.018;         // Laksh
    static constexpr float TIMESVDPP_LAM_P_U = 0.0155;          // Laksh
    static constexpr float TIMESVDPP_LAM_ALPHA_P_U = 0.0004;    // Laksh
    static constexpr float TIMESVDPP_LAM_P_U_T = 0.015;         // Laksh
    static constexpr float TIMESVDPP_LAM_Y_J = 0.0155;          // Laksh
    // static constexpr float TIMESVDPP_LAM_Y_J_BIN = 0.015;       // ?

    // Step sizes used for stochastic gradient descent. See the train()
    // method for more on which parameters these apply to. These came from
    // a combination of the abovementioned forum link, BellKor's [PQ1]
    // model, and other tweaks.
    float TIMESVDPP_GAMMA_B_U = 0.0054;                 // Laksh
    float TIMESVDPP_GAMMA_ALPHA_B_U = 0.00003;          // Laksh
    float TIMESVDPP_GAMMA_B_U_T = 0.0028;               // Laksh
    float TIMESVDPP_GAMMA_B_I = 0.005;                  // Laksh
    float TIMESVDPP_GAMMA_B_I_T = 0.0001;               // Laksh
    float TIMESVDPP_GAMMA_B_I_F_U_T = 0.00236;          // Laksh
    float TIMESVDPP_GAMMA_C_U = 0.006;                  // Laksh
    float TIMESVDPP_GAMMA_C_U_T = 0.001;                // Laksh
    float TIMESVDPP_GAMMA_Q_I = 0.005;                  // Laksh
    float TIMESVDPP_GAMMA_Q_I_BIN = 0.0007;             // Laksh
    float TIMESVDPP_GAMMA_Q_I_F = 0.00003;              // Laksh
    float TIMESVDPP_GAMMA_P_U = 0.0050;                 // Laksh
    float TIMESVDPP_GAMMA_ALPHA_P_U = 0.00001;          // Laksh
    float TIMESVDPP_GAMMA_P_U_T = 0.0040;               // Laksh
    float TIMESVDPP_GAMMA_Y_J = 0.0050;                 // Laksh 
    // float TIMESVDPP_GAMMA_Y_J_BIN = 0.0001;          // ?
    
    // The fraction by which the step sizes will be multiplied on each
    // iteration (as recommended in the SVD++ Koren paper).
    static constexpr float TIMESVDPP_GAMMA_MULT_PER_ITER = 0.89;
    
    // The number of factors used in matrix factorization.
    const int numFactors;
    
    // The number of users, items, and times in this dataset. The number of
    // times should be distinguished from the number of time bins (below);
    // the former is 2243 for the Netflix dataset.
    const int numUsers, numItems, numTimes;

    // The number of iterations for which SVD++ will be carried out.
    const int numIterations;

    // The number of time bins to use for the item-dependent bias. This is
    // usually around 30.
    const int numTimeBins;
    
    // The mean rating assigned to all items in the dataset. This is "mu"
    // in the BellKor paper.
    const float meanRating;

    // A mapping from a user's ID and a date ID to the hat{dev_u(t)} value
    // for that user and that date. This value essentially measures how
    // recently a user rated a given movie, relative to the median date at
    // which they've rated movies.
    std::unordered_map<UserDate, float, UserDateHasher> hatDevUT;

    // A mapping from a user ID and a date ID to the f_{ut} log-frequency
    // value for that user and that date. This measures how frequently a
    // user rated on a given date.
    std::unordered_map<UserDate, int, UserDateHasher> fUT;

    // A mapping from a user's ID to the items that the user indicated an
    // implicit preference for. These are essentially just the items that
    // we know the user rated (i.e. they show up in the data file), even if
    // we might not know their rating.  This is called N(u) in the BellKor
    // paper.
    std::unordered_map<int, std::vector<int> > N;

    // The constant bias for each user. Referred to as "b_u" in the BellKor
    // paper. The uth element in this is the constant bias for user u. Note
    // that this doesn't include times.
    fcolvec bUserConst;

    // The modifying bias factor for each user, referred to as "alpha_{u}"
    // in the BellKor paper. This governs how much weight is given to the
    // effect of time on the user's bias.
    fcolvec bUserAlpha;

    // The higher-resolution time-dependent bias factor for each user,
    // referred to as b_{ut} in the BellKor paper. This is a numTimes x
    // numUsers matrix, where the uth column corresponds to the
    // time-dependent bias entries for user u.
    sp_fmat bUserTime;

    // The constant bias for each item. This is b_i in the BellKor 09 paper
    // (it wasn't mentioned in BellKor 08). The nth entry in this is the
    // time-independent bias for the nth item.
    fcolvec bItemConst;

    // The time-dependent bias for each item. This is b_i(t) = b_{i,
    // Bin(t)} in the BellKor paper. We store this as a numTimeBins x
    // numItems matrix, where the ith column will correspond to the
    // time-bin-wise biases for the ith movie.
    fmat bItemTimewise;

    // The frequency-dependent bias for each item. This is b_{i, f_t} in
    // the BellKor paper. This is stored as a (MAX_F_U_T + 1) x numItems
    // matrix, where the ith column corresponds to the frequency-dependent
    // item biases for the ith movie. The +1 comes from the fact that
    // f_{ut} can range from 0 to MAX_F_U_T inclusive.
    fmat bItemFreq;

    // The constant, user-dependent scaling factor c_u for the item bias.
    // Note that this is a column vector with numUsers entries. We
    // initialize this to 1 and also regularize it towards 1.
    fcolvec cUserConst;

    // The time-dependent, user-dependent scaling factor c_{ut} for the
    // item bias. Note that this is a sparse fmat with a shape of numTimes
    // x numUsers, where the uth column corresponds to the time-dependent
    // scaling factor entries for user u.
    sp_fmat cUserTime;

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

    // The constant (time-independent) user factor matrix. This is a
    // numFactors x numUsers matrix. The nth column represents the user
    // factor array p_n, using the convention of the BellKor paper.
    fmat userFacMat;

    // The modifying factor matrix for each user, referred to as
    // "alpha_{uk}" in the BellKor paper. This governs how much weight is
    // given to the effect of time on the user's factor vector. Note that
    // this is also a numFactors x numUsers matrix.
    fmat userFacMatAlpha;

    // A mapping from UserDates to a time-dependent user factor vector of
    // size numFactors. This is called p_{ut} in the BellKor paper. Note:
    // to see the format that this is stored in on disk, refer to
    // loadUserFacMatTime().
    std::unordered_map<UserDate, std::vector<float>, UserDateHasher> 
        userFacMatTime;

    // The time-independent item factor matrix. This is a numFactors x
    // numItems matrix. The nth column represents the item factor array
    // q_n, using the convention of the BellKor paper.
    fmat itemFacMat;

    // The time-bin-dependent item factor matrix q_{i, Bin(t)}. This is a
    // numFactors x numTimeBins x numItems dense fcube. The ith slice
    // represents the data for the ith item, and the bth column of this
    // slice is the factor vector to use for bin "b".
    fcube itemFacMatTimewise;

    // The frequency-dependent item factor matrix q_{i, f_{ut}}. This is a
    // numFactors x (MAX_F_U_T + 1) x numItems dense fcube. The ith slice
    // represents data for the ith item, and the fth column of this slice
    // is the factor vector to use for log-frequency "f".
    fcube itemFacMatFreq;
    
    // The time-independent "y" matrix. This is a numFactors x numItems
    // matrix. The jth column of this is "y_j" in the convention of the
    // BellKor paper; it is supposed to weight the implicit preferences of
    // the user (i.e. the preferences in N(u)).
    fmat yMat;

    // The time-dependent (binwise) component of the "y" matrix. This is a
    // dense numFactors x numTimeBins x numItems cube.
    // fcube yMatBinwise;

    // Whether the algorithm has been trained yet or not.
    bool trained = false;

    // Whether we'll include userFacMatTime in the training algorithm.
    bool includeUserFacMatTime;
    
    // Whether we're using cached data or not.
    bool usingCachedData = false;

    void initInternalData();
    void populateHatDevUT(const std::string &fileNameHatDevUT);
    void populateN(const std::string &fileNameN);
    void populateFUT(const std::string &fileNamePUT);
    void populateNumItemsTrainingSet(const fmat &data);
    void updateSumMovieWeights(int lowUserNum, int highUserNum);
    inline void updateUserSumMovieWeights(int user);
    void loadUserFacMatTime(const std::string &fileNameUserFacMatTime);
    void saveUserFacMatTime(const std::string &fileNameUserFacMatTime);
    float computeRMSE(const std::string &testFileName);

public:
    TimeSVDPP(int numUsers, int numItems, int numTimes, float meanRating,
              int numFactors, int numIterations, int numTimeBins,
              bool includeUserFacMatTime,
              const std::string &fileNameN,
              const std::string &fileNameHatDevUT,
              const std::string &fileNameFUT);
    
    TimeSVDPP(int numUsers, int numItems, int numTimes, float meanRating,
              int numFactors, int numIterations, int numTimeBins,
              bool includeUserFacMatTime,
              const std::string &fileNameN,
              const std::string &fileNameHatDevUT,
              const std::string &fileNameFUT,
              const std::string &fileNameBUserConst,
              const std::string &fileNameBUserAlpha,
              const std::string &fileNameBUserTime,
              const std::string &fileNameBItemConst,
              const std::string &fileNameBItemTimewise,
              const std::string &fileNameBItemFreq,
              const std::string &fileNameCUserConst,
              const std::string &fileNameCUserTime,
              const std::string &fileNameUserFacMat,
              const std::string &fileNameUserFacMatAlpha,
              const std::string &fileNameUserFacMatTime,
              const std::string &fileNameItemFacMat,
              const std::string &fileNameItemFacMatTimewise,
              const std::string &fileNameItemFacMatFreq,
              const std::string &fileNameYMat,
              const std::string &fileNameSumMovieWeights);
     
    ~TimeSVDPP();
    
    void train(const fmat &data);
    
    void trainAndCache(const fmat &data,
                       const std::string &fileNameBUserConst,
                       const std::string &fileNameBUserAlpha,
                       const std::string &fileNameBUserTime,
                       const std::string &fileNameBItemConst,
                       const std::string &fileNameBItemTimewise,
                       const std::string &fileNameBItemFreq,
                       const std::string &fileNameCUserConst,
                       const std::string &fileNameCUserTime,
                       const std::string &fileNameUserFacMat,
                       const std::string &fileNameUserFacMatAlpha,
                       const std::string &fileNameUserFacMatTime,
                       const std::string &fileNameItemFacMat,
                       const std::string &fileNameItemFacMatTimewise,
                       const std::string &fileNameItemFacMatFreq,
                       const std::string &fileNameYMat,
                       const std::string &fileNameSumMovieWeights);
    
    void trainAndCache(const std::string &fileNameData,
                       const std::string &fileNameBUserConst,
                       const std::string &fileNameBUserAlpha,
                       const std::string &fileNameBUserTime,
                       const std::string &fileNameBItemConst,
                       const std::string &fileNameBItemTimewise,
                       const std::string &fileNameBItemFreq,
                       const std::string &fileNameCUserConst,
                       const std::string &fileNameCUserTime,
                       const std::string &fileNameUserFacMat,
                       const std::string &fileNameUserFacMatAlpha,
                       const std::string &fileNameUserFacMatTime,
                       const std::string &fileNameItemFacMat,
                       const std::string &fileNameItemFacMatTimewise,
                       const std::string &fileNameItemFacMatFreq,
                       const std::string &fileNameYMat,
                       const std::string &fileNameSumMovieWeights);
    
    float predict(int user, int item, int date, bool bound);
};

#endif // TIMESVDPP_HH
