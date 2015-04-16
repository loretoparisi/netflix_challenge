#ifndef NDEBUG
#include <chrono>
#include <iostream>

using namespace std::chrono;
#endif

#include "svdpp.hh"


/** 
 * A constructor for this run of SVD++. Note that this constructor does not
 * make use of previously cached data, and will need to be trained!
 *
 * @param numUsers:             Number of users in the entire data set (not
 *                              just training set).
 * @param numItems:             Number of items in the entire data set (not
 *                              just training set).
 * @param meanRating:           The mean rating of items in the training
 *                              set.
 * @param numFactors:           The number of factors to use for the SVD.
 * @param numIterations:        The number of iterations to use for SVD++.
 * @param fileNameN:            Name of the file that contains the
 *                              information needed to populate the N
 *                              mapping. This should be a plain text .dta
 *                              file (or equivalent).
 *
 */
SVDPP::SVDPP(int numUsers, int numItems, float meanRating, int numFactors,
             int numIterations, const string &fileNameN) :
    numUsers(numUsers), numItems(numItems), meanRating(meanRating),
    numFactors(numFactors), numIterations(numIterations), bUser(numUsers),
    bItem(numItems), userFacMat(numUsers, numFactors),
    itemFacMat(numItems, numFactors), yMat(numItems, numFactors)
{
    // Populate N by reading from fileNameN.
    populateN(fileNameN);
    
    // Initialize bUser, bItem, userFacMat, itemFacMat, and yMat.
    initInternalData();

}


/**
 * This constructor uses cached data to initialize the internals of the
 * SVDPP object. It is assumed that all of the cached data (except the file
 * containing N) is stored in Armadillo's machine-dependent binary format.
 *
 * Note: This constructor should be used for blending, not the other one!
 *
 * @param numUsers:             Number of users in the entire data set (not
 *                              just training set).
 * @param numItems:             Number of items in the entire data set (not
 *                              just training set).
 * @param meanRating:           The mean rating of items in the training
 *                              set.
 * @param numFactors:           The number of factors to use for the SVD.
 * @param numIterations:        The number of iterations to use for SVD++.
 * @param fileNameN:            Name of the file that contains the
 *                              information needed to populate the N
 *                              mapping. This should be a plain text .dta
 *                              file (or equivalent).
 * @param fileNameBUser:        Name of file containing data for bUser, in
 *                              Armadillo's machine-dependent binary
 *                              format.
 * @param fileNameBItem:        Same as above, but for bItem.
 * @param fileNameUserFacMat:   Same as above, but for userFacMat.
 * @param fileNameItemFacMat:   Same as above, but for itemFacMat.
 * @param fileNameYMat:         Same as above, but for yMat.
 *
 */
SVDPP::SVDPP(int numUsers, int numItems, float meanRating, int numFactors,
             int numIterations, const string &fileNameN,
             const string &fileNameBUser, const string &fileNameBItem,
             const string &fileNameUserFacMat,
             const string &fileNameItemFacMat, const string &fileNameYMat) :
    numUsers(numUsers), numItems(numItems), meanRating(meanRating),
    numFactors(numFactors), numIterations(numIterations), bUser(numUsers),
    bItem(numItems), userFacMat(numUsers, numFactors),
    itemFacMat(numItems, numFactors), yMat(numItems, numFactors)
{
    // Populate N by reading from fileNameN.
    populateN(fileNameN);

    // Initialize bUser, bItem, userFacMat, itemFacMat, and yMat by reading
    // from their binary files.
    bUser.load(fileNameBUser, arma_binary);
    bItem.load(fileNameBItem, arma_binary);
    userFacMat.load(fileNameUserFacMat, arma_binary);
    itemFacMat.load(fileNameItemFacMat, arma_binary);
    yMat.load(fileNameYMat, arma_binary);
    
    trained = true;
    usingCachedData = true;

#ifndef NDEBUG
    cout << "Created SVD++ predictor using cached data." << endl;
#endif
}


/**
 * This function populates N (the mapping from zero-indexed user IDs to the
 * zero-indexed item IDs which that user has shown an implicit preference
 * for).
 *
 * @param fileNameN: Name of the file that contains the information needed
 *                   to populate the N mapping. This should be a plain text
 *                   .dta file (or equivalent).
 *
 */
void SVDPP::populateN(const string &fileNameN)
{
    ifstream fileN(fileNameN);
    string line;

    while (getline(fileN, line))
    {
        // Split the string around the specified delimiter.
        vector<int> thisLineVec;
        splitIntoInts(line, NETFLIX_FILES_DELIMITER, thisLineVec);
        
        // The first int should be the user's ID. This should be
        // zero-indexed!
        int userID = thisLineVec[0];

        // The remaining ints should be the item IDs that the user gave
        // "implicit feedback" on (without actually rating them). These
        // item IDs should be zero-indexed!
        vector<int> userImplFeedbackItems;
        
        for(vector<int>::size_type i = 1; i < thisLineVec.size(); i++)
        {
            userImplFeedbackItems.push_back(thisLineVec[i]);
        }
        
        // Add this data to the map N.
        N[userID] = userImplFeedbackItems;
    }

    fileN.close();
}


/**
 *
 * This function initializes the internal data in this SVDPP object.
 * Currently, randomization is turned on, so this sets all of the
 * elements in bUser, bItem, userFacMat, itemFacMat, and yMat equal to
 * uniformly distributed random numbers between -0.05 and 0.05.
 *
 */
void SVDPP::initInternalData()
{
    uniform_real_distribution<float> distr(-0.05, 0.05);
    
    // Set the seed to a sequence of random numbers that's large enough to
    // fill the mt19937's state.
    array<int, mt19937::state_size> seedData;
    random_device r;
    generate_n(seedData.data(), seedData.size(), ref(r));
    seed_seq seedSeq(begin(seedData), end(seedData));
    
    // Mersenne twister random number engine, based on the earlier seed.
    mt19937 engine(seedSeq);
    
    bUser.imbue( [&]() { return distr(engine); } );
    bItem.imbue( [&]() { return distr(engine); } );
    userFacMat.imbue( [&]() { return distr(engine); } );
    itemFacMat.imbue( [&]() { return distr(engine); } );
    yMat.imbue( [&]() { return distr(engine); } );
   
    // Uncomment the lines below to set all of the data to zero instead.
    /*
    bUser.zeros();
    bItem.zeros();
    userFacMat.zeros();
    itemFacMat.zeros();
    yMat.zeros();
    */
}


/** 
 * This function trains on a given set of data, and then caches the
 * internal data of this SVDPP object.
 *
 * @param data:                 This is the training data to use for our
 *                              algorithm. See train() for more details.
 * @param fileNameBUser:        Name of file where we'll save bUser (in
 *                              Armadillo's machine-dependent binary
 *                              format).
 * @param fileNameBItem:        Same as above, but for bItem.
 * @param fileNameUserFacMat:   Same as above, but for userFacMat.
 * @param fileNameItemFacMat:   Same as above, but for itemFacMat.
 * @param fileNameYMat:         Same as above, but for yMat.
 * 
 */
void SVDPP::trainAndCache(const imat &data, const string &fileNameBUser,
                          const string &fileNameBItem,
                          const string &fileNameUserFacMat,
                          const string &fileNameItemFacMat,
                          const string &fileNameYMat)
{
    // Train the SVD++ algorithm, then save internal data to file.
    train(data);

    bUser.save(fileNameBUser, arma_binary);
    bItem.save(fileNameBItem, arma_binary);
    userFacMat.save(fileNameUserFacMat, arma_binary);
    itemFacMat.save(fileNameItemFacMat, arma_binary);
    yMat.save(fileNameYMat, arma_binary);

#ifndef NDEBUG
    cout << "Saved bUser to " << fileNameBUser << endl;
    cout << "Saved bItem to " << fileNameBItem << endl;
    cout << "Saved userFacMat to " << fileNameUserFacMat << endl;
    cout << "Saved itemFacMat to " << fileNameItemFacMat << endl;
    cout << "Saved yMat to " << fileNameYMat << endl;
#endif
}


/**
 * This function uses the given training data in order to set up all of the
 * internal matrices needed for SVD++. After training has been completed,
 * the "trained" boolean will be set to true.
 *
 * @param data: This is the training data to use for our algorithm. This
 *              must be an N x 3 matrix, where N is the total number of
 *              ratings in the training set. NOTE: The first column must
 *              contain user IDs, the second column most contain item IDs,
 *              and the last column must contain the rating the user gave.
 *              All of these are assumed to be integers.
 *
 * Precondition: "data" should be shuffled before the stochastic gradient
 * descent runs.
 *
 */

void SVDPP::train(const imat &data)
{
    // The predicted rating given by SVD++ for user u and item i is:
    //
    //      rHat_{ui} = mu + b_u + b_i + 
    //                  q_i^T * (p_u + |N(u)|^{-1/2} sum_{j in N(u)} y_j)
    //
    // Where we use the same naming convention as in the Koren paper. The
    // goal of this training procedure is to minimize the following
    // function with respect to q_*, p_*, y_*, and b_*:
    //
    //      min sum_{(u, i) in K} ( (r_{ui} - rHat_{ui})^2 +
    //                              SVDPP_REG_1 * (b_u^2 + b_i^2) +
    //                              SVDPP_REG_2 * (|q_i|^2 + |p_u|^2 + 
    //                                          sum_{j in N(u)} |y_j|^2)) )
    //
    // Where "K" is the training set and r_{ui} is the actual rating that
    // the user gave. The regularization terms here are used to prevent
    // overfitting.
    //
    // This minimization is accomplished via stochastic gradient descent on
    // the free parameters of b_u, b_i, q_i, p_u, and y_j.
    
    // Check that the data does in fact have three columns!
    if (data.n_cols != 3)
    {
        throw invalid_argument("Data array must have three columns!");
    }

    // If we're using cached data, we shouldn't be calling this method!
    if (usingCachedData)
    {
        throw logic_error("This algorithm shouldn't be trained if you're "
                          "using cached data!");
    }
        
    // If we've already trained on a previous dataset, we should reset all
    // of the internal data.
    if (trained)
    {
        initInternalData();
        
#ifndef NDEBUG
        cout << "Cleared old internal data" << endl;
#endif
    }
    
#ifndef NDEBUG
    time_point<system_clock> start, end;
    duration<float, ratio<60>> minutes_elapsed; 
#endif
    
    // Iterate for the specified number of iterations.
    for(int iterCount = 0; iterCount < numIterations; iterCount++)
    {
#ifndef NDEBUG
        start = system_clock::now();
#endif
        // Iterate through all elements of the training data, and predict a
        // rating. Use gradient descent to correct the relevant matrices.
        for(unsigned int ratingNum = 0; ratingNum < data.n_rows; 
            ratingNum ++)
        {
            // The user ID, item ID, and rating
            int user = data(ratingNum, 0);
            int item = data(ratingNum, 1);
            int actualRating = data(ratingNum, 2);
            
            // Check N(u) to see if this user has implicit feedback data.
            vector<int> nu = N[user];
            
            if (nu.size() == 0)
            {
                // If they don't, ignore them. After all, we're not gonna
                // predict anything for them anyways.
                continue;
            }

            // Get the predicted rating for this user and item, using the
            // aforementioned formula for rHat_{ui}.
            float predictedRating = meanRating + bUser(user) + bItem(item);

            // Compute the factorized term (i.e. q_i^T * (p_u + ...)).
            // First find p_u + |N(u)|^{-1/2} sum_{j in N(u)} y_j, the
            // "userFactorTerm".
            frowvec userFactorTerm = userFacMat.row(user); // p_u

            float nuNormFac = 1.0/sqrt(nu.size());

            // This is where we'll store |N(u)|^{-1/2} sum_{j in N(u)} y_j.
            frowvec implicitTerm = zeros<frowvec>(numFactors);

            for (vector<int>::size_type ind = 0; ind < nu.size(); ind++)
            {
                int j = nu[ind];
                implicitTerm += yMat.row(j);
            }

            implicitTerm *= nuNormFac;

            // Now we've computed p_u + |N(u)|^{-1/2} sum_{j in N(u)} y_j.
            userFactorTerm += implicitTerm;

            // Add the factorized term (q_i^T * ...) to the prediction.
            frowvec qi = itemFacMat.row(item);
            predictedRating += dot(qi, userFactorTerm);
            
            // Apply gradient descent on all of the free parameters in our
            // algorithm. This just involves subtracting off the gradient
            // of the error metric (which we're trying to minimize) with
            // respect to each free parameter. Note that factors of 2 have
            // been absorbed into the "gamma" step sizes.
            
            // The error in our prediction for this user and item.
            float eUI = (float) actualRating - predictedRating;

            // b_u <- b_u + gamma_1 * (e_{ui} - SVDPP_REG_1 * b_u)
            bUser(user) += SVDPP_GAMMA_1 * (eUI - SVDPP_REG_1 *
                                                  bUser(user));
            
            // b_i <- b_i + gamma_1 * (e_{ui} - SVDPP_REG_1 * b_i)
            bItem(item) += SVDPP_GAMMA_1 * (eUI - SVDPP_REG_1 *
                                                  bItem(item));

            // q_i <- q_i + gamma_2 * (e_{ui} * (p_u + |N(u)|^{-1/2} *
            //                                   sum_{j in N(u)} y_j)
            //                         - SVDPP_REG_2 * q_i)
            itemFacMat.row(item) += SVDPP_GAMMA_2 * (eUI * userFactorTerm -
                                                     SVDPP_REG_2 * qi);
            
            // p_u <- p_u + gamma_2 * (e_{ui} * q_i - SVDPP_REG_2 * p_u)
            userFacMat.row(user) += SVDPP_GAMMA_2 * (eUI * qi - 
                                                     SVDPP_REG_2 *
                                                     userFacMat.row(user));

            // For all j in N(u), we want to set:
            // y_j <- y_j + SVDPP_GAMMA_2 * (e_{ui} |N(u)|^{-1/2} * q_i
            //                               - SVDPP_REG_2 * y_j)
            for (vector<int>::size_type ind = 0; ind < nu.size(); ind++)
            {
                int j = nu[ind];
                yMat.row(j) += SVDPP_GAMMA_2 * (eUI * nuNormFac * qi -
                                                SVDPP_REG_2 * yMat.row(j));
            }

        }
        
        
        // At the end of each iteration, decrease the gammas by the
        // constant factor declared in the header file.
        SVDPP_GAMMA_1 *= SVDPP_GAMMA_MULT_PER_ITER;
        SVDPP_GAMMA_2 *= SVDPP_GAMMA_MULT_PER_ITER;

#ifndef NDEBUG
        end = system_clock::now();
        minutes_elapsed = end - start;
        cout << "Finished iteration " << (iterCount + 1) << " of SVD++ in " 
             << minutes_elapsed.count() << " minutes" << endl;
#endif
    }

#ifndef NDEBUG
    cout << endl;
#endif

    trained = true;
}

/** 
 * This function predicts a rating for a given user and item. If the SVDPP
 * has not been trained yet, a logic_error is thrown.
 *
 * @param user: the user ID of interest.
 * @param item: the item ID of interest.
 *
 * @return A prediction of the user's rating for the given item. This will
 *         always end up being between MIN_RATING and MAX_RATING.
 *
 */
float SVDPP::predict(int user, int item)
{
    if (!trained)
    {
        throw logic_error("Tried to predict a rating but the SVD++ "
                          "algorithm was not done training!");
    }

    // The formula for the predicted rating for user u and item i is:
    //
    //      rHat_{ui} = mu + b_u + b_i + 
    //                  q_i^T * (p_u + |N(u)|^{-1/2} sum_{j in N(u)} y_j)
    //
    // Where we use the same naming convention as in the Koren paper.
    
    float predictedRating = meanRating + bUser(user) + bItem(item);

    // Compute the factorized term (i.e. q_i^T * (p_u + ...)).
    // First find p_u + |N(u)|^{-1/2} sum_{j in N(u)} y_j, the
    // "userFactorTerm".
    frowvec userFactorTerm = userFacMat.row(user); // p_u
    
    vector<int> nu = N[user];
    float nuNormFac = 1.0/sqrt(nu.size());

    // This is where we'll store |N(u)|^{-1/2} sum_{j in N(u)} y_j.
    frowvec implicitTerm = zeros<frowvec>(numFactors);

    for (vector<int>::size_type ind = 0; ind < nu.size(); ind++)
    {
        int j = nu[ind];
        implicitTerm += yMat.row(j);
    }

    implicitTerm *= nuNormFac;

    // Now we've computed p_u + |N(u)|^{-1/2} sum_{j in N(u)} y_j.
    userFactorTerm += implicitTerm;

    // Add the factorized term (q_i^T * ...) to the prediction. 
    frowvec qi = itemFacMat.row(item);
    predictedRating += dot(qi, userFactorTerm);

    // Put the rating between MIN_RATING and MAX_RATING! Otherwise, the
    // error will be bad.
    if (predictedRating < MIN_RATING)
    {
        predictedRating = (float) MIN_RATING;
    }
    else if (predictedRating > MAX_RATING)
    {
        predictedRating = (float) MAX_RATING;
    }

    return predictedRating;
}


SVDPP::~SVDPP()
{
    // No dynamically allocated resources to free at the moment.
}
