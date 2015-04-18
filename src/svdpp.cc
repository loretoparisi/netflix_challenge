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
    bItem(numItems), userFacMat(numFactors, numUsers),
    itemFacMat(numFactors, numItems), yMat(numFactors, numItems),
    numItemsTrainingSet(numUsers), sumMovieWeights(numFactors, numUsers)
{
    // Populate N by reading from fileNameN.
    populateN(fileNameN);
    
    // Initialize bUser, bItem, userFacMat, itemFacMat, and yMat.
    initInternalData();

#ifndef NDEBUG
    cout << "Initialized data for SVD++ predictor.\n" << endl;
#endif
}


/**
 * This constructor uses cached data to initialize the internals of the
 * SVDPP object. It is assumed that all of the cached data (except the file
 * containing N) is stored in Armadillo's machine-dependent binary format.
 *
 * Note: This constructor should be used for blending, not the other one!
 *
 * @param numUsers:                 Number of users in the entire data set
 *                                  (not just training set).
 * @param numItems:                 Number of items in the entire data set (not
 *                                  just training set).
 * @param meanRating:               The mean rating of items in the training
 *                                  set.
 * @param numFactors:               The number of factors to use for the
 *                                  SVD.
 * @param numIterations:            The number of iterations to use for
 *                                  SVD++.
 * @param fileNameN:                Name of the file that contains the
 *                                  information needed to populate the N
 *                                  mapping. This should be a plain text
 *                                  .dta file (or equivalent).
 * @param fileNameBUser:            Name of file containing data for bUser,
 *                                  in Armadillo's machine-dependent binary
 *                                  format.
 * @param fileNameBItem:            Same as above, but for bItem.
 * @param fileNameUserFacMat:       Same as above, but for userFacMat.
 * @param fileNameItemFacMat:       Same as above, but for itemFacMat.
 * @param fileNameYMat:             Same as above, but for yMat.
 * @param fileNameSumMovieWeights:  Same as above, but for sumMovieWeights.
 *
 */
SVDPP::SVDPP(int numUsers, int numItems, float meanRating, int numFactors,
             int numIterations, const string &fileNameN,
             const string &fileNameBUser, const string &fileNameBItem,
             const string &fileNameUserFacMat,
             const string &fileNameItemFacMat, const string &fileNameYMat,
             const string &fileNameSumMovieWeights) :
    numUsers(numUsers), numItems(numItems), meanRating(meanRating),
    numFactors(numFactors), numIterations(numIterations), bUser(numUsers),
    bItem(numItems), userFacMat(numFactors, numUsers),
    itemFacMat(numFactors, numItems), yMat(numFactors, numItems),
    numItemsTrainingSet(numUsers), sumMovieWeights(numFactors, numUsers)
{
    // Populate N by reading from fileNameN.
    populateN(fileNameN);

    // Initialize bUser, bItem, userFacMat, itemFacMat, yMat, and
    // sumMovieWeights by reading from their binary files.
    bUser.load(fileNameBUser, arma_binary);
    bItem.load(fileNameBItem, arma_binary);
    userFacMat.load(fileNameUserFacMat, arma_binary);
    itemFacMat.load(fileNameItemFacMat, arma_binary);
    yMat.load(fileNameYMat, arma_binary);
    sumMovieWeights.load(fileNameSumMovieWeights, arma_binary);
    
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
 * Given a training set, this function updates numItemsTrainingSet -- an
 * array that stores the number of items in the training set that a given
 * user rated.
 *
 * @param data: This is the training data to use for our algorithm. See
 *              train() for more details.
 *
 */
void SVDPP::populateNumItemsTrainingSet(const imat &data)
{
    for(unsigned int i = 0; i < data.n_cols; i++)
    {
        // Based on the user that this rating was by, increment the
        // appropriate element of numItemsTrainingSet.
        int user = data(USER_ROW, i);
        
        numItemsTrainingSet(user) ++;
    }
    
}


/**
 *
 * This function initializes the internal data in this SVDPP object.
 * Currently, randomization is turned on.
 *
 * See post #36 on 
 *  http://www.netflixprize.com/community/viewtopic.php?id=1359&p=2
 *
 */
void SVDPP::initInternalData()
{
    // Different distributions based on the matrix being initialized.
    uniform_real_distribution<float> distrBUser(-0.01, 0.1);
    uniform_real_distribution<float> distrBItem(-0.5, -0.1);
    uniform_real_distribution<float> distrUserFacMat(-0.01, -0.002);
    uniform_real_distribution<float> distrItemFacMat(0.01, 0.02);
    uniform_real_distribution<float> distrYMat(0.0, 0.1);

    // Set the seed to a sequence of random numbers that's large enough to
    // fill the mt19937's state.
    array<int, mt19937::state_size> seedData;
    random_device r;
    generate_n(seedData.data(), seedData.size(), ref(r));
    seed_seq seedSeq(begin(seedData), end(seedData));
    
    // Mersenne twister random number engine, based on the earlier seed.
    mt19937 engine(seedSeq);
    
    bUser.imbue( [&]() { return distrBUser(engine); } );
    bItem.imbue( [&]() { return distrBItem(engine); } );
    userFacMat.imbue( [&]() { return distrUserFacMat(engine); } );
    itemFacMat.imbue( [&]() { return distrItemFacMat(engine); } );
    yMat.imbue( [&]() { return distrItemFacMat(engine); } );

    // This is the count of the number of items rated by users in the given
    // training set. We'll set this to zero for now.
    numItemsTrainingSet.zeros();

    // Don't worry about sum_{j in N(u)} y_j (i.e. sumMovieWeights) for
    // now, since this will be set up while training.
     
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
 * @param data:                     This is the training data to use for
 *                                  our algorithm. See train() for more
 *                                  details.
 * @param fileNameBUser:            Name of file where we'll save bUser (in
 *                                  Armadillo's machine-dependent binary
 *                                  format).
 * @param fileNameBItem:            Same as above, but for bItem.
 * @param fileNameUserFacMat:       Same as above, but for userFacMat.
 * @param fileNameItemFacMat:       Same as above, but for itemFacMat.
 * @param fileNameYMat:             Same as above, but for yMat.
 * @param fileNameSumMovieWeights:  Same as above, but for sumMovieWeights.
 * 
 */
void SVDPP::trainAndCache(const imat &data, const string &fileNameBUser,
                          const string &fileNameBItem,
                          const string &fileNameUserFacMat,
                          const string &fileNameItemFacMat,
                          const string &fileNameYMat,
                          const string &fileNameSumMovieWeights)
{
    // Train the SVD++ algorithm, then save internal data to file.
    train(data);

    bUser.save(fileNameBUser, arma_binary);
    bItem.save(fileNameBItem, arma_binary);
    userFacMat.save(fileNameUserFacMat, arma_binary);
    itemFacMat.save(fileNameItemFacMat, arma_binary);
    yMat.save(fileNameYMat, arma_binary);
    sumMovieWeights.save(fileNameSumMovieWeights, arma_binary);
    
#ifndef NDEBUG
    cout << "Saved bUser to " << fileNameBUser << endl;
    cout << "Saved bItem to " << fileNameBItem << endl;
    cout << "Saved userFacMat to " << fileNameUserFacMat << endl;
    cout << "Saved itemFacMat to " << fileNameItemFacMat << endl;
    cout << "Saved yMat to " << fileNameYMat << endl;
    cout << "Saved sumMovieWeights to " << fileNameSumMovieWeights << endl;
#endif
}


/**
 * This function updates the sum of movie weights (i.e. sum_{j in N(u)}
 * y_j) for each user u between lowUserNum (inclusive) and highUserNum
 * (exclusive). It is assumed that both of these user IDs are valid!
 *
 * @param lowUserNum:   The lower bound on user IDs to update.
 * @param highUserNum:  The upper bound (exclusive) on user IDs to update.
 *
 */
void SVDPP::updateSumMovieWeights(int lowUserNum, int highUserNum)
{
    // Iterate over all users, get N[u], and compute.
    for(int user = lowUserNum; user < highUserNum; user++)
    {
        updateUserSumMovieWeights(user);
    }
}


/**
 * This function updates the sum of movie weights (i.e. sum_{j in N(u)}
 * y_j) for a single user.
 *
 * @param user:     The user ID of interest.
 *
 */
inline void SVDPP::updateUserSumMovieWeights(int user)
{
    // Get N[u] and compute the desired sum.
    vector<int> nu = N[user];
    
    // Each column in sumMovieWeights has numFactors rows.
    fcolvec sumColVec = zeros<fcolvec>(numFactors);

    for (vector<int>::size_type ind = 0; ind < nu.size(); ind++)
    {
        int j = nu[ind];
        sumColVec += yMat.col(j);
    }

    sumMovieWeights.col(user) = sumColVec;
}


/**
 * This function uses the given training data in order to set up all of the
 * internal matrices needed for SVD++. After training has been completed,
 * the "trained" boolean will be set to true.
 *
 * @param data: This is the training data to use for our algorithm. This
 *              must be a 3 x N matrix, where N is the total number of
 *              ratings in the training set. NOTE: The first column must
 *              contain user IDs, the second column most contain item IDs,
 *              and the last column must contain the rating the user gave.
 *              All of these are assumed to be integers.
 *
 * Precondition: "data" should be in column-major order as stated above.
 * Also, the users should be sorted by their user ID (i.e. no shuffling
 * should take place).
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
    //                              SVDPP_LAM_B_U * b_u^2 +
    //                              SVDPP_LAM_B_I * b_i^2 +
    //                              SVDPP_LAM_Q_I * |q_i|^2 +
    //                              SVDPP_LAM_P_U * |p_u|^2 + 
    //                              SVDPP_LAM_Y_J * sum_{j in N(u)} |y_j|^2 )
    //
    // Where "K" is the training set and r_{ui} is the actual rating that
    // the user gave. The regularization terms here are used to prevent
    // overfitting.
    //
    // This minimization is accomplished via stochastic gradient descent on
    // the free parameters of b_u, b_i, q_i, p_u, and y_j.
    
    // Check that the data does in fact have three rows!
    if (data.n_rows != 3)
    {
        throw invalid_argument("Data array must have three rows!");
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

    
    // We want to find the number of items rated by each user in the
    // training set, since this will help us go through our training data
    // in a more organized fashion.
    populateNumItemsTrainingSet(data);
    
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
        
        // The rating number that we're looking at right now (i.e. the
        // column in our training set).
        unsigned int ratingNum = 0;
        
        // Iterate through all users in the training data. We're assuming
        // that the data is sorted (column-wise) by user ID!
        for (unsigned int user = 0; user < (unsigned int) numUsers;
             user ++)
        {
            // Update sumMovieWeights for this user.
            updateUserSumMovieWeights(user);

            // Check N(u) to see if this user has implicit feedback data.
            vector<int> nu = N[user];
            int nuSize = nu.size();
            
            if (nuSize == 0)
            {
                // If they don't, ignore them. After all, we're not gonna
                // predict anything for them anyways.
                continue;
            }
            
            // Norm factor put in front of userSumMovieWeights, etc.
            float nuNormFac = 1.0/sqrt((float) nu.size());
            
            // Find the number of items rated by this user in the training
            // set, so we know how many entries to parse.
            int numItemsUserTrainSet = numItemsTrainingSet[user];

            // The value of sum_{j in N(u)} y_j for this user.
            fcolvec userSumMovieWeights(sumMovieWeights.col(user));
            
            // The sum of all values of e_{ui} |N(u)|^{-1/2} * q_i over all
            // items watched by this user. This is used to update yMat via
            // gradient descent at the very end.
            fcolvec sumErrNuNormQi = zeros<fcolvec>(numFactors);
            
            // Increment ratingNum as we iterate over items rated by the
            // user.
            for(int itemNum = 0; itemNum < numItemsUserTrainSet; itemNum++,
                                                                 ratingNum++)
            {
                int item = data(ITEM_ROW, ratingNum);
                int actualRating = data(RATING_ROW, ratingNum);
                
                // Get the predicted rating for this user and item, using the
                // aforementioned formula for rHat_{ui}.
                float predictedRating = meanRating + bUser(user) + bItem(item);
                
                // Compute the factorized term (i.e. q_i^T * (p_u + ...)).
                // First find p_u + |N(u)|^{-1/2} sum_{j in N(u)} y_j, the
                // "userFactorTerm". Start off by making a copy of
                // p_u.
                fcolvec userFactorTerm(userFacMat.col(user));

                // sumMovieWeights should already have sum_{j in N(u)} y_j
                // cached (from the previous iteration), so use that old
                // value.
                userFactorTerm += userSumMovieWeights * nuNormFac;
                
                // Add the factorized term (q_i^T * userFactorTerm) to the
                // prediction.
                fcolvec qi(itemFacMat.col(item));
                predictedRating += dot(qi, userFactorTerm);

                // Apply gradient descent on all of the free parameters in our
                // algorithm EXCEPT FOR yMat (which only needs to be updated at
                // the end for this user). This just involves subtracting off
                // the gradient of the error metric (which we're trying to
                // minimize) with respect to each free parameter. Note that
                // factors of 2 have been absorbed into the "gamma" step
                // sizes.
                
                // The error in our prediction for this user and item.
                float eUI = (float) actualRating - predictedRating;

                // b_u <- b_u + gamma_b_u * (e_{ui} - SVDPP_LAM_B_U * b_u)
                bUser(user) += SVDPP_GAMMA_B_U * (eUI - SVDPP_LAM_B_U *
                                                      bUser(user));
                
                // b_i <- b_i + gamma_b_i * (e_{ui} - SVDPP_LAM_B_I * b_i)
                bItem(item) += SVDPP_GAMMA_B_I * (eUI - SVDPP_LAM_B_I *
                                                        bItem(item));

                // q_i <- q_i + gamma_2 * (e_{ui} * (p_u + |N(u)|^{-1/2} *
                //                                   sum_{j in N(u)} y_j)
                //                         - SVDPP_LAM_Q_I * q_i)
                itemFacMat.col(item) += SVDPP_GAMMA_Q_I * (eUI * 
                        userFactorTerm - SVDPP_LAM_Q_I * qi);
                
                // p_u <- p_u + gamma_2 * (e_{ui} * q_i - SVDPP_LAM_P_U * 
                //                                        p_u)
                userFacMat.col(user) += SVDPP_GAMMA_P_U * (eUI * qi - 
                        SVDPP_LAM_P_U * userFacMat.col(user));
                
                // Ideally, for all j in N(u) (for each rating), we'd want
                // to set:
                //
                // y_j <- y_j + SVDPP_GAMMA_Y_J * (e_{ui} |N(u)|^{-1/2} * q_i
                //                                  - SVDPP_LAM_Y_J * y_j)
                // 
                // However, repeatedly changing all y_j for j in N(u) is
                // very expensive. Instead, we just note that the term
                // e_{ui} |N(u)|^{-1/2} * q_i is independent of j, and so
                // we can actually update yMat's columns at the very end by
                // adding the sum of all e_{ui} |N(u)|^{-1/2} * q_i
                // (multiplied by the learning rate). This is what
                // sumErrNuNormQi is. Of course, we also need to modify the
                // regularization constant on y_j since we're adding a much
                // bigger quantity on each SGD update step.
                //
                // This is pretty hacky and not going to give an accurate
                // result as per the gradient. But it's fast.
                //
                // For now, just update sumErrNuNormQi.
                sumErrNuNormQi += eUI * nuNormFac * qi;
                
            }

            
            // Go through each item in N[u] and update yMat for those
            // columns. Don't update sumMovieWeights for this user yet;
            // that'll happen on the next iteration.
            for (vector<int>::size_type ind = 0; ind < nu.size(); ind++)
            {
                int j = nu[ind];
                yMat.col(j) += SVDPP_GAMMA_Y_J * (sumErrNuNormQi -
                                                SVDPP_LAM_Y_J * yMat.col(j));
            }

#if 0
            if (user % 10000 == 0)
            {
                cout << "Finished processing user #" << user << "." 
                     << endl;
            }
#endif
        }

        // At the end of each iteration, decrease the gammas by the
        // constant factor declared in the header file.
        SVDPP_GAMMA_B_U *= SVDPP_GAMMA_MULT_PER_ITER;
        SVDPP_GAMMA_B_I *= SVDPP_GAMMA_MULT_PER_ITER;
        SVDPP_GAMMA_Q_I *= SVDPP_GAMMA_MULT_PER_ITER;
        SVDPP_GAMMA_P_U *= SVDPP_GAMMA_MULT_PER_ITER; 
        SVDPP_GAMMA_Y_J *= SVDPP_GAMMA_MULT_PER_ITER;


#ifndef NDEBUG
        end = system_clock::now();
        minutes_elapsed = end - start;
        cout << "Finished iteration " << (iterCount + 1) << " of SVD++ in " 
             << minutes_elapsed.count() << " minutes" << endl;
#endif
    }

    // Update sumMovieWeights for the last time, so that the data used by
    // predict() (and the data cached to file) is accurate!
    updateSumMovieWeights(0, numUsers);

    trained = true;

#ifndef NDEBUG
    cout << endl;
#endif
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
 * Precondition: It is assumed that sumMovieWeights has been accurately set
 * after training!
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
    fcolvec userFactorTerm(userFacMat.col(user)); // p_u
    
    vector<int> nu = N[user];
    float nuNormFac = 1.0/sqrt(nu.size());

    // Get sum_{j in N(u)} y_j and multiply by nuNormFac.
    userFactorTerm += sumMovieWeights.col(user) * nuNormFac;

    // Add the factorized term (q_i^T * userFactorTerm) to the prediction.
    fcolvec qi(itemFacMat.col(item));
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
