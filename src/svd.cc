#ifndef NDEBUG
#include <chrono>
#include <iostream>

using namespace std::chrono;
#endif

#include <svd.hh>


/** 
 * A constructor for this run of SVD. Note that this constructor does not
 * make use of previously cached data, and will need to be trained!
 *
 * @param numUsers:             Number of users in the entire data set (not
 *                              just training set).
 * @param numItems:             Number of items in the entire data set (not
 *                              just training set).
 * @param meanRating:           The mean rating of items in the training
 *                              set.
 * @param numFactors:           The number of factors to use for the SVD.
 * @param numIterations:        The number of iterations to use for SVD.
 *
 */
SVD::SVD(int numUsers, int numItems, float meanRating, int numFactors,
         int numIterations) :
    numUsers(numUsers), numItems(numItems), meanRating(meanRating),
    numFactors(numFactors), numIterations(numIterations), bUser(numUsers),
    bItem(numItems), userFacMat(numFactors, numUsers),
    itemFacMat(numFactors, numItems), numItemsTrainingSet(numUsers)
{
    // Initialize bUser, bItem, userFacMat, and itemFacMat.
    initInternalData();

#ifndef NDEBUG
    cout << "Initialized data for SVD predictor.\n" << endl;
#endif
}


/**
 * This constructor uses cached data to initialize the internals of the
 * SVD object. It is assumed that all of the cached data is stored in
 * Armadillo's machine-dependent binary format.
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
 *                                  SVD.
 * @param fileNameBUser:            Name of file containing data for bUser,
 *                                  in Armadillo's machine-dependent binary
 *                                  format.
 * @param fileNameBItem:            Same as above, but for bItem.
 * @param fileNameUserFacMat:       Same as above, but for userFacMat.
 * @param fileNameItemFacMat:       Same as above, but for itemFacMat.
 *
 */
SVD::SVD(int numUsers, int numItems, float meanRating, int numFactors,
         int numIterations,
         const string &fileNameBUser, const string &fileNameBItem,
         const string &fileNameUserFacMat,
         const string &fileNameItemFacMat) :
    numUsers(numUsers), numItems(numItems), meanRating(meanRating),
    numFactors(numFactors), numIterations(numIterations), bUser(numUsers),
    bItem(numItems), userFacMat(numFactors, numUsers),
    itemFacMat(numFactors, numItems), numItemsTrainingSet(numUsers)
{
    // Initialize bUser, bItem, userFacMat, and itemFacMat by reading from
    // their binary files.
    bUser.load(fileNameBUser, arma_binary);
    bItem.load(fileNameBItem, arma_binary);
    userFacMat.load(fileNameUserFacMat, arma_binary);
    itemFacMat.load(fileNameItemFacMat, arma_binary);
     
    trained = true;
    usingCachedData = true;

#ifndef NDEBUG
    cout << "Created SVD predictor using cached data." << endl;
#endif
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
void SVD::populateNumItemsTrainingSet(const fmat &data)
{
    for(unsigned int i = 0; i < data.n_cols; i++)
    {
        // Based on the user that this rating was by, increment the
        // appropriate element of numItemsTrainingSet.
        int user = roundToInt(data(USER_ROW, i));
        
        numItemsTrainingSet(user) ++;
    }
}


/**
 *
 * This function initializes the internal data in this SVD object.
 * Currently, randomization is turned on.
 *
 * See post #36 on 
 *  http://www.netflixprize.com/community/viewtopic.php?id=1359&p=2
 *
 */
void SVD::initInternalData()
{
    // Different distributions based on the matrix being initialized.
    //uniform_real_distribution<float> distrBUser(-0.01, 0.1);
    //uniform_real_distribution<float> distrBItem(-0.5, -0.1);
    //uniform_real_distribution<float> distrUserFacMat(-0.01, -0.002);
    //uniform_real_distribution<float> distrItemFacMat(0.01, 0.02);

    uniform_real_distribution<float> coinFlip(-1.0, 1.0);

    // Set the seed to a sequence of random numbers that's large enough to
    // fill the mt19937's state.
    array<int, mt19937::state_size> seedData;
    random_device r;
    generate_n(seedData.data(), seedData.size(), ref(r));
    seed_seq seedSeq(begin(seedData), end(seedData));
    
    // Mersenne twister random number engine, based on the earlier seed.
    mt19937 engine(seedSeq);

    srand(time(NULL));
    
    userFacMat.imbue( [&]()
            {
                // Modified from post #55 of 
                // http://www.netflixprize.com/community/viewtopic.php?id=1342&p=3
                // Note that copysign(1.0, coinFlip) gives a random sign
                // (either -1 or +1) with 50% probability of each case.
                return (rand() % 4500 + 500) * 0.000001235 *
                       copysign(1.0, coinFlip(engine)); 
            });
    itemFacMat.imbue( [&]()
            {
                // Modified from same source as above.
                return (rand() % 4500 + 500) * 0.000001235 *
                       copysign(1.0, coinFlip(engine)); 
            });
    
    //bUser.imbue( [&]() { return distrBUser(engine); } );
    //bItem.imbue( [&]() { return distrBItem(engine); } );
    //userFacMat.imbue( [&]() { return distrUserFacMat(engine); } );
    //itemFacMat.imbue( [&]() { return distrItemFacMat(engine); } );

    // This is the count of the number of items rated by users in the given
    // training set. We'll set this to zero for now.
    numItemsTrainingSet.zeros();

    // Don't worry about sum_{j in N(u)} y_j (i.e. sumMovieWeights) for
    // now, since this will be set up while training.
    bUser.zeros();
    bItem.zeros();
}


/** 
 * This function trains on a given set of data, and then caches the
 * internal data of this SVD object.
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
 * 
 */
void SVD::trainAndCache(const fmat &data, const string &fileNameBUser,
                        const string &fileNameBItem,
                        const string &fileNameUserFacMat,
                        const string &fileNameItemFacMat)
{
    // Train the SVD algorithm, then save internal data to file.
    train(data);
    
    bUser.save(fileNameBUser, arma_binary);
    bItem.save(fileNameBItem, arma_binary);
    userFacMat.save(fileNameUserFacMat, arma_binary);
    itemFacMat.save(fileNameItemFacMat, arma_binary);
    
#ifndef NDEBUG
    cout << "Saved bUser to " << fileNameBUser << endl;
    cout << "Saved bItem to " << fileNameBItem << endl;
    cout << "Saved userFacMat to " << fileNameUserFacMat << endl;
    cout << "Saved itemFacMat to " << fileNameItemFacMat << endl;
#endif
}


/**
 * This function also trains and caches, but it first loads a file from
 * fileNameData. This file must be an Armadillo binary of an fmat.
 *
 * @param fileNameData: The file where "data" is stored. This binary file
 *                      must hold matrix data in the format specified in
 *                      the train(const fmat &data) function.
 *
 * The other params are the same as in the other trainAndCache()
 * function.
 *
 */
void SVD::trainAndCache(const string &fileNameData,
                        const string &fileNameBUser,
                        const string &fileNameBItem,
                        const string &fileNameUserFacMat,
                        const string &fileNameItemFacMat)
{
    fmat data;
    
    data.load(fileNameData, arma_binary);
    trainAndCache(data, fileNameBUser, fileNameBItem, fileNameUserFacMat,
                  fileNameItemFacMat);
}


/**
 * This function uses the given training data in order to set up all of the
 * internal matrices needed for SVD. After training has been completed,
 * the "trained" boolean will be set to true.
 *
 * @param data: This is the training data to use for our algorithm. This
 *              must be a 4 x N matrix, where N is the total number of
 *              ratings in the training set. NOTE: The first column must
 *              contain user IDs, the second column must contain item IDs,
 *              the third column must contain date IDs, and the last column
 *              must contain the rating the user gave.  All of these are
 *              assumed to be floats.
 *
 * Precondition: "data" should be in column-major order as stated above.
 * Also, the users should be sorted by their user ID (i.e. no shuffling
 * should take place).
 *
 */

void SVD::train(const fmat &data)
{
    // The predicted rating given by SVD++ for user u and item i is:
    //
    //      rHat_{ui} = mu + b_u + b_i + q_i^T * p_u
    //
    // Where we use the same naming convention as in the Koren paper. The
    // goal of this training procedure is to minimize the following
    // function with respect to q_*, p_*, y_*, and b_*:
    //
    //      min sum_{(u, i) in K} ( (r_{ui} - rHat_{ui})^2 +
    //                              SVD_LAM_B_U * b_u^2 +
    //                              SVD_LAM_B_I * b_i^2 +
    //                              SVD_LAM_Q_I * |q_i|^2 +
    //                              SVD_LAM_P_U * |p_u|^2 )
    //
    // Where "K" is the training set and r_{ui} is the actual rating that
    // the user gave. The regularization terms here are used to prevent
    // overfitting.
    //
    // This minimization is accomplished via stochastic gradient descent on
    // the free parameters of b_u, b_i, q_i, and p_u.
    
    // Check that the data does in fact have four rows!
    if (data.n_rows != 4)
    {
        throw invalid_argument("Data array must have four rows!");
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
            // Find the number of items rated by this user in the training
            // set, so we know how many entries to parse.
            int numItemsUserTrainSet = numItemsTrainingSet[user];
            
            // Increment ratingNum as we iterate over items rated by the
            // user.
            for(int itemNum = 0; itemNum < numItemsUserTrainSet; itemNum++,
                                                                 ratingNum++)
            {
                int item = roundToInt(data(MOVIE_ROW, ratingNum));
                float actualRating = data(RATING_ROW, ratingNum);
                
                // Get the predicted rating for this user and item, using the
                // aforementioned formula for rHat_{ui}.
                float predictedRating = meanRating + bUser(user) + bItem(item);
                
                // Compute the factorized term (i.e. q_i^T * p_u).
                fcolvec userFactorTerm(userFacMat.col(user));
                fcolvec qi(itemFacMat.col(item));
                predictedRating += dot(qi, userFactorTerm);
                
                // Apply gradient descent on all of the free parameters in
                // our algorithm. This just involves subtracting off the
                // gradient of the error metric (which we're trying to
                // minimize) with respect to each free parameter. Note
                // that factors of 2 have been absorbed into the "gamma"
                // step sizes.
                
                // The error in our prediction for this user and item.
                float eUI = actualRating - predictedRating;

                // b_u <- b_u + gamma_b_u * (e_{ui} - SVD_LAM_B_U * b_u)
                bUser(user) += SVD_GAMMA_B_U * (eUI - SVD_LAM_B_U *
                                                bUser(user));
                
                // b_i <- b_i + gamma_b_i * (e_{ui} - SVD_LAM_B_I * b_i)
                bItem(item) += SVD_GAMMA_B_I * (eUI - SVD_LAM_B_I *
                                                bItem(item));

                // q_i <- q_i + gamma_2 * (e_{ui} * p_u
                //                         - SVD_LAM_Q_I * q_i)
                itemFacMat.col(item) += SVD_GAMMA_Q_I * (eUI * 
                        userFactorTerm - SVD_LAM_Q_I * qi);
                
                // p_u <- p_u + gamma_2 * (e_{ui} * q_i - SVD_LAM_P_U * 
                //                                        p_u)
                userFacMat.col(user) += SVD_GAMMA_P_U * (eUI * qi - 
                        SVD_LAM_P_U * userFactorTerm);
            }
            
        }
        
        // At the end of each iteration, decrease the gammas by the
        // constant factor declared in the header file.
        SVD_GAMMA_B_U *= SVD_GAMMA_MULT_PER_ITER;
        SVD_GAMMA_B_I *= SVD_GAMMA_MULT_PER_ITER;
        SVD_GAMMA_Q_I *= SVD_GAMMA_MULT_PER_ITER;
        SVD_GAMMA_P_U *= SVD_GAMMA_MULT_PER_ITER; 
        
        
#ifndef NDEBUG
        end = system_clock::now();
        minutes_elapsed = end - start;
        cout << "\nFinished iteration " << (iterCount + 1) << " of SVD "
             << "in " << minutes_elapsed.count() << " minutes" << endl;

        float probeRMSE = computeRMSE(PROBE_BIN);
        cout << "Probe RMSE: " << probeRMSE << endl;
#endif
    }

    trained = true;

#ifndef NDEBUG
    cout << endl;
#endif
}


/**
 *
 * TODO: remove this later!
 *
 * Compute the RMSE of this algorithm on a certain set of data. Note that
 * testFileName must refer to an **Armadillo binary** in this case. This
 * binary must represent a 4 x N matrix, where N is the number of test
 * points.
 *
 */
float SVD::computeRMSE(const string &testFileName)
{
    // Load from binary.
    fmat testSet;
    testSet.load(testFileName, arma_binary);

    // The test set should have four rows.
    if (testSet.n_rows != 4)
    {
        throw invalid_argument("File " + testFileName + " did not "
                               + "have four rows!");
    }

    // The number we divide by in computing the RMSE.
    int nMinusOne = testSet.n_cols - 1;

    // Accumulator for RMSE (take square root at the end)
    float rmse = 0.0;

    for (unsigned int i = 0; i < testSet.n_cols; i ++)
    {
        int user = roundToInt(testSet(USER_ROW, i));
        int item = roundToInt(testSet(MOVIE_ROW, i));
        int date = roundToInt(testSet(DATE_ROW, i));
        float actualRating = testSet(RATING_ROW, i);
        
        float prediction = predict(user, item, date, true);
        
        rmse += pow(actualRating - prediction, 2.0)/nMinusOne;
    }

    return sqrt(rmse);
}



/** 
 * This function predicts a rating for a given user and item. If the SVD
 * has not been trained yet, a logic_error is thrown.
 *
 * @param user:     the user ID of interest.
 * @param item:     the item ID of interest.
 * @param date:     the date ID of interest (not used in SVD).
 * @param bound:    whether to bound predictions between MIN_RATING and
 *                  MAX_RATING, or leave them as they are.
 *
 * @return A prediction of the user's rating for the given item. This will
 *         always end up being between MIN_RATING and MAX_RATING if "bound"
 *         is true.
 *
 * Precondition: It is assumed that sumMovieWeights has been accurately set
 * after training!
 *
 */
float SVD::predict(int user, int item, int date, bool bound)
{
    /*if (!trained)
    {
        throw logic_error("Tried to predict a rating but the SVD "
                          "algorithm was not done training!");
    }*/

    // The formula for the predicted rating for user u and item i is:
    //
    //      rHat_{ui} = mu + b_u + b_i + q_i^T * p_u
    //
    // Where we use the same naming convention as in the Koren paper.
    
    float predictedRating = meanRating + bUser(user) + bItem(item);

    // Compute the factorized term (i.e. q_i^T * p_u).
    fcolvec userFactorTerm(userFacMat.col(user)); // p_u
    fcolvec qi(itemFacMat.col(item));
    predictedRating += dot(qi, userFactorTerm);

    if (bound)
    {
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
    }

    return predictedRating;
}


SVD::~SVD()
{
    // No dynamically allocated resources to free at the moment.
}
