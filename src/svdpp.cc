#include "svdpp.hh"


/** 
 * A constructor for this run of SVD++.
 *
 * @param numUsers: Number of users in the entire data set (not just
 *                  training set).
 * @param numItems: Number of items in the entire data set (not just
 *                  training set).
 * @param meanRating: The mean rating of items in the training set.
 * @param numFactors: The number of factors to use for the SVD.
 * @param numIterations: The number of iterations to use for SVD++.
 * @param fileNameN: Name of the file that contains the information needed
 *                   to populate the N mapping.
 * @param verbose: If true, some print statements are outputted.
 *
 */
SVDPP::SVDPP(int numUsers, int numItems, float meanRating, int numFactors,
             int numIterations, const string &fileNameN, 
             bool verbose = false) :
    numUsers(numUsers), numItems(numItems), meanRating(meanRating),
    numFactors(numFactors), numIterations(numIterations), bUser(numUsers),
    bItem(numItems), userFacMat(numUsers, numFactors),
    itemFacMat(numItems, numFactors), yMat(numItems, numFactors),
    verbose(verbose)
{
    // Populate N by reading from fileNameN.
    ifstream fileN(fileNameN);
    string line;

    while (getline(fileN, line))
    {
        // Split the string around the specified delimiter.
        vector<int> thisLineVec;
        splitIntoInts(line, NETFLIX_FILES_DELIMITER, thisLineVec);
        
        // The first int should be the user's ID.
        int userID = thisLineVec[0];

        // The remaining ints should be the item IDs that the user gave
        // "implicit feedback" on (without actually rating them). We want
        // to subtract 1 from these IDs since item IDs are **one-indexed**,
        // which is slightly inconvenient for our purposes...
        vector<int> userImplFeedbackItems;
        
        for(vector<int>::size_type i = 1; i < thisLineVec.size(); i++)
        {
            // Subtract 1 to get rid of one-indexing
            userImplFeedbackItems.push_back(thisLineVec[i] - 1);
        }
        
        // Add this data to the map N. Subtract off 1 because of the
        // userID!
        N[userID - 1] = userImplFeedbackItems;
    }

    fileN.close();

    // Before performing stochastic gradient descent, we should fill bUser,
    // bItem, userFacMat, itemFacMat, and yMat with uniformly distributed
    // values ranging from -0.05 to 0.05. This is the "stochastic" part.
    randomizeInternalData();

}


// TODO: Write a constructor that can work with cached matrix data so we
// can accommodate blending in the future... This constructor should also
// set "trained" to true at the end.


/**
 * This function randomizes the internal data in this SVDPP object.
 * Specifically, all of the items in bUser, bItem, userFacMat, itemFacMat,
 * and yMat are set to uniformly distributed random numbers ranging from
 * -0.05 to 0.05.
 *
 */
void SVDPP::randomizeInternalData()
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

    // If we've already trained on a previous dataset, we should reset all
    // of the internal data to random values.
    if (trained)
    {
        randomizeInternalData();
        
        if (verbose)
        {
            cout << "Cleared old internal data" << endl;
        }
    }

    // Iterate for the specified number of iterations.
    for(int iterCount = 0; iterCount < numIterations; iterCount++)
    {
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

        if (verbose)
        {
            cout << "Finished iteration " << (iterCount+1) << " of SVD++" 
                << endl;
        }
    }

    if (verbose)
    {
        cout << endl;
    }

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

    float predictedRating = meanRating;
    predictedRating += bUser(user);
    predictedRating += bItem(item);

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
