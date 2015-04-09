#include "svdpp.cc"


/** 
 * A constructor for this run of SVD++.
 *
 * @param numUsers: Number of users in the training set.
 * @param numItems: Number of items in the training set.
 * @param meanRating: The mean rating of items in the training set.
 * @param numFactors: The number of factors to use for the SVD.
 * @param numIterations: The number of iterations to use for SVD++.
 * @param fileNameN: Name of the file that contains the information needed
 *                   to populate the N mapping.
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
    ifstream fileN(fileNameN);
    string line;

    while (getline(fileN, line))
    {
        // Split the string around whitespaces.
        vector<int> thisLineVec;
        splitIntoInts(line, DELIMITER, thisLineVec);
        
        // The first int should be the user's ID.
        int userID = thisLineVec[0];

        // The remaining ints should be the item IDs that the user gave
        // "implicit feedback" on (without actually rating them).
        vector<int> userImplFeedbackItems;
        
        for(vector<int>::size_type i = 1; i < thisLineVec.size(); i++)
        {
            userImplFeedbackItems.push_back(thisLineVec[i]);
        }
        
        // Add this data to the map N.
        N[userID] = userImplFeedbackItems;
    }

    fileN.close();

    // Before performing stochastic gradient descent, we should fill bUser,
    // bItem, userFacMat, itemFacMat, and yMat with uniformly distributed
    // values ranging from -0.1 to 0.1. This is the "stochastic" part.
    uniform_real_distribution<float> distr(-0.1, 0.1);

    // Mersenne twister random number engine (default params).
    mt19937 engine;
    
    bUser.imbue( [&]() { return distr(engine); } );
    bItem.imbue( [&]() { return distr(engine); } );
    userFacMat.imbue( [&]() { return distr(engine); } );
    itemFacMat.imbue( [&]() { return distr(engine); } );
    yMat.imbue( [&]() { return distr(engine); } );
}


// TODO: Write a constructor that can work with cached matrix data so we
// can accommodate blending in the future... This constructor should also
// set "trained" to true at the end.


SVDPP::~SVDPP()
{
    // No dynamically allocated resources to free at the moment.
}


/**
 * TODO: refactor this into a separate file containing convenience
 * functions...
 *
 * This function splits a string around an input delimiter string. The
 * parts of the string between delimiters are converted into ints, and are
 * returned in a vector of ints.
 *
 * @param str: The input string.
 * @param delimiter: The delimiter string that separates data.
 * @param output: A vector containing the ints that were in the original
 *                string, after separating around "delimiter".
 *
 */
void SVDPP::splitIntoInts(const string &str, const string &delimiter,
                          vector<int> &output)
{
    string::size_type start = 0;
    string::size_type delimPos = str.find(delimiter);
    string::size_type length = 0;

    // Keep adding elements to the vector as long as the delimiter is found.
    while (delimPos != string::npos)
    {
        length = delimPos - start;
        
        // Convert the substring to an integer and add to the vector.
        output.push_back(stoi(str.substr(start, length)));
        
        // Go look for the next delimiter after this last one.
        start = delimPos + 1;
        delimPos = str.find(delimiter, start);
        
        // If there are no more delimiters left, add in the remainder of
        // the string (from "start" to the end of the string).
        if (delimPos == string::npos)
        {
            output.push_back(stoi(str.substr(start, str.length())));
        }
    }
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

    // Iterate for the specified number of iterations.
    for(int iterCount = 0; iterCount < numIterations; iterCount++)
    {
        // Iterate through all elements of the training data, and predict a
        // rating. Use gradient descent to correct the relevant matrices.
        for(int ratingNum = 0; ratingNum < data.n_rows; ratingNum ++)
        {
            // The user ID, item ID, and rating
            int user = data(ratingNum, 0);
            int item = data(ratingNum, 1);
            int actualRating = data(ratingNum, 2);

            // Get the predicted rating for this user and item, using the
            // aforementioned formula for rHat_{ui}.
            float predictedRating = meanRating;
            predictedRating += bUser(user);
            predictedRating += bItem(item);

            // Compute the factorized term (i.e. q_i^T * (p_u + ...)).
            // First find p_u + |N(u)|^{-1/2} sum_{j in N(u)} y_j, the
            // "userFactorTerm".
            frowvec userFactorTerm = userFacMat.row(user); // p_u
            
            vector<int> nu = N[user];
            float nuNormFac = nu.size()

            for (vector<int>::size_type j = 0; j < nu.size(); j++)
            {
                
            }


        }
        
        
        
        // At the end of each iteration, decrease the gammas by the
        // constant factor declared in the header file.
        SVDPP_GAMMA_1 *= SVDPP_GAMMA_MULT_PER_ITER;
        SVDPP_GAMMA_2 *= SVDPP_GAMMA_MULT_PER_ITER;
    }


    trained = true;
}


float SVDPP::predict(int user, int item)
{
    if (!trained)
    {
        throw logic_error("Tried to predict a rating but the SVD++ "
                          "algorithm was not done training!");
    }

    return 3.0;
}
