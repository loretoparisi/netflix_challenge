#include "knn.hh"

/** 
 * A constructor for this run of KNN. Note that this constructor does not
 * make use of previously cached data, and will need to be trained!
 *
 * @param numUsers:             Number of users in the entire data set (not
 *                              just training set).
 * @param numItems:             Number of items in the entire data set (not
 *                              just training set).
 * @param trainFilename:        Name of the file that contains the
 *                              training data set. This should be a plain
 *                              text .dta file (or equivalent).
 * @param qualFilename:         Name of the file that contains the
 *                              testing data set. This should be a plain
 *                              text .dta file (or equivalent).
 * @param outputFilename:       Name of the path that we wish to store.
 * @param test:                 Indicating whether the "qual" data is
 *                              merely for testing (aka has "answers") or
 *                              we do not have answers and wish to output.
 *
 */
KNN::KNN(int numUsers, int numItems, const string &trainFilename,
    const string &qualFilename, const string &outputFilename, bool test) :
    numUsers(numUsers), numItems(numItems), qualFilename(qualFilename), 
    outputFilename(outputFilename)
{
    initInternalData(trainFilename);
}

void KNN::initInternalData(const string &trainFilename)
{
    ifstream fileTrain(trainFilename);
    string line;
    int userId, movieId, rating, latestUser = 0, movieCount = 0;
    float sum = 0.0;

    while (getline(fileTrain, line))
    {
        // Split the string around the specified delimiter.
        vector<int> thisLineVec;
        splitIntoInts(line, NETFLIX_FILES_DELIMITER, thisLineVec);

        userId = thisLineVec[0];
        movieId = thisLineVec[1];
        rating = thisLineVec[3];
        rateMatrix[userId][movieId] = rating;
        if (userId != latestUser)
        {
            averageUser[latestUser] = sum / movieCount;
            sum = 0.0;
            movieCount = 0;
            latestUser = userId;
        }
        sum += rating;
        movieCount += 1;
    }

    cout << "Finished building rating matrix." << endl;

    userUser.resize(numUsers, numUsers);

    fileTrain.close();
}

float KNN::simPearson(int userId1, int userId2)
{
    if (userId1 == userId2) return 1;
    unordered_map<int, float> common;
    unordered_map<int, int>::const_iterator sim;
    for (auto movie = rateMatrix[userId1].begin();
        movie != rateMatrix[userId1].end(); movie++) {
        sim = rateMatrix[userId2].find(movie->first);
        if (sim != rateMatrix[userId2].end())
            common[movie->first] = 1;
    }

    int n = common.size();
    if (n == 0) return 0;

    // Means of users.
    float mean1 = averageUser[userId1], mean2 = \
        averageUser[userId2];
  
    // Sums of the squares
    unordered_map<int, float>::const_iterator it;
    float sum1Sq = 0, sum2Sq = 0, pSum = 0;
    for (auto item = common.begin();
        item != common.end(); item++)
    {
        sum1Sq += pow(rateMatrix[userId1][item->first] - mean1, 2);
        sum2Sq += pow(rateMatrix[userId2][item->first] - mean2, 2);
    }

    // Sum of the products
    for (auto item = common.begin();
        item != common.end(); item++)
    {
        pSum += (rateMatrix[userId1][item->first] - mean1) * \
            (rateMatrix[userId2][item->first] - mean2);
    }

    // Calculate Pearson score
    float den = sqrt(sum1Sq * sum2Sq);
    if (den == 0) return 0;

    return pSum / den;
}

void KNN::eachUser()
{
    float similarity;
    for(int u1 = 0; u1 < numUsers; u1++)
    {
        for (int u2 = 0; u2 < numUsers; u2++)
        {
            similarity = simPearson(u1, u2);
            userUser(u1, u2) = similarity;
        }
    }
    cout << "Finished building the giant user-user matrix." << endl;
}

float KNN::predict(int userId, int movieId)
{
    float simSum = 0;
    float total  = 0;
    float sim = 0, predictResult = 0;
    unordered_map<int, int>::const_iterator rated;
    for (int user = 0; user < numUsers; user++)
    {
        // Don't compare the user to himself
        if (user == userId) continue;
        sim = userUser(userId, user);
        
        // Ignore scores of zero or lower.
        if (sim <= 0) continue;

        rated = rateMatrix[user].find(movieId);
        if (rated == rateMatrix[user].end())
            continue;
        // Only calculate those who have rated.
        total += (rateMatrix[user][movieId] - averageUser[user]) * sim;
        // Sum of similarities
        simSum += sim;
    }

    // Create the normalized list
    if (simSum != 0)
        predictResult = averageUser[userId] + (1 / simSum) * total;
    else
        predictResult = averageUser[userId];

    return predictResult;
}

void KNN::beginKNN()
{
    clock_t time;
    stringstream fname;
    fname << outputFilename;
    ofstream out (fname.str().c_str(), ios::trunc); 
    ifstream fileQual(qualFilename);
    string line;
    int userId, movieId, count = 0;
    float predictedRating;

    // if "test" mode
    float ae = 0, se = 0;
    int testNum = 0;

    while (getline(fileQual, line))
    {
        // Split the string around the specified delimiter.
        vector<int> thisLineVec;
        splitIntoInts(line, NETFLIX_FILES_DELIMITER, thisLineVec);

        userId = thisLineVec[0];
        movieId = thisLineVec[1];
        predictedRating = predict(userId, movieId);
        out << predictedRating << '\n';
        count += 0;
        if (count % 100000 == 0)
            cout << "Completed " << count << " lines." << endl;
        if (test)
        {
            ae += abs(predictedRating - (float)thisLineVec[3]);
            se += pow((predictedRating - (float)thisLineVec[3]), 2);
            testNum += 1;
        }

    }

    fileQual.close();

    // Output time and RMSE info if on probe set.
    time = clock() - time;
    if (test) {
        cout << "MAE is: " << (ae / testNum) << endl;
        cout << "RMSE is: " << (se / testNum) << endl;
    }
    cout << "Time Spent: " << ((float) time) / CLOCKS_PER_SEC / 60 << endl;

}

void KNN::train(const imat &data) {}

void KNN::run()
{
    eachUser();
    beginKNN();
}

KNN::~KNN()
{
    // No dynamically allocated resources to free at the moment.
}
