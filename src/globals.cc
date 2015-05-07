#include <globals.hh>

// Initialize.
Globals::Globals (int numUsers, int numItems,
                  int levels, const std::string &trainFileName) :
                  numUsers(numUsers), numItems(numItems),
                  level(levels), numUsersTrainingSet(numItems),
                  numItemsTrainingSet(numUsers)
{
    dataMU.load(trainFileName, arma_binary);
    level = levels; // Default level.
    // Initialize and fill first date vectors with high numbers.
    userFirstDates.resize(numUsers);
    movieFirstDates.resize(numItems);
    userLastDates.resize(numUsers);
    movieLastDates.resize(numItems);
    std::fill(userFirstDates.begin(), userFirstDates.end(), 999999);
    std::fill(movieFirstDates.begin(), movieFirstDates.end(), 999999);
    std::fill(userLastDates.begin(), userLastDates.end(), 0);
    std::fill(movieLastDates.begin(), movieLastDates.end(), 0);
    initInternalData();
}

//  Level 2 or less: movieAverages and userAverages vectors.
//  Level 2.5: movieFirstDates and userFirstDates vectors,
//             but not the sqrt time averages.
//  Level 3 or higher: sqrt time averages.
bool Globals::setAverages(const fmat &dataUM)
{
    fprintf(stderr, "Setting global averages.\n");
    movieAverages.clear();
    userAverages.clear();
    movieUserAverages.clear();
    movieUserSupportAverages.clear();
    userMovieSupportAverages.clear();
    float globalsum = 0;
    float sqrtmoviesum = 0;
    int curr = 0;
    for(int i = 0; i < numItems; i++)
    {
        int count = numUsersTrainingSet[i];
        sqrtmoviesum += std::sqrt(count);
        float sum = 0.0;
        int votedatesum = 0;
        for(int j = 0; j < count; j++)
        {
            float rating = roundToInt(dataMU(RATING_ROW, curr));
            sum += rating;
            if(level > 2)
            {
                int votedate = roundToInt(dataMU(DATE_ROW, curr));
                votedatesum += votedate;
                if(votedate < movieFirstDates.at(i))
                    movieFirstDates.at(i) = votedate;
                if(votedate > movieLastDates.at(i))
                    movieLastDates.at(i) = votedate;
            }
            curr++;
        }
        float average = (float)sum / count;
        globalsum += sum;
        movieAverages.push_back(average);
        if(level > 2)
        {
            float votedateaverage = (float) votedatesum / count;
            movieAverageDates[i] = votedateaverage;
        }
    }
    sqrtMovieCountAverage = sqrtmoviesum / numItems;
    globalAverage = globalsum / dataUM.n_cols;
    curr = 0;
    float sqrtusersum = 0;
    for(int i = 0; i < numUsers; i++)
    {
        int count = numItemsTrainingSet[i];
        sqrtusersum += std::sqrt(count);
        float sum = 0.0;
        int votedatesum = 0;
        for(int j = 0; j < count; j++)
        {
            float rating = roundToInt(dataUM(RATING_ROW, curr));
            sum += rating;
            if(level > 2)
            {
                int votedate = roundToInt(dataUM(DATE_ROW, curr));
                votedatesum += votedate;
                if(votedate<userFirstDates.at(i))
                    userFirstDates.at(i) = votedate;
                if(votedate>userLastDates.at(i))
                    userLastDates.at(i) = votedate;
            }
            curr++;
        }
        float average = (float)sum / count;
        userAverages.push_back(average);
        if(level > 2)
        {
            float votedateaverage = (float) votedatesum / count;
            userAverageDates[i] = votedateaverage;
        }
    }
    sqrtUserCountAverage = sqrtusersum / numUsers;

    //  Break out of the function if the level is not 3 or higher
    if(level < 3)
    {
        fprintf(stderr, "Done setting global averages. Average: %f\n",
            globalAverage);
        return true;
    }

    // Iterate over again, now the min dates are set so we
    // can calculate average time differences.
    float sqrtMovieTimeMovieSum = 0;
    float sqrtMovieTimeUserSum = 0;
    cout << "Start calculating sqrtMovieTimeSum..." << endl;
    curr = 0;
    for(int i = 0; i< numItems; i++)
    {
        int count = numUsersTrainingSet[i];
        for(int j = 0; j < count; j++){
            int votedate = roundToInt(dataMU(DATE_ROW, curr));
            int userindex = roundToInt(dataMU(USER_ROW, curr));
            sqrtMovieTimeMovieSum +=
                std::sqrt( votedate - movieFirstDates.at(i) );
            sqrtMovieTimeUserSum +=
                std::sqrt( votedate - userFirstDates.at(userindex) );
            curr++;
        }
    }
    sqrtMovieTimeMovieAverage = sqrtMovieTimeMovieSum / dataUM.n_cols;
    sqrtMovieTimeUserAverage = sqrtMovieTimeUserSum / dataUM.n_cols;
    float sqrtUserTimeUserSum = 0;
    float sqrtUserTimeMovieSum = 0;
    curr = 0;
    for(int i=0; i < numUsers; i++)
    {
        int count = numItemsTrainingSet[i];
        for(int j = 0; j < count; j++)
        {
            int votedate = roundToInt(dataUM(DATE_ROW, curr));
            int movie = roundToInt(dataUM(MOVIE_ROW, curr));
            sqrtUserTimeUserSum += 
                std::sqrt( votedate - userFirstDates.at(i) );
            sqrtUserTimeMovieSum += 
                std::sqrt(votedate - movieFirstDates.at(movie) );
            curr++;
        }
    }
    sqrtUserTimeUserAverage = sqrtUserTimeUserSum / dataUM.n_cols;
    sqrtUserTimeMovieAverage = sqrtUserTimeMovieSum / dataUM.n_cols;
    fprintf(stderr, "Done setting global averages.\n");
    fprintf(stderr, "sqrtMovieTimeMovieAverage: %f\n"
        "sqrtMovieTimeUserAverage: %f\nsqrtUserTimeUserAverage: %f\n"
        "sqrtUserTimeMovieAverage: %f\n", sqrtMovieTimeMovieAverage,
        sqrtMovieTimeUserAverage, sqrtUserTimeUserAverage,
        sqrtUserTimeMovieAverage);
    cout << "Start calculating user averages of movie..." << endl;
    curr = 0;
    for(int i = 0; i < numItems; i++)
    {
        int count = numUsersTrainingSet[i];
        float sum = 0.0;
        for(int j = 0; j < count; j++)
        {
            int currUser = roundToInt(dataMU(USER_ROW, curr));
            sum += userAverages.at(currUser);
            curr++;
        }
        float average = (float)sum / count;
        movieUserAverages.push_back(average);
    }
    cout << "Start calculating averages of user and movie support..." << endl;
    curr = 0;
    for(int i = 0; i < numItems; i++)
    {
        int count = numUsersTrainingSet[i];
        float sum = 0.0;
        for(int j = 0; j < count; j++)
        {
            int currUser = roundToInt(dataMU(USER_ROW, curr));
            sum += numItemsTrainingSet[currUser];
            curr++;
        }
        float average = (float)sum / count;
        movieUserSupportAverages.push_back(std::sqrt(average));
    }
    curr = 0;
    for(int i = 0; i < numUsers; i++)
    {
        int count = numItemsTrainingSet[i];
        float sum = 0.0;
        for(int j = 0; j < count; j++)
        {
            int currItem = roundToInt(dataMU(MOVIE_ROW, curr));
            sum += numUsersTrainingSet[currItem];
            curr++;
        }
        float average = (float)sum / count;
        userMovieSupportAverages.push_back(std::sqrt(average));
    }
    return true;
}

/**
 * Given a training set, this function updates numItemsTrainingSet -- an
 * array that stores the number of items in the training set that a given
 * user rated.
 */
void Globals::populateNumItemsTrainingSet(const fmat &data)
{
    cout << "Populated numItemsTrainingSet." << endl;
    for(unsigned int i = 0; i < data.n_cols; i++)
    {
        // Based on the user that this rating was by, increment the
        // appropriate element of numItemsTrainingSet.
        int user = roundToInt(data(USER_ROW, i));
        numItemsTrainingSet(user)++;
    }
}

/**
 * Given a training set, this function updates numUsersTrainingSet -- an
 * array that stores the number of users in the training set of a given
 * movie.
 */
void Globals::populateNumUsersTrainingSet(const fmat &data)
{
    cout << "Populated numUsersTrainingSet." << endl;
    for(unsigned int i = 0; i < data.n_cols; i++)
    {
        // Based on the movie that this rating was by, increment the
        // appropriate element of numUsersTrainingSet.
        int movie = roundToInt(data(MOVIE_ROW, i));
        numUsersTrainingSet(movie)++;
    }
}

void Globals::initInternalData()
{
    // This is the count of the number of items rated by users
    // and number of users for each item in the given training
    // set. We'll set this to zero for now.
    numItemsTrainingSet.zeros();
    numUsersTrainingSet.zeros();
}

void Globals::setVariances(const fmat &dataUM){
    movieVariances.clear();
    userVariances.clear();
    int curr = 0;
    for(int i = 0; i < numItems; i++)
    {
        int count = numUsersTrainingSet[i];
        float varraw = 0;
        for(int j = 0; j < count; j++)
        {
            float rating = roundToInt(dataMU(RATING_ROW, curr));
            varraw += pow(rating - movieAverages.at(i), 2);
            curr ++;
        }
        float variance = varraw / (count - 1);
        movieVariances.push_back(variance);
    }
    curr = 0;
    for(int i = 0; i < numUsers; i++)
    {
        int count = numItemsTrainingSet[i];
        float varraw = 0;
        for(int j = 0; j < count; j++)
        {
            float rating = roundToInt(dataUM(RATING_ROW, curr));
            varraw += pow(rating - userAverages.at(i), 2);
        }
        float variance = varraw / (count - 1);
        userVariances.push_back(variance);
    }
}

bool Globals::setThetas(const fmat &dataUM)
{
    movieThetas.clear();
    userThetas.clear();
    userMovieAverageThetas.clear();
    userMovieSupportThetas.clear();
    float xysum = 0;
    float xxsum = 0;
    int curr = 0;

    // Movie effect
    for(int i = 0; i < numItems; i++)
    {
        xysum = 0;
        xxsum = 0;
        int count = numUsersTrainingSet[i];
        for(int j = 0; j < count; j++)
        {
            float rating = roundToInt(dataMU(RATING_ROW, curr));
            int userid = roundToInt(dataMU(USER_ROW, curr));
            int movieid = roundToInt(dataMU(MOVIE_ROW, curr));
            float residual = rating - globalAverage;
            UserMovie thisUserMovie;
            thisUserMovie.userID = userid;
            thisUserMovie.movieID = (unsigned short) movieid;
            residualStorage[thisUserMovie] = residual;
            xysum += residual * 1;
            xxsum += 1 * 1;
            curr++;
        }
        float theta = 0;
        if(xxsum != 0) theta = xysum / xxsum;
        theta = count * theta / (count + LEVEL1_ALPHA);
        movieThetas.push_back(theta);
    }
    if(level <= 1) return true;

    // User effect
    curr = 0;
    for(int i = 0; i < numUsers; i++)
    {
        xysum = 0;
        xxsum = 0;
        int count = numItemsTrainingSet[i];
        for(int j = 0; j < count; j++)
        {
            int userid = roundToInt(dataUM(USER_ROW, curr));
            int movieid = roundToInt(dataUM(MOVIE_ROW, curr));
            UserMovie thisUserMovie;
            thisUserMovie.userID = userid;
            thisUserMovie.movieID = (unsigned short) movieid;
            float residual = residualStorage[thisUserMovie]
                - movieThetas.at(movieid);
            residualStorage[thisUserMovie] = residual;
            xysum += residual * 1;
            xxsum += 1 * 1;
            curr ++;
        }
        float theta = 0;
        if(xxsum != 0) theta = xysum / xxsum;
        theta = count*theta / (count + LEVEL2_ALPHA);
        //theta = log(count)*theta / (log(count+43));
        userThetas.push_back(theta);
    }
    if(level <= 2) return true;

    // User*Time(user)
    curr = 0;
    for(int i = 0; i < numUsers; i++)
    {
        xysum = 0;
        xxsum = 0;
        int count = numItemsTrainingSet[i];
        for(int j = 0; j < count; j++)
        {
            int userid = roundToInt(dataUM(USER_ROW, curr));
            int movieid = roundToInt(dataUM(MOVIE_ROW, curr));
            int votedate = roundToInt(dataUM(DATE_ROW, curr));
            UserMovie thisUserMovie;
            thisUserMovie.userID = userid;
            thisUserMovie.movieID = (unsigned short) movieid;
            float residual = residualStorage[thisUserMovie]
                - userThetas.at(i);
            residualStorage[thisUserMovie] = residual;
            float x = std::sqrt(votedate - userFirstDates.at(i))
                - sqrtUserTimeUserAverage;
            xysum += residual * x;
            xxsum += x * x;
            curr++;
        }
        float theta = 0;
        if(xxsum != 0) theta = xysum / xxsum;
        theta = count * theta / (count + LEVEL3_ALPHA);
        userTimeUserThetas.push_back(theta);
    }

    if(level <= 3) return true;

    // User*Time(movie)
    curr = 0;
    for(int i = 0; i < numUsers; i++)
    {
        xysum = 0;
        xxsum = 0;
        int count = numItemsTrainingSet[i];
        for(int j = 0; j < count; j++)
        {
            int userid = roundToInt(dataUM(USER_ROW, curr));
            int movieid = roundToInt(dataUM(MOVIE_ROW, curr));
            int votedate = roundToInt(dataUM(DATE_ROW, curr));
            UserMovie thisUserMovie;
            thisUserMovie.userID = userid;
            thisUserMovie.movieID = (unsigned short) movieid;
            float residual = residualStorage[thisUserMovie]
                - userTimeUserThetas.at(i)
                * ( std::sqrt(votedate-userFirstDates.at(i))
                    - sqrtUserTimeUserAverage);
            residualStorage[thisUserMovie] = residual;
            float x = std::sqrt(votedate - movieFirstDates.at(movieid))
                - sqrtUserTimeMovieAverage;
            xysum += residual * x;
            xxsum += x * x;
            curr++;
        }
        float theta = 0;
        if(xxsum != 0) theta = xysum / xxsum;
        theta = 0; //THIS DISABLES User*Time(movie). Comment out to enable.
        theta = count * theta / (count + LEVEL4_ALPHA);
        userTimeMovieThetas.push_back(theta);
    }
    if(level <= 4) return true;

    // Movie*Time(movie)
    curr = 0;
    for(int i = 0; i < numItems; i++)
    {
        xysum = 0;
        xxsum = 0;
        int count = numUsersTrainingSet[i];
        for(int j = 0; j < count; j++)
        {
            int userindex = roundToInt(dataMU(USER_ROW, curr));
            int movieid = roundToInt(dataMU(MOVIE_ROW, curr));
            int votedate = roundToInt(dataMU(DATE_ROW, curr));
            UserMovie thisUserMovie;
            thisUserMovie.userID = userindex;
            thisUserMovie.movieID = (unsigned short) movieid;
            float residual = residualStorage[thisUserMovie]
                - userTimeMovieThetas.at(userindex)
                * ( std::sqrt(votedate - movieFirstDates.at(i))
                    - sqrtUserTimeMovieAverage );
            residualStorage[thisUserMovie] = residual;
            float x = std::sqrt(votedate - movieFirstDates.at(i))
                - sqrtMovieTimeMovieAverage;
            xysum += residual * x;
            xxsum += x * x;
            curr ++;
        }
        float theta = 0;
        if(xxsum != 0) theta = xysum / xxsum;
        theta = count*theta / (count + LEVEL5_ALPHA);
        movieTimeMovieThetas.push_back(theta);
    }
    if(level <= 5) return true;

    //  Movie*Time(user)
    curr = 0;
    for(int i = 0; i < numItems; i++)
    {
        xysum = 0;
        xxsum = 0;
        int count = numUsersTrainingSet[i];
        for(int j = 0; j < count; j++)
        {
            int userindex = roundToInt(dataMU(USER_ROW, curr));
            int movieid = roundToInt(dataMU(MOVIE_ROW, curr));
            int votedate = roundToInt(dataMU(DATE_ROW, curr));
            UserMovie thisUserMovie;
            thisUserMovie.userID = userindex;
            thisUserMovie.movieID = (unsigned short) movieid;
            float residual = residualStorage[thisUserMovie]
                - movieTimeMovieThetas.at(i)
                * ( std::sqrt(votedate-movieFirstDates.at(i))
                    - sqrtMovieTimeMovieAverage );
            residualStorage[thisUserMovie] = residual;
            float x = std::sqrt(votedate - userFirstDates.at(userindex))
                - sqrtMovieTimeUserAverage;
            xysum += residual * x;
            xxsum += x * x;
            curr++;
        }
        float theta = 0;
        if(xxsum != 0) theta = xysum / xxsum;
        theta = count*theta / (count + LEVEL6_ALPHA);
        movieTimeUserThetas.push_back(theta);
    }
    if(level <= 6) return true;

    // User*Movie Average
    curr = 0;
    for(int i = 0; i < numUsers; i++)
    {
        xysum = 0;
        xxsum = 0;
        int count = numItemsTrainingSet[i];
        for(int j = 0; j < count; j++)
        {
            int userid = roundToInt(dataUM(USER_ROW, curr));
            int movieid = roundToInt(dataUM(MOVIE_ROW, curr));
            int votedate = roundToInt(dataUM(DATE_ROW, curr));
            UserMovie thisUserMovie;
            thisUserMovie.userID = userid;
            thisUserMovie.movieID = (unsigned short) movieid;
            float residual = residualStorage[thisUserMovie]
                - movieTimeUserThetas.at(movieid)
                * ( std::sqrt(votedate - userFirstDates.at(i))
                    - sqrtMovieTimeUserAverage );
            residualStorage[thisUserMovie] = residual;
            float x = movieAverages.at(movieid) - globalAverage;
            xysum += residual * x;
            xxsum += x * x;
            curr++;
        }
        float theta = 0;
        if(xxsum != 0) theta = xysum / xxsum;
        theta = count * theta / (count + LEVEL7_ALPHA);
        userMovieAverageThetas.push_back(theta);
    }

    if(level <= 7) return true;

    // User*Movie Support
    curr = 0;
    for(int i = 0; i < numUsers; i++)
    {
        xysum = 0;
        xxsum = 0;
        int count = numItemsTrainingSet[i];
        for(int j=0; j < count; j++)
        {
            int userid = roundToInt(dataUM(USER_ROW, curr));
            int movieid = roundToInt(dataUM(MOVIE_ROW, curr));
            UserMovie thisUserMovie;
            thisUserMovie.userID = userid;
            thisUserMovie.movieID = (unsigned short) movieid;
            float residual = residualStorage[thisUserMovie]
                - userMovieAverageThetas.at(i)
                * (movieAverages.at(movieid) - globalAverage);
            residualStorage[thisUserMovie] = residual;
            float x = std::sqrt(numUsersTrainingSet[movieid])
                - userMovieSupportAverages.at(i);
            xysum += residual * x;
            xxsum += x * x;
            curr++;
        }
        float theta = 0;
        if(xxsum != 0) theta = xysum / xxsum;
        theta = count * theta / (count + LEVEL8_ALPHA);
        userMovieSupportThetas.push_back(theta);
    }

    if(level <= 8) return true;

    // Movie*User(Average)
    curr = 0;
    for(int i = 0; i < numItems; i++)
    {
        xysum = 0;
        xxsum = 0;
        int count = numUsersTrainingSet[i];
        for(int j = 0; j < count; j++)
        {
            int userindex = roundToInt(dataMU(USER_ROW, curr));
            int movieid = roundToInt(dataMU(MOVIE_ROW, curr));
            UserMovie thisUserMovie;
            thisUserMovie.userID = userindex;
            thisUserMovie.movieID = (unsigned short) movieid;
            float residual = residualStorage[thisUserMovie]
                - userMovieSupportThetas.at(userindex)
                * (std::sqrt(count)
                    - userMovieSupportAverages.at(userindex));
            residualStorage[thisUserMovie] = residual;
            float x = userAverages.at(userindex) - movieUserAverages.at(i);
            xysum += residual * x;
            xxsum += x * x;
            curr++;
        }
        float theta = 0;
        if(xxsum != 0) theta = xysum / xxsum;
        theta = count * theta / (count + LEVEL9_ALPHA);
        movieUserAverageThetas.push_back(theta);
    }
    if(level <= 9) return true;

    //  Movie*User(support)
    curr = 0;
    for(int i = 0; i < numItems; i++)
    {
        xysum = 0;
        xxsum = 0;
        int count = numUsersTrainingSet[i];
        for(int j = 0; j < count; j++)
        {
            int userindex = roundToInt(dataMU(USER_ROW, curr));
            int movieid = roundToInt(dataMU(MOVIE_ROW, curr));
            UserMovie thisUserMovie;
            thisUserMovie.userID = userindex;
            thisUserMovie.movieID = (unsigned short) movieid;
            float residual = residualStorage[thisUserMovie]
                - movieUserAverageThetas.at(i)
                * (userAverages.at(userindex) - movieUserAverages.at(i));
            residualStorage[thisUserMovie] = residual;
            float x = std::sqrt(numItemsTrainingSet[userindex])
                - movieUserSupportAverages.at(i);
            xysum += residual * x;
            xxsum += x*x;
            curr++;
        }
        float theta = 0;
        if(xxsum != 0) theta = xysum / xxsum;
        theta = count * theta / (count + LEVEL10_ALPHA);
        movieUserSupportThetas.push_back(theta);
    }
    return true;
}

void Globals::train(const fmat &data)
{
    populateNumItemsTrainingSet(data);
    populateNumUsersTrainingSet(dataMU);

    setAverages(data);
    setVariances(data);
    setThetas(data);
}

float Globals::predict(int user, int item, int date, bool bound)
{
    float pred;
    pred = globalAverage;
    if(level >= 1)
        pred += movieThetas.at(item);
    if(level >= 2)
        pred += userThetas.at(user);
    // The probe date can potentially be before the first date,
    // and can't take the square root of a negative
    if(level >= 3)
    {
        pred += userTimeUserThetas.at(user) * 
            (std::sqrt(std::max(date - userFirstDates.at(user), 0))
                - sqrtUserTimeUserAverage);
    }
    if(level >= 4)
    {
        pred += userTimeMovieThetas.at(user) * 
            (std::sqrt(std::max(date - movieFirstDates.at(item), 0))
                - sqrtUserTimeMovieAverage);
    }
    if(level >= 5)
    {
        pred += movieTimeMovieThetas.at(item) *
            (std::sqrt(std::max(date - movieFirstDates.at(item), 0))
                - sqrtMovieTimeMovieAverage);
    }
    if(level >= 6)
    {
        pred += movieTimeUserThetas.at(item) *
            (std::sqrt(std::max(date - userFirstDates.at(user), 0))
                - sqrtMovieTimeUserAverage);
    }
    if(level >= 7)
    {
        pred += userMovieAverageThetas.at(user) *
            (movieAverages.at(item) - globalAverage);
    }
    if(level >= 8)
    {
        pred += userMovieSupportThetas.at(user) * 
            (std::sqrt(numUsersTrainingSet[item])
                - userMovieSupportAverages.at(user));
    }
    if(level >= 9)
    {
        pred += movieUserAverageThetas.at(item) * (userAverages.at(user)
            - movieUserAverages.at(item));
    }
    if(level >= 10)
    {
        pred += movieUserSupportThetas.at(item) *
            (std::sqrt(numItemsTrainingSet[user])
                - movieUserSupportAverages.at(item));
    }

    if (bound)
    {
        if (pred < MIN_RATING)
            pred = (float) MIN_RATING;
        else if (pred > MAX_RATING)
            pred = (float) MAX_RATING;
    }

    return pred;
}

Globals::~Globals()
{
    // No dynamically allocated resources to free at the moment.
}
