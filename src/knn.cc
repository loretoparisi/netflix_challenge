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
KNN::KNN(const string &trainFilenameUM, const string &trainFilenameMU,
    const string &pFilename, const string &qualFilename,
    const string &outputFilename, bool test) :
    trainFilenameUM(trainFilenameUM), trainFilenameMU(trainFilenameMU),
    pFilename(pFilename), qualFilename(qualFilename),
    outputFilename(outputFilename)
{}

// Comparison operator for s_neighors
int operator<(const s_neighbors &a, const s_neighbors &b) {
    return a.weight > b.weight;
}

void KNN::loadData() {
    string line;
    char c_line[20];
    int userId;
    int movieId;
    int time;
    int rating;

    int i = -1;
    int last_seen = 0;

    // Used for movie avgs
    int num_ratings = 0;
    int avg = 0;

    ifstream trainingDta (trainFilenameUM); 
    if (trainingDta.fail()) {
        cout << "train um file open failed.\n";
        exit(-1);
    }
    while (getline(trainingDta, line)) {
        memcpy(c_line, line.c_str(), 20);
        userId = atoi(strtok(c_line, " "));
        movieId = (short) atoi(strtok(NULL, " "));
        time = atoi(strtok(NULL, " ")); 
        rating = (char) atoi(strtok(NULL, " "));
        
        if (last_seen == userId) {
            i++;
        }
        else {
            i = 0;
            last_seen = userId;
        }

        um[userId].push_back(um_pair());
        um[userId][i].movie = movieId;
        um[userId][i].rating = rating;
    }
    trainingDta.close();

    cout << "Loaded train um file." << endl;

    i = -1;
    last_seen = 0;

    // Repeat again, now for mu dta
    ifstream trainingDtaMu (trainFilenameMU); 
    if (trainingDtaMu.fail()) {
        cout << "train mu file open failed.\n";
        exit(-1);
    }
    while (getline(trainingDtaMu, line)) {
        memcpy(c_line, line.c_str(), 20);
        userId = atoi(strtok(c_line, " "));
        movieId = (short) atoi(strtok(NULL, " "));
        time = atoi(strtok(NULL, " ")); 
        rating = (char) atoi(strtok(NULL, " "));

        // If we're still on the same movie
        if (last_seen == movieId) {
            i++;
            num_ratings += 1;
            avg += rating;
        }
        else {
            i = 0;
            last_seen = movieId;
            movieAvg[movieId] = float(avg) / num_ratings;
            num_ratings = 1;
            avg = rating;
        }
        
        mu[movieId].push_back(mu_pair());
        mu[movieId][i].user = userId;
        mu[movieId][i].rating = rating;
    }
    trainingDtaMu.close();
    cout << "Loaded train mu file." << endl;

}

void KNN::calcP() {
    int i, u, m, user, z;
    short movie;
    float x, y, xy, xx, yy;
    unsigned int n;

    char rating_i, rating_j;

    // Vector size
    int size1, size2;

    // Intermediates for every movie pair
    s_inter tmp[NUM_MOVIES];

    cout << "Calculating P..." << endl;

    float tmp_f;

    // Compute intermediates
    for (i = 0; i < NUM_MOVIES; i++) {
        // Zero out intermediates
        for (z = 0; z < NUM_MOVIES; z++) {
            tmp[z].x = 0;
            tmp[z].y = 0;
            tmp[z].xy = 0;
            tmp[z].xx = 0;
            tmp[z].yy = 0;
            tmp[z].n = 0;
        }

        size1 = mu[i].size();

        if ((i % 100) == 0) {
            cout << i << endl;
        }

        // For each user that rated movie i
        for (u = 0; u < size1; u++)
        {
            user = mu[i][u].user;
            size2 = um[user].size();
            // For each movie j rated by current user
            for (m = 0; m < size2; m++)
            {
                movie = um[user][m].movie; // id of movie j

                // At this point, we know that user rated both movie i
                // AND movie. Thus we can update the pearson coeff for
                // the pair XY

                // Rating of movie i
                rating_i = mu[i][u].rating;

                // Rating of movie j
                rating_j = um[user][m].rating;

                // Increment rating of movie i
                tmp[movie].x += rating_i;

                // Increment rating of movie j
                tmp[movie].y += rating_j;

                tmp[movie].xy += rating_i * rating_j;
                tmp[movie].xx += rating_i * rating_i;
                tmp[movie].yy += rating_j * rating_j;

                // Increment number of viewers of movies i AND j
                tmp[movie].n += 1;
            }
        }

        // Calculate Pearson coeff. based on: 
        // https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
        for (z = 0; z < NUM_MOVIES; z++)
        {
            x = tmp[z].x;
            y = tmp[z].y;
            xy = tmp[z].xy;
            xx = tmp[z].xx;
            yy = tmp[z].yy;
            n = tmp[z].n;
            if (n == 0)
            {
                P[i][z].p = 0;
            }
            else
            {
                tmp_f = (n * xy - x * y) / (sqrt(n * xx - x*x) * sqrt(n * yy - y*y));
                // Test for NaN
                if (tmp_f != tmp_f)
                {
                    tmp_f = 0.0;
                }
                P[i][z].p = tmp_f;
                P[i][z].common = n;
            }
        }

    }
    cout << "P calculated." << endl;
}

/* Generate out of sample RMSE for the current number of features, then
write this to a rmseOut. */
void KNN::outputRMSE(short numFeats) {
    string line;
    char c_line[20];
    int userId, movieId, time;
    double predicted, actual; // ratings
    double err, sq, rmse;
    ifstream probe("../../netflix_challenge/data/probe.dta");
    sq = 0;
    while (getline(probe, line)) {
        memcpy(c_line, line.c_str(), 20);
        userId = atoi(strtok(c_line, " "));
        movieId = atoi(strtok(NULL, " "));
        time = atoi(strtok(NULL, " "));
        actual = (double) atoi(strtok(NULL, " "));
        predicted = predict(userId, movieId, -1);
        err = actual - predicted;
        sq += err * err;
    }
    rmse = sqrt(sq/1374739);
    cout << "RMSE is: " << rmse << endl;
}


void KNN::saveP() {
    int i, j;

    cout << "Saving P..." << endl;

    ofstream pfile(pFilename, ios::app);
    if (!pfile.is_open())
    {
        cout << "Cannot save p values.\n";
        exit(-1);
    }
    
    for (i = 0; i < NUM_MOVIES; i++)
    {
        for (j = i; j < NUM_MOVIES; j++)
        {
            if (P[i][j].common != 0)
            {
                pfile << i << " " << j << " " << P[i][j].p << " " << P[i][j].common << endl;
            }
        }
    }
    pfile.close();
    cout << "P saved." << endl;
}

void KNN::loadP() {
    int i, j, common;
    float p;
    char c_line[100];
    string line;

    cout << "Loading P..." << endl;

    ifstream pfile(pFilename, ios::app);
    if (!pfile.is_open()) {
        cout << "Cannot open p file.\n";
        exit(-1);
    }
    
    while (getline(pfile, line)) {
        memcpy(c_line, line.c_str(), 100);
        i = atoi(strtok(c_line, " "));
        j = atoi(strtok(NULL, " "));
        p = (float) atof(strtok(NULL, " "));
        common = atof(strtok(NULL, " "));
        if (isinf(p)) {
            P[i][j].p = 0;
        }
        else {
            P[i][j].p = p;
        }
        P[i][j].common = common;
    }

    pfile.close();
    cout << "P loaded." << endl;

}

float KNN::predict(int user, int item, int date) {
    // NOTE: making item and n unsigned ints might make it easier for the compiler
    // to implement branchless min()
    float prediction = 0;
    float denom = 0;
    float diff;
    float result;

    int n;

    s_pear tmp;

    s_neighbors neighbors[NUM_MOVIES];

    priority_queue<s_neighbors> q;
    
    s_neighbors tmp_pair;

    float p_lower, pearson;

    int common_users;

    // Len neighbors
    unsigned int i, size, j = 0;
    
    int n_item_1 = 0, n_item_2 = 0;

    // For each item rated by user
    size = um[user].size();
    
    for (i = 0; i < size; i++) {
        n = um[user][i].movie; // n: item watched by user

        n_item_1 = (item < n) ? item : n;
        n_item_2 = (item > n) ? item : n;
        tmp = P[n_item_1][n_item_2];
        common_users = tmp.common;

        // If item and m2 have >= MIN_COMMON viewers
        if (common_users >= MIN_COMMON) {
            neighbors[j].common = common_users;
            neighbors[j].m_avg = movieAvg[item];
            neighbors[j].n_avg = movieAvg[n];

            neighbors[j].n_rating = um[user][i].rating;

            pearson = tmp.p;
            neighbors[j].pearson = pearson;

            // Fisher and inverse-fisher transform (from wikipedia)
            p_lower = tanh(atanh(pearson) - 1.96 / sqrt(common_users - 3));
//             p_lower = pearson;
            neighbors[j].p_lower = p_lower;
            neighbors[j].weight = p_lower * p_lower * log(common_users);
            j++;
        }

    }

    // Add the dummy element described in the blog
    neighbors[j].common = 0;
    neighbors[j].m_avg = movieAvg[item];
    neighbors[j].n_avg = 0;

    neighbors[j].n_rating = 0;

    neighbors[j].pearson = 0;

    neighbors[j].p_lower = 0;
    neighbors[j].weight = log(MIN_COMMON);
    j++;



    // At this point we have an array of neighbors, length j. Let's find the
    // MAX_W elements of the array using 

    // For each item-pair in neighbors
    for (i = 0; i < j; i++) {
        // If there is place in queue, just push it
        if (q.size() < MAX_W) {
            q.push(neighbors[i]);
        }

        // Else, push it only if this pair has a higher weight than the top
        // (smallest in top-MAX_W).
        // Remove the current top first
        else {
            if (q.top().weight < neighbors[i].weight) {
                q.pop();
                q.push(neighbors[i]);
            }
        }
    }

    // Now we can go ahead and calculate rating
    size = q.size();
    for (i = 0; i < size; i++) {
        tmp_pair = q.top();
        q.pop();
        diff = tmp_pair.n_rating - tmp_pair.n_avg;
        if (tmp_pair.pearson < 0) {
            diff = -diff;
        }
        prediction += tmp_pair.pearson * (tmp_pair.m_avg + diff);
        denom += tmp_pair.pearson;

    }

    result = ((float) prediction) / denom;

    // If result is nan, return avg
    if (result != result) {
        return MEAN_RATING_TRAINING_SET;
    }
    else if (result < 1) {
        return 1;
    }
    else if (result > 5) {
        return 5;
    }

    return result;

}


// TODO: refactor.
void KNN::output() {
    string line;
    char c_line[20];
    int userId;
    int movieId;
    float rating;
    stringstream fname;

    cout << "Generating output" << endl;

    fname << outputFilename;

    ifstream qual (qualFilename);
    ofstream out (fname.str().c_str(), ios::trunc); 
    if (qual.fail() || out.fail()) {
        cout << "qual file cannot be opened.\n";
        exit(-1);
    }
    while (getline(qual, line)) {
        memcpy(c_line, line.c_str(), 20);
        userId = atoi(strtok(c_line, " "));
        movieId = (short) atoi(strtok(NULL, " "));

        // TODO (from Laksh): I know this doesn't use times, but it's
        // probably a good idea to pass in "time" anyways, and not just -1.
        rating = predict(userId, movieId, -1);
        out << rating << '\n';
    }

    cout << "Output generated" << endl;
    outputRMSE(200);
}


/**
 * TODO: Move training here! Also describe the format that you expect
 * "data" to be. It should be in column-major format with a shape of 4 x
 * NUM_RATINGS.
 */

void KNN::train(const fmat &data)
{
    
}


/**
 * This function also trains, but it first loads a file from fileNameData.
 * This file must be an Armadillo binary of an fmat.
 *
 * @param fileNameData: The file where "data" is stored. This binary file
 *                      must hold matrix data in the format specified in
 *                      the train(const fmat &data) function.
 *
 */
void KNN::train(const string &fileNameData)
{
    fmat data;

    data.load(fileNameData, arma_binary);
    train(data);
}



KNN::~KNN()
{
    // No dynamically allocated resources to free at the moment.
}
