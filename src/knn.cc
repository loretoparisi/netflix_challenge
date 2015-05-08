#include <knn.hh>

// Comparison operator for s_neighors
int operator<(const s_neighbors &a, const s_neighbors &b)
{
    return a.weight > b.weight;
}


KNN::KNN(const int numUsers, const int numItems, const int minCommon,
         const unsigned int maxWeight, bool loadPFromFile, 
         bool savePToFile, const std::string &pFilename) :
    numUsers(numUsers), numItems(numItems), minCommon(minCommon),
    maxWeight(maxWeight), loadPFromFile(loadPFromFile), 
    savePToFile(savePToFile), pFilename(pFilename)
{

    // We should only save P if we're not already loading it.
    if (savePToFile && loadPFromFile)
    {
        throw std::logic_error("Should not be saving P to file "
                "if we're going to be loading it from file.");
    }

    um.resize(numUsers);
    mu.resize(numItems);
    P.resize(numItems);
    for (unsigned int j = 0; j < P.size(); j++)
    {
        P[j].resize(numItems);
    }
    movieAvg.resize(numItems);
    
    // Make sure specified file paths make sense.
    std::ofstream pfile(pFilename, ios::app);
    
    if (pfile.fail())
    {
        throw std::invalid_argument(pFilename + " cannot be opened");
    }
    
    pfile.close();
}


void KNN::train(const fmat &data)
{
    unsigned int i;
    int last_seen = 0, curr_count = -1;
    int user, item;
    float rating;
    for(i = 0; i < data.n_cols; i++)
    {
        user = roundToInt(data(USER_ROW, i));
        item = roundToInt(data(MOVIE_ROW, i));
        rating = data(RATING_ROW, i);

        if (last_seen == user)
        {
            curr_count++;
        }
        else
        {
            curr_count = 0;
            last_seen = user;
        }
        um_pair u_pair = um_pair();
        u_pair.movie = item;
        u_pair.rating = rating;
        um[user].push_back(u_pair);

        mu_pair m_pair = mu_pair();
        m_pair.user = user;
        m_pair.rating = rating;
        mu[item].push_back(m_pair);
    }

#ifndef NDEBUG
    cout << "Finished populating UM and MU data for kNN." << endl;
#endif
    
    // Load P or calculate P, depending on what was specified.
    if (loadPFromFile)
    {
        loadP();
    }
    else
    {
        calcP();
        
        if (savePToFile)
        {
            saveP();
        }
    }

#ifndef NDEBUG
    cout << "Finished training kNN predictior." << endl;
#endif
}


void KNN::calcP()
{
    int i, u, m, user, z;
    short movie;
    float x, y, xy, xx, yy, denom;
    unsigned int n;
    char rating_i, rating_j;
    // Vector size
    int size1, size2;
    // Intermediates for every movie pair
    s_inter tmp[numItems];
    float tmp_f;

#ifndef NDEBUG
    cout << "Calculating P..." << endl;
#endif
    
    // Compute intermediates
    for (i = 0; i < numItems; i++)
    {
        // Zero out intermediates
        for (z = 0; z < numItems; z++)
        {
            tmp[z].x = 0;
            tmp[z].y = 0;
            tmp[z].xy = 0;
            tmp[z].xx = 0;
            tmp[z].yy = 0;
            tmp[z].n = 0;
        }

        size1 = mu[i].size();

#ifndef NDEBUG
        if ((i % 1000) == 0)
            cout << "Finished handling movie " << i << "." << endl;
#endif

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
        // https://en.wikipedia.org/wiki/
        // Pearson_product-moment_correlation_coefficient
        for (z = 0; z < numItems; z++)
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
                denom = (float)std::sqrt(n * xx - x * x) * (float)std::sqrt(n * yy - y * y);
                // Check for NaN
                if (std::abs(denom) < EPSILON)
                {
                    tmp_f = 0.0;
                }
                else
                {
                    tmp_f = (float)(n * xy - x * y) / denom;
                }
                //cout << tmp_f << endl;
                //cout << tmp_f << " bool: " << (tmp_f != tmp_f) <<  endl;
                P[i][z].p = tmp_f;
                P[i][z].common = n;
            }
        }
    }
#ifndef NDEBUG
    cout << "P calculated." << endl;
#endif
}


void KNN::saveP()
{
    int i, j;
    
    std::ofstream pfile(pFilename, ios::app);
    for (i = 0; i < numItems; i++)
    {
        for (j = i; j < numItems; j++)
        {
            if (P[i][j].common != 0)
            {
                pfile << i << " " << j << " " << P[i][j].p << " " << P[i][j].common << endl;
            }
        }
    }
    pfile.close();
    
#ifndef NDEBUG
    cout << "P saved to " << pFilename << "." << endl;
#endif
}


void KNN::loadP()
{
    int i, j, common;
    float p;
    char c_line[100];
    std::string line;

    std::ifstream pfile(pFilename, ios::app);
    if (!pfile.is_open())
    {
        cerr << "Cannot open p val file at " << pFilename << "." << endl;
        exit(-1);
    }
    while (getline(pfile, line))
    {
        memcpy(c_line, line.c_str(), 100);
        i = atoi(strtok(c_line, " "));
        j = atoi(strtok(NULL, " "));
        p = (float) atof(strtok(NULL, " "));
        common = atof(strtok(NULL, " "));
        if (isinf(p))
        {
            P[i][j].p = 0;
        }
        else
        {
            P[i][j].p = p;
        }
        P[i][j].common = common;
    }
    pfile.close();

#ifndef NDEBUG
    cout << "P loaded from " << pFilename << "." << endl;
#endif
}


float KNN::predict(int user, int item, int date, bool bound)
{
    // NOTE: making item and n unsigned ints might make it easier for
    // the compiler to implement branchless min().
    float prediction = 0, denom = 0, diff, result;
    int n;
    s_pear tmp;
    s_neighbors neighbors[numItems];
    std::priority_queue<s_neighbors> q;
    s_neighbors tmp_pair;
    float p_lower, pearson;
    int common_users;

    // Len neighbors
    unsigned int i, size, j = 0;
    int n_item_1 = 0, n_item_2 = 0;

    // For each item rated by user
    size = um[user].size();
    
    for (i = 0; i < size; i++)
    {
        n = um[user][i].movie; // n: item watched by user

        n_item_1 = (item < n) ? item : n;
        n_item_2 = (item > n) ? item : n;
        tmp = P[n_item_1][n_item_2];
        common_users = tmp.common;

        // If item and m2 have >= minCommon viewers
        if (common_users >= minCommon)
        {
            neighbors[j].common = common_users;
            neighbors[j].m_avg = movieAvg[item];
            neighbors[j].n_avg = movieAvg[n];

            neighbors[j].n_rating = um[user][i].rating;

            pearson = tmp.p;
            neighbors[j].pearson = pearson;

            // Fisher and inverse-fisher transform (from wikipedia)
            p_lower = tanh(atanh(pearson) - 1.96 / sqrt(common_users - 3));
            //p_lower = pearson;
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
    neighbors[j].weight = log(minCommon);
    j++;

    // At this point we have an array of neighbors, length j.
    // Let's find the maxWeight elements of the array.

    // For each item-pair in neighbors
    for (i = 0; i < j; i++)
    {
        // If there is place in queue, just push it
        if (q.size() < maxWeight)
        {
            q.push(neighbors[i]);
        }

        // Else, push it only if this pair has a higher weight than the top
        // (smallest in top-maxWeight).
        // Remove the current top first
        else
        {
            if (q.top().weight < neighbors[i].weight)
            {
                q.pop();
                q.push(neighbors[i]);
            }
        }
    }

    // Now we can go ahead and calculate rating
    size = q.size();
    for (i = 0; i < size; i++)
    {
        tmp_pair = q.top();
        q.pop();
        diff = tmp_pair.n_rating - tmp_pair.n_avg;
        if (tmp_pair.pearson < 0)
        {
            diff = -diff;
        }
        prediction += tmp_pair.pearson * (tmp_pair.m_avg + diff);
        denom += tmp_pair.pearson;
    }

    // If result is nan, return avg
    if (std::abs(denom) < EPSILON)
    {
        result = MEAN_RATING_TRAINING_SET;
    }
    else
    {
        result = ((float) prediction) / denom;
    }
    if (bound)
    {
        if (result < MIN_RATING) {
            result = MIN_RATING;
        }
        else if (result > MAX_RATING) {
            result = MAX_RATING;
        }
    }
    
    return result;
}


KNN::~KNN()
{
    // No dynamically allocated resources to free at the moment.
}

