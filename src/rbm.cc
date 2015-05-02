#include <algorithm>
#include <cstdint>
#include <ctime>
#include <dirent.h>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>

#ifndef NDEBUG
#include <chrono>
#include <iostream>
#endif

#include <rbm.hh>

#define CACHE_EXT ".bin"
#define DATA_DIR "data/rbm_cached/"
#define INDICATOR_DATA_DIR DATA_DIR "users/"
#define PMF_PATH DATA_DIR "rating_pmf" CACHE_EXT

// Get an element in a flat "2d" array
#define GET2D(array, d1, i, j) *(array + d1 * i + j)
// Get an element in a flat "3d" array
#define GET3D(array, d1, d2, i, j, k) *(array + (k * d2 + i) * d1 + j)
// Get a pointer to an element in a flat "2d" array
#define GET2DPTR(array, d1, i, j) array + d1 * i + j
// Get a pointer to an element in a flat "3d" array
#define GET3DPTR(array, d1, d2, i, j, k) array + (k * d2 + i) * d1 + j

#define PAIR_WIDTH 3

#define EPOCHS 10
#define CD_STEPS 1
#define DECAY 0.0001

RBM::RBM (int users, int movies, int hidden, float rate, float momentum) : 
    users(users), movies(movies), hidden(hidden), rate(rate), 
    momentum(momentum) {
    // Allocate weight matrix
    this->weights = new data_t[movies * MAX_RATING * hidden];
    // Allocate visible biases
    this->visibleBias = new data_t[movies * MAX_RATING]();
    // Allocate hidden biases
    this->hiddenBias = new data_t[hidden]();
}

RBM::~RBM () {
    delete [] this->weights;
    delete [] this->visibleBias;
    delete [] this->hiddenBias;
}

// Helper function for building the path to a user's cached indicator matrix
inline static std::string userCachePath(const int &user) {
    // Stream buffer for building the path of this user's cache 
    std::ostringstream cachePath;
    // Cached matrices are stored with the user's id as a filename
    cachePath << INDICATOR_DATA_DIR << user << CACHE_EXT;

    return cachePath.str();
}

// Compute an indicator matrix, cache it to disk & store it in memory
inline static void cacheIndicator (uint8_t **indicators, uint16_t *counts,
                                   const Mat<data_t> &data, int &col, 
                                   const int &user) {
    // Vector for storing (movie, rating) pairs (indices of sparse indicator
    // matrix)
    std::vector<std::pair<uint16_t, uint8_t>> ratings;
    // Store the starting column index
    int count = -col;
    // While the current column of data is for this user
    while ( data.at(USER_ROW, col) == user ) {
        // Record the rating for this movie
        ratings.push_back(std::make_pair(data.at(MOVIE_ROW, col),
                                         data.at(RATING_ROW, col)));
        // Check the next column
        ++col;
    }
    // Number of ratings is equal to the change in column index
    count += col;
    // Store the number of ratings for this user.  Assumes that there are
    // < 2^16 movies in the data set
    counts[user] = (uint16_t) count;
    // Allocate an array for storing this user's (movie, rating) pairs
    char *indicator = new char[PAIR_WIDTH * ratings.size()];
    // Loop pointer
    char *ind_ptr = indicator;
    // Load the (movie, rating) pairs into the array from the ratings vector
    for ( uint16_t i = 0; i < ratings.size(); ++i ) {
        *((uint16_t *) ind_ptr) = std::get<0>(ratings[i]);
        ind_ptr += sizeof(uint16_t);
        *ind_ptr++ = std::get<1>(ratings[i]);
    }
    // Get the destination path of the cache
    std::string cachePath = userCachePath(user);
    // Cache the matrix
    std::ofstream indicatorCache(cachePath, ios::binary);
    indicatorCache.write(indicator, PAIR_WIDTH * ratings.size());
    indicatorCache.close();
    // Put the matrix in the array of indicator matrices
    indicators[user] = (uint8_t *) indicator;
}

// Load a cached indicator matrix
static void loadIndicator (uint8_t **indicators, uint16_t *counts, 
                                  const int &user) {
    // Get the source path of the cached indicator matrix for this user
    std::string cachePath = userCachePath(user);
    // Open this user's cached (sparse) indicator matrix
    std::ifstream indicatorCache(cachePath, ios::binary | ios::ate);
    // Calculate the number of (movie, rating) pairs for this user
    uint16_t nratings = indicatorCache.tellg() / PAIR_WIDTH;
    // Store the number of ratings for this user
    counts[user] = nratings;
    // Allocate an array to store all of the (movie, rating) pairs
    uint8_t *indicator = new uint8_t[PAIR_WIDTH * nratings];
    // Reset stream to the beginning of the file
    indicatorCache.seekg(0, ios::beg);
    // Load this user's cached (sparse) indicator matrix
    indicatorCache.read((char *) indicator, PAIR_WIDTH * nratings);
    indicatorCache.close();
    // Put the matrix in the array of indicator matrices
    indicators[user] = (uint8_t *) indicator;
}

// Loop helper for geting the next (movie, rating) pair from an array of the
// set indices of a sparse indicator matrix
inline static void getPair (uint8_t *indicator, uint16_t &movie, 
                            uint8_t &rating) {
    // Extract the movie
    movie = *((uint16_t *) indicator);
    // Increment the pointer past the movie entry
    indicator += sizeof(uint16_t);
    // Extract the rating, and increment the pointer past the rating entry
    rating = *indicator++;
}
// Loop helper for getting the next movie, ignoring the rating
inline static void getMovie (uint8_t *indicator, uint16_t &movie) {
    // Extract the movie
    movie = *((uint16_t *) indicator);
    // Increment the pointer past this (movie, rating) pair
    indicator += PAIR_WIDTH;
}

void RBM::train (const Mat<data_t> &data) {
    // Allocate an array for storing each user's sparse indicator matrix
    // This array is not contiguous in memory on purpose; we don't want
    // all users' (sparse) indicator matrices taking up cache space when we
    // only need one at a time, and it makes indexing way easier
    uint8_t **indicators = new uint8_t*[this->users];
    // User id-indexed array containing the number of ratings per user
    uint16_t *ratingCount = new uint16_t[this->users];

    // Initialize weights & biases

    // Mean & standard deviation of our normal distribution
    const data_t mean = 0.0;
    const data_t stddev = 0.01;
    std::mt19937 engine(time(NULL));
    std::normal_distribution<data_t> normal(mean, stddev);
#ifndef NDEBUG
    std::cout << "Initializing weight matrix" << std::endl;
#endif
    // Initialize weight matrix using our normal distribution
    for ( int i = 0; i < this->movies * MAX_RATING * this->hidden; ++i ) {
        this->weights[i] = normal(engine);
    }
#ifdef RANDOM
  #ifndef NDEBUG
    std::cout << "Initializing biases of hidden units" << std::endl;
  #endif
    // Randomly initialize the biases of the hidden units
    for ( int i = 0; i < this->hidden; ++i ) {
        this->hiddenBias[i] = normal(engine);
    }
#endif
#ifndef NDEBUG
    std::cout << "Initializing biases of the visible units" << std::endl;
#endif
    struct stat statBuffer;
    // If a cached ranking pmf matrix does not exist
    if ( stat(PMF_PATH, &statBuffer) != 0 ) {
#ifndef NDEBUG
        std::cout << "Caching rating pmf's for all movies" 
                  << std::endl;
#endif
        int movie, rating;
        // For each column in the data matrix (rating entry)
        for ( unsigned i = 0; i < data.n_cols; ++i ) {
            movie = data.at(MOVIE_ROW, i);
            rating = data.at(RATING_ROW, i) - 1;
            // Increment the count for that movie, rating pair
            GET2D(this->visibleBias, MAX_RATING, movie, rating) += 1;
        }
        data_t total;
        for ( int i = 0; i < movies * MAX_RATING; ) {
            // Compute the total number of ratings for this movie
            total = *(this->visibleBias + i);
            total += *(this->visibleBias + i + 1);
            total += *(this->visibleBias + i + 2);
            total += *(this->visibleBias + i + 3);
            total += *(this->visibleBias + i + 4);
            // Normalize the individual counts to produce probabilities
            *(this->visibleBias + i++) /= total;
            *(this->visibleBias + i++) /= total;
            *(this->visibleBias + i++) /= total;
            *(this->visibleBias + i++) /= total;
            *(this->visibleBias + i++) /= total;
        }
        // Open the binary file used to cache the rating pmf's
        std::ofstream biasCache(PMF_PATH, ios::binary);
        // Cache the pmf matrix (initial biases of the visible units)
        biasCache.write((char *) this->visibleBias, 
                        this->movies * MAX_RATING * sizeof(data_t));
    } else {
#ifndef NDEBUG
        std::cout << "Loading cached rating pmf's" << std::endl;
#endif
        // Open the binary file used to cache the rating pmf's
        std::ifstream biasCache(PMF_PATH, ios::binary);
        // Initialize biases of visible units
        biasCache.read((char *) this->visibleBias,
                       this->movies * MAX_RATING * sizeof(data_t));
    }
    // Vector for marking indicator matrices as found; all are initialized as
    // missing
    std::vector<bool> found(this->users, 0);
    // Try opening the indicator data directory
    DIR *directory = opendir(INDICATOR_DATA_DIR);
    struct dirent *entity;
    // If the indicator data directory exists
    if ( directory != NULL ) {
        // For every entry in the directory
        while ( entity = readdir(directory) ) {
            // Ignore hidden files
            if ( entity->d_name[0] == '.' ) continue;
            // Convert the filename into a string
            std::string filename = std::string(entity->d_name);
            // Trim the file's extension (see CACHE_EXT macro)
            filename.resize(filename.size() - 4);
            // Mark this indicator matrix as cached
            found[std::stoi(filename)] = 1;
        }
        // Close directory
        closedir(directory);
    // If it doesn't exist
    } else {
        // Try making the cache directory
        if ( mkdir(INDICATOR_DATA_DIR, 
                   S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH) != 0 ) {
            // Report that making the directory failed
            throw std::runtime_error("Unable to make directory " 
                                     INDICATOR_DATA_DIR);
        }
    }
    // The number of cached indicator matrices present in the filesystem
    int present = std::accumulate(found.cbegin(), found.cend(), 0);
#ifndef NDEBUG
    std::cout << present << " indicator matrices found" << std::endl;
#endif
    // If no matrices are found
    if ( present == 0 ) {
#ifndef NDEBUG
        std::cout << "Caching indicator matrices for all users" << std::endl;
#endif
        // Current column index in the data matrix
        int col = 0;
        // For each user (assuming the data is sorted by user ID)
        for ( int user = 0; user < this->users; ++user ) {
            // Cache their indicator matrix; col is passed by reference, and
            // incremented in cacheIndicator, so upon termination of
            // cacheIndicator, col contains the first column for the next user
            cacheIndicator(indicators, ratingCount, data, col, user);
        }
    // If only some matrices were not found
    } else if ( present < this->users ) {
#ifndef NDEBUG
        std::cout << "Caching missing indicator matrices" << std::endl;
#endif
        // Let uid be the user ID in the column of data specified by index.
        // probe returns -1 if the key is less than uid, 1 if it's greater,
        // and 0 otherwise (key == uid)
        const std::function<int(Mat<data_t>, int, int)> probe = 
        [&] (Mat<data_t> data, int key, int index) {
            return ( key < data.at(USER_ROW, index) ) ? -1 : 
                   (( key > data.at(USER_ROW, index) ) ? 1 : 0 );
        };
        int col;
        // For each user
        for ( int user = 0; user < this->users; ++user ) {
            // If their indicator matrix is cached
            if ( found[user] ) {
                // Load it into memory
                loadIndicator(indicators, ratingCount, user);
                continue;
            }
            // Otherwise, serch for a rating by this user in the data
            col = binary_search<Mat<data_t>, int>(data, user, probe, 0, 
                                                  data.n_cols - 1);
            // Rewind past their first rating
            while ( col > 0 && data.at(USER_ROW, --col) == user ) { }
            // Move to the user's first rating if necessary
            col += col > 0;
            // Cache the indicator matrix for this user
            cacheIndicator(indicators, ratingCount, data, col, user);
        }
    // More cached matrices were found than there are users
    } else if ( present > this->users ) {
        std::ostringstream msg;
        msg << present << " indicator matrices found for " << this->users
            << " users";
        throw std::runtime_error(msg.str());
    } else {
#ifndef NDEBUG
        std::cout << "Loading indicator matrices from caches" << std::endl;
#endif
        // Load all cached indicator matrices into memory
        for ( int user = 0; user < this->users; ++user ) {
            loadIndicator(indicators, ratingCount, user);

            // for ( uint8_t *ind = indicators[user]; 
            //       ind < ind + ratingCount[user]; ) {
            //     getPair(ind, movie, rating)
            // }
        }
    }

    // Begin training procedure

    // Activation probabilities for the hidden units
    data_t *hiddenProbs = new data_t[this->hidden]();
    // Store the state of the hidden units across contrastive divergence steps
    uint8_t *hiddenStatesBuffer = new uint8_t[this->hidden]();
    // Hidden states sampled from the data
    uint8_t *posHiddenStates = new uint8_t[this->hidden]();
    // Activation histogram for hidden units sampled from the data
    uint8_t *posHiddenAct = new uint8_t[this->hidden]();
    // Hidden states sampled using contrastive divergence (approximation of
    // sampling from the model's distribution, which is intractable)
    uint8_t *negHiddenStates = new uint8_t[this->hidden]();
    // Activation histogram using contrastive divergence
    uint8_t *negHiddenAct = new uint8_t[this->hidden]();

    // Activation probabilities for the visible units
    data_t *visibleProbs = new data_t[this->movies * MAX_RATING]();
    // Non-regularized activation probabilities for the visible units, used to
    // compute RMSE
    data_t *visibleProbsRMSE = new data_t[this->movies * MAX_RATING]();
    // Activation histogram for visible units sampled from the data
    uint8_t *posVisibleAct = new uint8_t[this->movies * MAX_RATING]();
    // Activation histogram for visible units sampled using contrastive
    // divergence
    uint8_t *negVisibleAct = new uint8_t[this->movies * MAX_RATING]();
    // Movie-id indexed array used as a buffer for storing the sampled states
    // of the visible softmax units
    uint8_t *softmax = new uint8_t[this->movies]();

    // Change in the weight matrix
    data_t *deltaCD = new data_t[this->movies * MAX_RATING * this->hidden]();
    // Change in the visible unit biases
    data_t *deltaVisibleBias = new data_t[this->movies * MAX_RATING]();
    // Change in the hidden unit biases
    data_t *deltaHiddenBias = new data_t[this->hidden]();
    // Contrastive divergence 
    uint8_t *posCD = new uint8_t[this->movies * MAX_RATING * this->hidden]();
    uint8_t *negCD = new uint8_t[this->movies * MAX_RATING * this->hidden]();

#ifndef NDEBUG
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double, std::ratio<60>> minutes_elapsed;
#endif

    // Initialize uniform distribution once
    std::uniform_real_distribution<data_t> uniform(0.0, 1.0);
    bool active;

    // For the specified number of epochs (should be while RMSE is decreasing)
    for ( int epoch = 0; epoch < EPOCHS; ++epoch ) {
#ifndef NDEBUG
        start = std::chrono::system_clock::now();
#endif
        // For each user
        for ( int user = 0; user < this->users; ++user ) {
            // This user's sparse indicator matrix
            uint8_t *indicator = indicators[user];
            // A past-the-end pointer for this user's indicator matrix
            const uint8_t *indicatorEnd = 
                indicator + ratingCount[user] * PAIR_WIDTH;

            // Run a training iteration until we have to sample the visible
            // units, i.e., sample the hidden units given the data

            uint16_t movie;
            uint8_t rating;
            // For all (movie, rating) pairs (active visible units)
            for ( uint8_t *ind = indicator; ind < indicatorEnd; ) {
                // Extract the next movie, rating pair (indices in a sparse
                // indicator matrix)
                getPair(ind, movie, rating);
                // Add bias contribution of this set visible unit (sample its
                // state from the data)
                GET2D(posVisibleAct, MAX_RATING, movie, rating) += 1;
            }
            // For each hidden unit
            for ( int h = 0; h < this->hidden; ++h ) {
                // Calculate the sum of the elements of the Schur product of
                // this user's indicator matrix & the hth slice of the weight
                // cube
                for ( uint8_t *ind = indicator; ind < indicatorEnd; ) {
                    // Extract the next movie, rating pair (indices in a sparse
                    // indicator matrix)
                    getPair(ind, movie, rating);
                    // Accumulate the contribution of this set visible unit
                    hiddenProbs[h] += GET3D(this->weights, MAX_RATING,
                                            this->hidden, movie, rating, h);
                }
            }
            // For each hidden unit
            for ( int h = 0; h < this->hidden; ++h ) {
                // Calculate P[h_j = 1 | V] (Eq. 9 of Salakhutdinov, Mnih,
                // & Hinton 2007)
                hiddenProbs[h] = sigmoid<data_t>(
                    hiddenProbs[h] + this->hiddenBias[h]);
                // Sample the state of this hidden unit
                active = hiddenProbs[h] > uniform(engine);
                // Record the sampled state of this unit for training purposes
                // (sample from data)
                posHiddenStates[h] = hiddenStatesBuffer[h] = (uint8_t) active;
                posHiddenAct += (uint8_t) active;
            }
            // For each hidden unit
            for ( int h = 0; h < this->hidden; ++h ) {
                // For all (movie, rating) pairs
                for ( uint8_t *ind = indicator; ind < indicatorEnd; ) {
                    // Extract the next movie index
                    getMovie(ind, movie);
                    // Calculate non-regularized probabilities for RMSE
                    // reporting
                    data_t *rmse = 
                        GET2DPTR(visibleProbsRMSE, MAX_RATING, movie, 0);
                    data_t *weight = GET3DPTR(this->weights, MAX_RATING, 
                                              this->hidden, movie, 0, h);
                    // Calculate activation probability of "1" softmax unit
                    *rmse++ = hiddenProbs[h] * *weight++;
                    // Calculate activation probability of "2" softmax unit
                    *rmse++ = hiddenProbs[h] * *weight++;
                    // Calculate activation probability of "3" softmax unit
                    *rmse++ = hiddenProbs[h] * *weight++;
                    // Calculate activation probability of "4" softmax unit
                    *rmse++ = hiddenProbs[h] * *weight++;
                    // Calculate activation probability of "5" softmax unit
                    *rmse = hiddenProbs[h] * *weight;
                }
            }

            // Run the desired number of contrastive divergence iterations
            for ( int k = CD_STEPS; k > 0; --k ) {
                const bool lastStep = k <= 1;
                
                // Begin reconstruction

                // For each hidden unit
                for ( int h = 0; h < this->hidden; ++h ) {
                    // If it is not active, skip it
                    if ( ! hiddenStatesBuffer[h] ) continue;
                    // For all (movie, rating) pairs
                    for ( uint8_t *ind = indicator; ind < indicatorEnd; ) {
                        // Extract the next movie index
                        getMovie(ind, movie);
                        // Accumulate contributions from this set hidden
                        // unit
                        data_t *prob = GET2DPTR(visibleProbs, MAX_RATING, 
                                                movie, 0);
                        data_t *weight = GET3DPTR(this->weights, MAX_RATING,
                                                  this->hidden, movie, 0, h);
                        // Accumulate contribution to "1" softmax unit
                        *prob++ += *weights++;
                        // Accumulate contribution to "2" softmax unit
                        *prob++ += *weights++;
                        // Accumulate contribution to "3" softmax unit
                        *prob++ += *weights++;
                        // Accumulate contribution to "4" softmax unit
                        *prob++ += *weights++;
                        // Accumulate contribution to "5" softmax unit
                        *prob += *weights;
                    }
                }
                // For all (movie, rating) pairs
                for ( uint8_t *ind = indicator; ind < indicatorEnd; ) {
                    // Extract the next movie index
                    getMovie(ind, movie);
                    data_t total = 0;
                    data_t *prob = 
                        GET2DPTR(visibleProbs, MAX_RATING, movie, 0);
                    data_t *bias = 
                        GET2DPTR(this->visibleBias, MAX_RATING, movie, 0);
                    // Calculate P[v_q^k == 1 | h] (Eq. 10 of Salakhutdinov,
                    // Mnih, & Hinton 2007) & accumulate total probability
                    // (for normalization)
                    // For "1" softmax unit
                    *prob = sigmoid<data_t>(*prob + *bias++);
                    total += *prob++;
                    // For "2" softmax unit
                    *prob = sigmoid<data_t>(*prob + *bias++);
                    total += *prob++;
                    // For "3" softmax unit
                    *prob = sigmoid<data_t>(*prob + *bias++);
                    total += *prob++;
                    // For "4" softmax unit
                    *prob = sigmoid<data_t>(*prob + *bias++);
                    total += *prob++;
                    // For "5" softmax unit
                    *prob = sigmoid<data_t>(*prob + *bias++);
                    total += *prob;
                    // Normalize activation probabilities
                    // For "5" softmax unit
                    *prob-- /= total;
                    // For "4" softmax unit
                    *prob-- /= total;
                    // For "3" softmax unit
                    *prob-- /= total;
                    // For "2" softmax unit
                    *prob-- /= total;
                    // For "1" softmax unit
                    *prob /= total;

                    // Sample the state of this visible softmax unit (sample
                    // from approximation of model)
                    data_t *sampledRating = prob;
                    data_t r = uniform(engine) - *sampledRating;
                    while ( r > 0 ) {
                        r -= *(++sampledRating);
                    }
                    // Record the sampled rating
                    softmax[movie] = sampledRating - prob;

                }

                // If this is the last CD step
                if ( lastStep ) {
                    // For all (movie, rating) pairs
                    for ( uint8_t *ind = indicator; ind < indicatorEnd; ) {
                        // Extract the next movie index
                        getMovie(ind, movie);
                        // Train on the sampled data
                        GET2D(negVisibleAct, MAX_RATING, movie, softmax[movie])
                            += 1;
                    }
                }
                // Sample the states of the hidden units given the states of
                // visible units sampled from the approximation of the model

                std::fill(hiddenProbs, hiddenProbs + this->hidden, 0.0);
                // For each hidden unit
                for ( int h = 0; h < this->hidden; ++h ) {
                    // For all (movie, rating) pairs
                    for ( uint8_t *ind = indicator; ind < indicatorEnd; ) {
                        // Extract the next movie index
                        getMovie(ind, movie);
                        // Accumulate contribution of the sampled visible unit 
                        hiddenProbs[h] += 
                            GET3D(this->weights, MAX_RATING, this->hidden,
                                  movie, softmax[movie], h);
                    }
                    // Calculate P[h_j = 1 | V] (Eq. 9 of Salakhutdinov, Mnih,
                    // & Hinton 2007) from the sampled data
                    hiddenProbs[h] = sigmoid<data_t>(
                        hiddenProbs[h] + this->hiddenBias[h]);
                    // Sample the state of this hidden unit
                    active = hiddenProbs[h] > uniform(engine);
                    // Record the sampled state of this unit for training
                    // purposes (sample from approximation of model)
                    negHiddenStates[h] = (uint8_t) active;

                    // If this is the last CD step
                    if ( lastStep ) {
                        // Train on this data
                        negHiddenAct[h] += 1;
                    } else {
                        // Reset sampled visible unit activation probabilities
                        std::fill(visibleProbs,
                                  visibleProbs + this->movies * MAX_RATING, 0);
                        // Load the sampled states of the hidden unit into the
                        // buffer, for the next contrastive divergence step
                        for ( int h = 0; h < this->hidden; ++h ) {
                            hiddenStatesBuffer[h] = negHiddenStates[h];
                        }
                    }
                }
            } // CD-k iterations

            // Accumulate changes in the weights

            // For each hidden unit
            for ( int h = 0; h < this->hidden; ++h ) {
                // For all (movie, rating) pairs
                for ( uint8_t *ind = indicator; ind < indicatorEnd; ) {
                    // Extract the next movie index
                    getMovie(ind, movie);
                    // Accumulate contributions from Gibbs sampling
                    GET3D(negCD, MAX_RATING, this->hidden, 
                          movie, softmax[movie], h) += negHiddenStates[h];
                }
            }
            // For each hidden unit
            for ( int h = 0; h < this->hidden; ++h ) {
                // Skip units not activated when sampled given the data
                // distribution
                if ( ! posHiddenStates[h] ) continue;
                // For all (movie, rating) pairs
                for ( uint8_t *ind = indicator; ind < indicatorEnd; ) {
                    // Extract the next (movie, rating) pair
                    getPair(ind, movie, rating);
                    // Accumulate contributions the data distribution
                    GET3D(posCD, MAX_RATING, this->hidden, movie, rating, h)
                        += 1;
                }
            }

            // Update weights & biases

            uint8_t pos, neg;

            // For each hidden unit
            for ( int h = 0; h < this->hidden; ++h ) {

                // Update weights

                // For all (movie, rating) pairs
                for ( uint8_t *ind = indicator; ind < indicatorEnd; ) {
                    // Extract the next movie index
                    getMovie(ind, movie);
                    // For each individual softmax unit for this movie
                    for ( int r = 0; r < MAX_RATING; ++r ) {
                        pos = GET3D(posCD, MAX_RATING, this->hidden,
                                    movie, r, h);
                        neg = GET3D(negCD, MAX_RATING, this->hidden,
                                    movie, r, h);
                        // Ignore weights that were not effected
                        if ( (pos | neg) == 0.0 ) continue;
                        // Calculate the change in this weight
                        GET3D(deltaCD, MAX_RATING, this->hidden, movie, r, h) =
                            this->momentum * GET3D(deltaCD, MAX_RATING, 
                                                   this->hidden, movie, r, h)
                            + this->rate
                                * ((data_t)(pos - neg) 
                                   - DECAY * GET3D(this->weights, MAX_RATING,
                                                   this->hidden, movie, r, h));
                        // Update the weight matrix
                        GET3D(this->weights, MAX_RATING, this->hidden,
                              movie, r, h) += GET3D(deltaCD, MAX_RATING,
                                                    this->hidden, movie, r, h);
                    }
                }

                // Update hidden unit biases

                pos = posHiddenAct[h];
                neg = negHiddenAct[h];
                // Ignore hidden unit biases for units that were never active
                if ( (pos | neg) == 0 ) continue;
                // Calculate the change in bias
                deltaHiddenBias[h] = this->momentum * deltaHiddenBias[h]
                    + this->rate * (data_t)(pos - neg);
                // Update the bias for this hidden unit
                this->hiddenBias[h] += deltaHiddenBias[h];
            }

            // For all (movie, rating) pairs
            for ( uint8_t *ind = indicator; ind < indicatorEnd; ) {
                // Extract the next movie index
                getMovie(ind, movie);
                // For each individual softmax unit for this movie
                for ( int r = 0; r < MAX_RATING; ++r ) {
                    pos = GET2D(posVisibleAct, MAX_RATING, movie, r);
                    neg = GET2D(posVisibleAct, MAX_RATING, movie, r);
                    // Ignore visible unit biases for units that were never
                    // active
                    if ( (pos | neg) == 0.0 ) continue;
                    // Calculate the change in bias
                    GET2D(deltaVisibleBias, MAX_RATING, movie, r) =
                        this->momentum 
                            * GET2D(deltaVisibleBias, MAX_RATING, movie, r)
                        + this->rate * (data_t)(pos - neg);
                    // Update the bias for this softmax unit
                    GET2D(this->visibleBias, MAX_RATING, movie, r) +=
                        GET2D(deltaVisibleBias, MAX_RATING, movie, r);
                }
            }

            std::fill(posCD, 
                      posCD + this->movies * MAX_RATING * this->hidden, 0);
            std::fill(negCD, 
                      negCD + this->movies * MAX_RATING * this->hidden, 0);
            std::fill(posVisibleAct,
                      posVisibleAct + this->movies * MAX_RATING, 0);
            std::fill(negVisibleAct,
                      negVisibleAct + this->movies * MAX_RATING, 0);
            std::fill(posHiddenAct, posHiddenAct + this->hidden, 0);
            std::fill(negHiddenAct, negHiddenAct + this->hidden, 0);
        } // For all users
#ifndef NDEBUG
    end = std::chrono::system_clock::now();
    minutes_elapsed = end - start;
    cout << "Finished epoch " << (epoch + 1) << " of RBM training in " 
         << minutes_elapsed.count() << " minutes" << endl;
#endif
    } // Epochs

    delete [] hiddenProbs;
    delete [] hiddenStatesBuffer;
    delete [] posHiddenStates;
    delete [] posHiddenAct;
    delete [] negHiddenStates;
    delete [] negHiddenAct;

    delete [] visibleProbs;
    delete [] visibleProbsRMSE;
    delete [] posVisibleAct;
    delete [] negVisibleAct;
    delete [] softmax;

    delete [] deltaCD;
    delete [] deltaVisibleBias;
    delete [] deltaHiddenBias;
    delete [] posCD;
    delete [] negCD;

    delete [] ratingCount;
    for ( int i = 0; i < this->users; ++i ) {
        delete [] indicators[i];
    }
    delete indicators;
}

float RBM::predict (int user, int movie, int date) {
    // Col<data_t> hiddenProbabilities(this->hidden, fill::zeros);
    // Col<uint8_t> hiddenStates(this->hidden, fill::zeros);
    // Mat<data_t> visibleProbabilities(MAX_RATING, this->movies, fill::zeros);

    // // Load this user's cached indicator matrix
    // Mat<data_t> indicator = conv_to<Mat<data_t>>::from(
    //     loadIndicator(this->movies, user));
    // // Find the indices of all columns (movies) that have a rating
    // uvec ratedMovies = find(any(indicator, 0));
    // // For each hidden unit
    // for ( int j = 0; j < this->hidden; ++j ) {
    //     // Select the weights for set visible units
    //     indicator %= this->weights.slice(j);
    //     // Accumulate the weights for this hidden unit
    //     hiddenProbabilities[j] = sum(nonzeros(indicator));
    // }
    // // Add bias contributions for hidden units
    // hiddenProbabilities += this->hiddenBias;
    // // Calculate P[h_j = 1 | V] (Eq. 9 of Salakhutdinov, Mnih,
    // // & Hinton 2007)
    // hiddenProbabilities = sigmoid<Mat<data_t>>(hiddenProbabilities);
    // // Sample the hidden states after computing the activation
    // // probabilities
    // hiddenStates = conv_to<Col<uint8_t>>::from(
    //     randomBernoulli<data_t>(hiddenProbabilities));
    // // For each movie that this user rated
    // for ( urowvec::const_iterator it = ratedMovies.begin();
    //       it != ratedMovies.end(); ++it ) {
    //     // Fill the corresponding buffer column with ones
    //     indicator.col(*it).ones();
    // }
    // // Mark the column for the new movie as active
    // indicator.col(movie).ones();
    // // Find the indices of all active sampled hidden units
    // uvec indexBuffer = find(hiddenStates);
    // // For each active hidden unit
    // for ( urowvec::const_iterator it = indexBuffer.begin();
    //       it != indexBuffer.end(); ++it ) {
    //     // Select the weights for active softmax units
    //     indicator %= this->weights.slice(*it);
    //     // Accumulate weights from this hidden unit
    //     visibleProbabilities += indicator;
    // }
    // // Add bias unit contributions
    // visibleProbabilities += this->visibleBias;
    // // Calculate P[v_q^k == 1 | h] (Eq. 10 of Salakhutdinov, Mnih,
    // // & Hinton 2007)
    // visibleProbabilities = 
    //     normalise(sigmoid<Mat<data_t>>(visibleProbabilities));
    // // Compute the expected value of the rating for this movie
    // float rating = visibleProbabilities(0, movie)
    //     + 2 * visibleProbabilities(1, movie)
    //     + 3 * visibleProbabilities(2, movie)
    //     + 4 * visibleProbabilities(3, movie)
    //     + 5 * visibleProbabilities(4, movie);

    // return rating;
    return 0.0;
}
