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

#if !defined(NDEBUG) || !defined(NTIME)
#include <chrono>
#include <iostream>
#endif

#include <rbm.hh>

#define CACHE_EXT ".bin"
#define DATA_DIR "data/rbm_cached/"
#define INDICATOR_DATA_DIR DATA_DIR "users/"
#define PMF_PATH DATA_DIR "rating_pmf" CACHE_EXT

// Calculate an index in a flat "2d" array
#define GET2DINDEX(d1, i, j) (d1 * i + j)
// Get an element in a flat "2d" array
#define GET2D(array, d1, i, j) *(array + GET2DINDEX(d1, i, j))
// Get a pointer to an element in a flat "2d" array
#define GET2DPTR(array, d1, i, j) array + GET2DINDEX(d1, i, j)

// Width of a (movie, rating) pair in words (assuming 8-byte words)
#define PAIR_WIDTH 3

#define EPOCHS 10
#define CD_STEPS 1
#define DECAY 0.0001

RBM::RBM (int users, int movies, int hidden, float rate, float momentum) : 
    users(users), movies(movies), hidden(hidden), rate(rate), 
    momentum(momentum) {
    // Allocate weight matrix
    this->weights = new data_t*[hidden];
    for ( int h = 0; h < hidden; ++h ) {
        this->weights[h] = new data_t[movies * MAX_RATING];
    }
    // Allocate visible biases
    this->visibleBias = new data_t[movies * MAX_RATING]();
    // Allocate hidden biases
    this->hiddenBias = new data_t[hidden]();
}

RBM::~RBM () {
    for ( int h = 0; h < hidden; ++h ) {
        delete [] this->weights[h];
    }
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
inline static void cacheIndicator (std::vector<struct rating_t> **indicators,
                                   const fmat &data, int &col, 
                                   const int &user) {
    // Vector for storing (movie, rating) pairs (indices of sparse indicator
    // matrix)
    std::vector<struct rating_t> *ratings = new std::vector<struct rating_t>();
    // While the current column of data is for this user
    while ( std::lround(data.at(USER_ROW, col)) == user ) {
        struct rating_t rating { 0, 0, 0 };
        rating.movie = std::lround(data.at(MOVIE_ROW, col));
        rating.score = std::lround(data.at(RATING_ROW, col));
        // Record the rating for this movie
        ratings->push_back(rating);
        // Check the next column
        ++col;
    }
    // Get the destination path of the cache
    std::string cachePath = userCachePath(user);
    // Cache the matrix
    std::ofstream indicatorCache(cachePath, ios::binary);
    indicatorCache.seekp(indicatorCache.tellp());
    indicatorCache.write((char *) ratings->data(), 
                         ratings->size() * sizeof(struct rating_t));
    indicatorCache.close();
    // Put the "matrix" in the array of indicator matrices
    indicators[user] = ratings;
}

// Load a cached indicator matrix
static void loadIndicator (std::vector<struct rating_t> **indicators, 
                           const int &user) {
    // Get the source path of the cached indicator matrix for this user
    std::string cachePath = userCachePath(user);
    struct stat statBuffer;
    stat(cachePath.c_str(), &statBuffer);
    // Open this user's cached (sparse) indicator matrix
    std::ifstream indicatorCache(cachePath, ios::binary);
    // Calculate the number of (movie, rating) pairs for this user
    int nratings = statBuffer.st_size / sizeof(struct rating_t);
    // Allocate an array to store all of the (movie, rating) pairs
    std::vector<struct rating_t> *ratings =
        new std::vector<struct rating_t>(nratings);
    // Load this user's cached (sparse) indicator matrix
    indicatorCache.read((char *) ratings->data(), 
                        nratings * sizeof(struct rating_t));
    indicatorCache.close();
    // Put the matrix in the array of indicator matrices
    indicators[user] = ratings;
}

void RBM::train (const fmat &data) {
    // Allocate an array for storing each user's sparse indicator matrix
    // This array is not contiguous in memory on purpose; we don't want
    // all users' (sparse) indicator matrices taking up cache space when we
    // only need one at a time, and it makes indexing way easier
    std::vector<struct rating_t> **const indicators = 
        new std::vector<struct rating_t>*[this->users];

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
    for ( int h = 0; h < hidden; ++h ) {
        for ( int i = 0; i < this->movies * MAX_RATING; ++i ) {
            this->weights[h][i] = normal(engine);
        }
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
        std::cout << "cols: " << data.n_cols << "; rows: " << data.n_rows << std::endl;
#endif
        int movie, rating;
        // For each column in the data matrix (rating entry)
        for ( unsigned i = 0; i < data.n_cols; ++i ) {
            movie = std::lround(data.at(MOVIE_ROW, i));
            rating = std::lround(data.at(RATING_ROW, i));
            // Increment the count for that movie, rating pair
            GET2D(this->visibleBias, MAX_RATING, movie, rating) += 1.0;

        }
        data_t total;
        for ( int i = 0; i < this->movies * MAX_RATING; i += MAX_RATING ) {
            // Compute the total number of ratings for this movie
            total = this->visibleBias[i];
            total += this->visibleBias[i + 1];
            total += this->visibleBias[i + 2];
            total += this->visibleBias[i + 3];
            total += this->visibleBias[i + 4];
            if ( total < 1 ) continue;
            // Normalize the individual counts to produce probabilities
            this->visibleBias[i] *= 1.0 / total;
            this->visibleBias[i + 1] *= 1.0 / total;
            this->visibleBias[i + 2] *= 1.0 / total;
            this->visibleBias[i + 3] *= 1.0 / total;
            this->visibleBias[i + 4] *= 1.0 / total;
        }
        // Open the binary file used to cache the rating pmf's
        std::ofstream biasCache(PMF_PATH, ios::binary);
        // Cache the pmf matrix (initial biases of the visible units)
        biasCache.write((char *) this->visibleBias, 
                        this->movies * MAX_RATING * sizeof(data_t));
        biasCache.close();
    } else {
#ifndef NDEBUG
        std::cout << "Loading cached rating pmf's" << std::endl;
#endif
        // Open the binary file used to cache the rating pmf's
        std::ifstream biasCache(PMF_PATH, ios::binary);
        // Initialize biases of visible units
        biasCache.read((char *) this->visibleBias,
                       this->movies * MAX_RATING * sizeof(data_t));
        biasCache.close();
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
            cacheIndicator(indicators, data, col, user);
        }
    // If only some matrices were not found
    } else if ( present < this->users ) {
#ifndef NDEBUG
        std::cout << "Caching missing indicator matrices" << std::endl;
#endif
        // Let uid be the user ID in the column of data specified by index.
        // probe returns -1 if the key is less than uid, 1 if it's greater,
        // and 0 otherwise (key == uid)
        const std::function<int(fmat, int, int)> probe = 
        [&] (fmat data, int key, int index) {
            return ( key < std::lround(data.at(USER_ROW, index)) ) ? -1 : 
                   (( key > std::lround(data.at(USER_ROW, index)) ) ? 1 : 0 );
        };
        int col;
        // For each user
        for ( int user = 0; user < this->users; ++user ) {
            // If their indicator matrix is cached
            if ( found[user] ) {
                // Load it into memory
                loadIndicator(indicators, user);
                continue;
            }
            // Otherwise, serch for a rating by this user in the data
            col = binary_search<fmat, int>(data, user, probe, 0, 
                                                  data.n_cols - 1);
            // Rewind past their first rating
            while ( col > 0 && std::lround(data.at(USER_ROW, col)) == user ) {
                --col;
            }
            // Move to the user's first rating if necessary
            col += col > 0;
            // Cache the indicator matrix for this user
            cacheIndicator(indicators, data, col, user);
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
            loadIndicator(indicators, user);
        }
    }

    // Begin training procedure

    // Activation probabilities for the hidden units
    // data_t *const hiddenProbs = new data_t[this->hidden]();
    // Store the state of the hidden units across contrastive divergence steps
    uint8_t *const hiddenStatesBuffer = new uint8_t[this->hidden]();
    // Hidden states sampled from the data
    uint8_t *const posHiddenStates = new uint8_t[this->hidden]();
    // Activation histogram for hidden units sampled from the data
    uint8_t *const posHiddenAct = new uint8_t[this->hidden]();
    // Hidden states sampled using contrastive divergence (approximation of
    // sampling from the model's distribution, which is intractable)
    uint8_t *const negHiddenStates = new uint8_t[this->hidden]();
    // Activation histogram using contrastive divergence
    uint8_t *const negHiddenAct = new uint8_t[this->hidden]();

    // Activation probabilities for the visible units
    data_t *const visibleProbs = new data_t[this->movies * MAX_RATING]();
    // Non-regularized activation probabilities for the visible units, used to
    // compute RMSE
    data_t *const visibleProbsRMSE = new data_t[this->movies * MAX_RATING]();
    // Activation histogram for visible units sampled from the data
    uint8_t *const posVisibleAct = new uint8_t[this->movies * MAX_RATING]();
    // Activation histogram for visible units sampled using contrastive
    // divergence
    uint8_t *const negVisibleAct = new uint8_t[this->movies * MAX_RATING]();
    // Movie-id indexed array used as a buffer for storing the sampled states
    // of the visible softmax units
    uint8_t *const softmax = new uint8_t[this->movies]();

    // Change in the visible unit biases
    data_t *const deltaVisibleBias = new data_t[this->movies * MAX_RATING]();
    // Change in the hidden unit biases
    data_t *const deltaHiddenBias = new data_t[this->hidden]();
    // Change in the weight matrix
    data_t **const deltaCD = new data_t*[this->hidden]();
    // Contrastive divergence 
    uint8_t **const posCD = new uint8_t*[this->hidden]();
    uint8_t **const negCD = new uint8_t*[this->hidden]();
    for ( int h = 0; h < this->hidden; ++h ) {
        posCD[h] = new uint8_t[this->movies * MAX_RATING]();
        negCD[h] = new uint8_t[this->movies * MAX_RATING]();
        deltaCD[h] = new data_t[this->movies * MAX_RATING]();
    }

#ifndef NDEBUG
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> seconds_elapsed;
#endif
#ifndef NTIME
    std::chrono::time_point<std::chrono::system_clock> 
        user_start, user_end, section_start, section_end;
#endif

    // Initialize uniform distribution once
    std::uniform_real_distribution<data_t> uniform(0.0, 1.0);
    bool active;

    data_t *__restrict__ weight, *__restrict__ delta;
    uint8_t *__restrict__ pos, *__restrict__ neg;
    unsigned visInd;

    // TODO: add per-epoch RMSE computation

#ifndef NDEBUG
    std::cout << "Beginning to learn" << std::endl;
#endif
    // For the specified number of epochs (should be while RMSE is decreasing)
    for ( int epoch = 0; epoch < EPOCHS; ++epoch ) {
        start = std::chrono::system_clock::now();
        // For each user
        for ( int user = 0; user < this->users; ++user ) {
#ifndef NTIME
            user_start = std::chrono::system_clock::now();
#endif
            // This user's sparse indicator matrix
            std::vector<struct rating_t> *const indicator = indicators[user];

            // Run a training iteration until we have to sample the visible
            // units, i.e., sample the hidden units given the data

            // For all (movie, rating) pairs (active visible units)
            for ( std::vector<struct rating_t>::const_iterator it = 
                  indicator->cbegin(); it != indicator->cend(); ++it ) {
                // Add bias contribution of this set visible unit (sample its
                // state from the data)
                GET2D(posVisibleAct, MAX_RATING, it->movie, it->score) += 1;
            }
            // For each hidden unit
            for ( int h = 0; h < this->hidden; ++h ) {
                weight = this->weights[h];
                data_t prob = 0;
                // Calculate the sum of the elements of the Schur product of
                // this user's indicator matrix & the hth slice of the weight
                // cube
                for ( std::vector<struct rating_t>::const_iterator it =
                      indicator->cbegin(); it != indicator->cend(); ++it ) {
                    // Accumulate the contribution of this set visible unit
                    prob += GET2D(weight, MAX_RATING, it->movie, it->score);
                }
                // Calculate P[h_j = 1 | V] (Eq. 9 of Salakhutdinov, Mnih,
                // & Hinton 2007)
                prob = sigmoid<data_t>(prob + this->hiddenBias[h]);
                // Sample the state of this hidden unit
                active = prob > uniform(engine);
                // Record the activation probability & sampled state of this
                // unit for training purposes (sample from data)
                posHiddenStates[h] = hiddenStatesBuffer[h] = (uint8_t) active;
                posHiddenAct[h] = (uint8_t) active;

                // For all (movie, rating) pairs
                for ( std::vector<struct rating_t>::const_iterator it = 
                      indicator->cbegin(); it != indicator->cend(); ++it ) {
                    // Calculate non-regularized probabilities for RMSE
                    // reporting
                    visInd = GET2DINDEX(MAX_RATING, it->movie, 0);
                    // For each softmax unit
                    for ( int r = 0; r < MAX_RATING; ++r ) {
                        // Calculate non-regularized activation probability
                        visibleProbsRMSE[visInd + r] = 
                            prob * weight[visInd + r];
                    }
                }
            }
#ifndef NTIME
            section_start = std::chrono::system_clock::now();
#endif
            // Run the desired number of contrastive divergence iterations
            for ( int k = CD_STEPS; k > 0; --k ) {
                const bool lastStep = k <= 1;
                
                // Begin reconstruction

                // For each hidden unit
                for ( int h = 0; h < this->hidden; ++h ) {
                    // If it is not active, skip it
                    if ( ! hiddenStatesBuffer[h] ) continue;
                    // For all (movie, rating) pairs
                    for ( std::vector<struct rating_t>::const_iterator it = 
                          indicator->cbegin(); it != indicator->cend(); 
                          ++it ) {
                        // Accumulate contributions from this set hidden
                        // unit
                        visInd = GET2DINDEX(MAX_RATING, it->movie, 0);
                        // For each softmax unit
                        for ( int r = 0; r < MAX_RATING; ++r ) {
                            // Accumulate contribution to softmax unit
                            visibleProbs[visInd + r] += weight[visInd + r];
                        }
                    }
                }
                // For all (movie, rating) pairs
                for ( std::vector<struct rating_t>::iterator it = 
                      indicator->begin(); it != indicator->end(); ++it ) {
                    visInd = GET2DINDEX(MAX_RATING, it->movie, 0);
                    // For each softmax unit
                    for ( int r = 0; r < MAX_RATING; ++r ) {
                        // Calculate P[v_q^k == 1 | h] (Eq. 10 of
                        // Salakhutdinov, Mnih, & Hinton 2007)
                        visibleProbs[visInd + r] = 
                            sigmoid<data_t>(visibleProbs[visInd + r] 
                                            + this->visibleBias[visInd + r]);
                    }
                    // Normalize the activation probabilities
                    data_t total = 
                        (visibleProbs[visInd] + visibleProbs[visInd + 1]) +
                        ((visibleProbs[visInd + 2] + visibleProbs[visInd + 3])
                         + visibleProbs[visInd + 4]);
                    for ( int r = 0; r < MAX_RATING; ++r ) {
                        visibleProbs[visInd + r] *= 1.0 / total;
                    }
                    // Sample the state of this visible softmax unit (sample
                    // from approximation of model)
                    int8_t sampledRating = -1;
                    data_t r = uniform(engine);
                    do {
                        r -= visibleProbs[visInd + ++sampledRating];
                    } while ( r > 0 );
                    // Record the (zero-indexed) sampled rating
                    it->softmax = (uint8_t) sampledRating;
                }

                // Sample the states of the hidden units given the states of
                // visible units sampled from the approximation of the model

                // For each hidden unit
                for ( int h = 0; h < this->hidden; ++h ) {
                    weight = this->weights[h];
                    data_t prob = 0;
                    // For all (movie, rating) pairs
                    for ( std::vector<struct rating_t>::const_iterator it = 
                          indicator->cbegin(); it != indicator->cend(); 
                          ++it ) {
                        // Accumulate contribution of the sampled visible unit 
                        prob += GET2D(weight, MAX_RATING,
                                      it->movie, it->softmax);
                    }
                    // Calculate P[h_j = 1 | V] (Eq. 9 of Salakhutdinov, Mnih,
                    // & Hinton 2007) from the sampled data
                    prob = sigmoid<data_t>(prob + this->hiddenBias[h]);
                    // Sample the state of this hidden unit
                    active = prob > uniform(engine);
                    // Record the sampled state of this unit for training
                    // purposes (sample from approximation of model)
                    negHiddenStates[h] = (uint8_t) active;
                }
                // Reset sampled visible unit activation probabilities
                memset(visibleProbs, 0, this->movies * MAX_RATING);
                // If this is the last CD step
                if ( lastStep ) {
                    // For each hidden unit
                    for ( int h = 0; h < this->hidden; ++h ) {
                        // Train on this data
                        negHiddenAct[h] = negHiddenStates[h];
                    }
                    // For all (movie, rating) pairs 
                    for ( std::vector<struct rating_t>::const_iterator it =
                          indicator->cbegin(); it != indicator->cend(); 
                          ++it ) {
                        // Train on the sampled data
                        GET2D(negVisibleAct, MAX_RATING, 
                              it->movie, it->softmax) += 1;
                    }
                    continue;
                }
                // Load the sampled states of the hidden unit into the
                // buffer, for the next contrastive divergence step
                for ( int h = 0; h < this->hidden; ++h ) {
                    hiddenStatesBuffer[h] = negHiddenStates[h];
                }
            } // CD-k iterations
#ifndef NTIME
            section_end = std::chrono::system_clock::now();
            seconds_elapsed = section_end - section_start;
            std::cout << "CD-k comleted in " << seconds_elapsed.count()
                      << " seconds for user " << user + 1 << std::endl;
            section_start = std::chrono::system_clock::now();
#endif
            // Accumulate changes in the weights

            // For each hidden unit
            for ( int h = 0; h < this->hidden; ++h ) {
                neg = negCD[h];
                // For all (movie, rating) pairs
                for ( std::vector<struct rating_t>::const_iterator it = 
                      indicator->cbegin(); it != indicator->cend(); ++it ) {
                    // Accumulate contributions from Gibbs sampling
                    GET2D(neg, MAX_RATING, it->movie, it->softmax)
                        += negHiddenStates[h];
                }
            }
            // For each hidden unit
            for ( int h = 0; h < this->hidden; ++h ) {
                // Skip units not activated when sampled given the data
                // distribution
                if ( ! posHiddenStates[h] ) continue;
                pos = posCD[h];
                // For all (movie, rating) pairs
                for ( std::vector<struct rating_t>::const_iterator it = 
                      indicator->cbegin(); it != indicator->cend(); ++it ) {
                    // Accumulate contributions the data distribution
                    GET2D(pos, MAX_RATING, it->movie, it->score) += 1;
                }
            }

            // Update weights

            // For each hidden unit
            for ( int h = 0; h < this->hidden; ++h ) {
                pos = posCD[h];
                neg = negCD[h];
                delta = deltaCD[h];
                weight = this->weights[h];
                // For all (movie, rating) pairs
                for ( std::vector<struct rating_t>::const_iterator it = 
                      indicator->cbegin(); it != indicator->cend(); ++it ) {
                    visInd = GET2DINDEX(MAX_RATING, it->movie, 0);
                    // For ecah softmax unit
                    for ( int r = 0; r < MAX_RATING; ++r ) {
                        // If it was ever activated
                        if ( (pos[visInd + r] | neg[visInd + r]) != 0 ) {
                            // Calculate the change in this weight
                            delta[visInd + r] = 
                                this->momentum * delta[visInd + r]
                                + this->rate * 
                                    ((data_t)(pos[visInd + r] 
                                              - neg[visInd + r])
                                     - DECAY * weight[visInd + r]);
                            // Update the weight matrix
                            weight[visInd + r] += delta[visInd + r];
                        }
                    }
                }
            }

            // Update visible biases

            // For all (movie, rating) pairs
            for ( std::vector<struct rating_t>::const_iterator it = 
                  indicator->cbegin(); it != indicator->cend(); ++it ) {
                visInd = GET2DINDEX(MAX_RATING, it->movie, 0);
                // For each softmax unit
                for ( int r = 0; r < MAX_RATING; ++r ) {
                    // If it was ever activated
                    if ( (posVisibleAct[visInd + r]
                          | negVisibleAct[visInd + r]) != 0 ) {
                        // Calculate the change in this bias
                        deltaVisibleBias[visInd + r] = 
                            this->momentum * deltaVisibleBias[visInd + r]
                            + this->rate
                                * (data_t)(posVisibleAct[visInd + r] 
                                           - negVisibleAct[visInd + r]);
                        // Update the bias of this unit
                        this->visibleBias[visInd + r] +=
                            deltaVisibleBias[visInd + r];
                    }
                }
            }

            // Update hidden biases

            // For each hidden unit
            for ( int h = 0; h < this->hidden; ++h ) {
                // Ignore hidden unit biases for units that were never active
                if ( (posHiddenAct[h] | negHiddenAct[h]) == 0 ) continue;
                // Calculate the change in bias
                deltaHiddenBias[h] = this->momentum * deltaHiddenBias[h]
                    + this->rate * (data_t)(posHiddenAct[h] - negHiddenAct[h]);
                // Update the bias for this hidden unit
                this->hiddenBias[h] += deltaHiddenBias[h];
            }
#ifndef NTIME
            section_end = std::chrono::system_clock::now();
            seconds_elapsed = section_end - section_start;
            std::cout << "weight & bias updates completed in "
                      << seconds_elapsed.count() << " seconds for user " 
                      << user + 1 << std::endl;
            section_start = std::chrono::system_clock::now();
#endif

            // TODO: is there a way to avoid zeroing all these arrays?
            //       is reallocating faster? can I bit pack an over-allocated
            //       array?

            memset(posVisibleAct, 0, this->movies * MAX_RATING);
            memset(negVisibleAct, 0, this->movies * MAX_RATING);
            for ( int h = 0; h < this->hidden; ++h ) {
                memset(posCD[h], 0, this->movies * MAX_RATING);
                memset(negCD[h], 0, this->movies * MAX_RATING);
            } 
#ifndef NTIME
            user_end = std::chrono::system_clock::now();
            seconds_elapsed = user_end - section_start;
            std::cout << "Zeroed loop arrays in " << seconds_elapsed.count()
                      << " seconds for user " << user + 1 << std::endl;
            seconds_elapsed = user_end - user_start;
            std::cout << "Processed user " << user + 1 << " of epoch "
                      << epoch + 1 << " in " << seconds_elapsed.count()
                      << " seconds" << std::endl;
#endif
        } // For all users
    end = std::chrono::system_clock::now();
    seconds_elapsed = end - start;
    cout << "Finished epoch " << (epoch + 1) << " of RBM training in " 
         << seconds_elapsed.count() << " seconds" << endl;
    } // Epochs

    // delete [] hiddenProbs;
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

    delete [] deltaVisibleBias;
    delete [] deltaHiddenBias;

    for ( int h = 0; h < this->hidden; ++h ) {
        delete [] posCD[h];
        delete [] negCD[h];
        delete [] deltaCD[h];
    }
    delete [] deltaCD;
    delete [] posCD;
    delete [] negCD;

    for ( int i = 0; i < this->users; ++i ) {
        delete indicators[i];
    }
    delete [] indicators;
}

// void RBM::save(const std::string &cachePath) {

// }

fmat RBM::predict (const fmat &targets) {
    // Activation probabilities for hidden units
    data_t *hiddenProbs = new data_t[this->hidden]();
    // Activation probabilities for visible units
    data_t *visibleProbs = new data_t[MAX_RATING * this->movies]();
    // Output matrix (for storing predictions)
    fmat output(3, targets.n_cols);
    // Column index in the matrix of targets, and outputs, respectively
    unsigned col = 0, outputCol = 0, visInd;
    data_t datum;
    data_t *__restrict__ weight;
#ifndef NTIME
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> seconds_elapsed;
#endif

    while ( col < targets.n_cols ) {
#ifndef NTIME
        start = std::chrono::system_clock::now();
#endif
        int user = std::lround(targets.at(USER_ROW, col));
        // Get the source path of the cached indicator matrix for this user
        std::string cachePath = userCachePath(user);
        struct stat statBuffer;
        stat(cachePath.c_str(), &statBuffer);
        // Open this user's cached (sparse) indicator matrix
        std::ifstream indicatorCache(cachePath, ios::binary);
        // Calculate the number of (movie, rating) pairs for this user
        int nratings = statBuffer.st_size / sizeof(struct rating_t);
        // Allocate a vector to store all of the (movie, rating) pairs
        std::vector<struct rating_t> indicator(nratings);
        // Load this user's cached (sparse) indicator matrix
        indicatorCache.read((char *) indicator.data(),
                            nratings * sizeof(struct rating_t));
        indicatorCache.close();    

        // Sample the hidden units given the data

        // For each hidden unit
        for ( int h = 0; h < this->hidden; ++h ) {
            weight = this->weights[h];
            hiddenProbs[h] = 0;
            // Calculate the sum of the elements of the Schur product of
            // this user's indicator matrix & the hth slice of the weight
            // cube
            for ( std::vector<struct rating_t>::const_iterator it = 
                  indicator.cbegin(); it != indicator.cend(); ++it ) {
                // Accumulate the contribution of this set visible unit
                hiddenProbs[h] += GET2D(weight, MAX_RATING,
                                        it->movie, it->score);
            }
            // Calculate P[h_j = 1 | V] (Eq. 9 of Salakhutdinov, Mnih,
            // & Hinton 2007)
            hiddenProbs[h] =
                sigmoid<data_t>(hiddenProbs[h] + this->hiddenBias[h]);
        }

        // Reconstruct visible units

        // Vector for storing the indices of movies whose ratings we need to
        // predict (that do not exist in the data)
        std::vector<int> targetMovies;
        // Extract all movie indices from the input matrix (assumes input is
        // sorted by user id) for this user
        while ( col < targets.n_cols && targets.at(USER_ROW, col) == user ) {
            targetMovies.push_back(std::lround(targets.at(MOVIE_ROW, col++)));
        }

        // Compute activation probabilities of the visible units

        // For each hidden unit
        for ( int h = 0; h < this->hidden; ++h ) {
            datum = hiddenProbs[h];
            weight = this->weights[h];
            // For all movies we need ratings for
            for ( std::vector<int>::const_iterator it = 
                  targetMovies.begin(); it != targetMovies.end(); ++it ) {
                visInd = GET2DINDEX(MAX_RATING, *it, 0);
                // For ecah softmax unit
                for ( int r = 0; r < MAX_RATING; ++r ) {
                    // Accumulate contribution to this softmax unit
                    visibleProbs[visInd + r] += datum * weight[visInd + r];
                }
            }
        }
        // For all movies we need ratings for
        for ( std::vector<int>::const_iterator it = targetMovies.cbegin();
              it != targetMovies.cend(); ++it ) {
            visInd = GET2DINDEX(MAX_RATING, *it, 0);
            // For each softmax unit
            for ( int r = 0; r < MAX_RATING; ++r ) {
                // Calculate P[v_q^k == 1 | h] (Eq. 10 of Salakhutdinov,
                // Mnih, & Hinton 2007)
                visibleProbs[visInd + r] = 
                    sigmoid<data_t>(visibleProbs[visInd + r] 
                                    + this->visibleBias[visInd + r]);
            }
            // Normalize the activation probabilities
            data_t total = 
                (visibleProbs[visInd] + visibleProbs[visInd + 1]) +
                ((visibleProbs[visInd + 2] + visibleProbs[visInd + 3])
                 + visibleProbs[visInd + 4]);
            for ( int r = 0; r < MAX_RATING; ++r ) {
                visibleProbs[visInd + r] *= 1.0 / total;
            }
        }

        // Compute the expected value (rating) for target visible units
        for ( std::vector<int>::const_iterator it = targetMovies.cbegin();
              it != targetMovies.cend(); ++it ) {
            visInd = GET2DINDEX(MAX_RATING, *it, 1);
            data_t r = visibleProbs[visInd] 
                + 2.0 * visibleProbs[visInd + 1]
                + 3.0 * visibleProbs[visInd + 2] 
                + 4.0 * visibleProbs[visInd + 3];
            // Store this rating in the output matrix
            output.at(0, outputCol) = user;
            output.at(1, outputCol) = *it;
            output.at(2, outputCol++) = r;
        }

        memset(visibleProbs, 0, this->movies * MAX_RATING * sizeof(data_t));
#ifndef NTIME
        end = std::chrono::system_clock::now();
        seconds_elapsed = end - start;
        std::cout << "Generated predictions for user " << user + 1 << " in "
                  << seconds_elapsed.count() << " seconds" << std::endl;
#endif
    }

    return output;
}

fmat RBM::predict (const std::string &targetPath) {
    fmat targets;
    targets.load(targetPath);
    return this->predict(targets);
}
