#include <dirent.h>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#ifndef NDEBUG
#include <chrono>
#include <iostream>
#endif

#include <rbm.hh>

#define CACHE_EXT ".mat"
#define DATA_DIR "data/rbm_cached/"
#define INDICATOR_DATA_DIR DATA_DIR "users/"
#define RANK_PATH DATA_DIR "rank_prob" CACHE_EXT

#define EPOCHS 50
#define CD_STEPS 1
#define DECAY 0.0001

RBM::RBM (int users, int movies, int hidden, float rate, float momentum) : 
    users(users), movies(movies), hidden(hidden), rate(rate), 
    momentum(momentum), weights(MAX_RATING, movies, hidden, fill::zeros), 
    visibleBias(MAX_RATING, movies, fill::zeros),
    hiddenBias(hidden, fill::zeros) { }

RBM::~RBM () { }

// Helper function for building the path to a user's cached indicator matrix
inline static std::string userCachePath(const int &user) {
    // Stream buffer for building the path of this user's cache 
    std::ostringstream cachePath;
    // Cached matrices are stored with the user's id as a filename
    cachePath << INDICATOR_DATA_DIR << user << CACHE_EXT;

    return cachePath.str();
}

// Compute & cache an indicator matrix
static void cacheIndicator(const Mat<data_t> &data, int &col, 
                           const int &movies, const int &user) {
    Mat<ind_t> indicator(MAX_RATING, movies);
    // While the current column of data is for this user
    while ( data.at(USER_ROW, col) == user ) {
        // Set the value of the user's indicator matrix to one for
        // this rating, movie pair
        indicator.at(data.at(RATING_ROW, col) - 1, 
                     data.at(MOVIE_ROW, col)) = 1;
        // Process the next column
        ++col;
    }
    // Get the destination path (of the cached indicator matrix)
    std::string cachePath = userCachePath(user);
    // Cache the matrix
    indicator.save(cachePath);
}

// Load a cached indicator matrix
inline static Mat<ind_t> loadIndicator(const int &movies, const int &user) {
    Mat<ind_t> indicator(MAX_RATING, movies);
    // Get the source path (of the cached indicator matrix)
    std::string cachePath = userCachePath(user);
    // Load the cached indicator matrix
    indicator.load(cachePath);
    // Reshape the matrix to the correct dimensions
    indicator.reshape(MAX_RATING, movies);

    return indicator;
}

void RBM::train(const Mat<data_t> &data) {
    // Mean & standard deviation of our normal distribution
    const float mean = 0.0;
    const float stddev = 0.01;
    std::mt19937 engine;
    std::normal_distribution<float> normal(mean, stddev);
#ifndef NDEBUG
    std::cout << "Initializing weight matrix" << std::endl;
#endif
    // Initialize weight matrix using our normal distribution
    this->weights.imbue( [&] () { return normal(engine); } );
#ifndef NDEBUG
    std::cout << "Initializing biases of hidden units" << std::endl;
#endif
#ifdef RANDOM
    // Randomly initialize the biases of the hidden units
    this->hiddenBias.imbue( [&] () { return normal(engine); } );
#else
    // Zero the biases of the hidden units
    this->hiddenBias.zeros();
#endif
#ifndef NDEBUG
    std::cout << "Initializing biases of the visible units" << std::endl;
#endif
    struct stat statBuffer;
    // If a cached ranking pmf matrix does not exist
    if ( stat(RANK_PATH, &statBuffer) != 0 ) {
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
            this->visibleBias.at(rating, movie) += 1;
        }
        // Normalize counts in each column to produce probabilities
        this->visibleBias = normalise(this->visibleBias);
        // Cache the pmf matrix (initial biases of the visible units)
        this->visibleBias.save(RANK_PATH);
    } else {
#ifndef NDEBUG
        std::cout << "Loading cached rating pmf's" << std::endl;
#endif
        // Initialize biases of visible units
        this->visibleBias.load(RANK_PATH);
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
    std::cout << present << " indicator matrices found" << std::endl;
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
            cacheIndicator(data, col, this->movies, user);
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
            // If their indicator matrix is cached, skip them
            if ( found[user] ) continue;
            // Otherwise, serch for a rating by this user in the data
            col = binary_search<Mat<data_t>, int>(data, user, probe, 0, 
                                                  data.n_cols - 1);
            // Rewind past their first rating
            while ( col > 0 && data.at(USER_ROW, --col) == user ) { }
            // Move to the user's first rating if necessary
            col += col > 0;
            // Cache the indicator matrix for this user
            cacheIndicator(data, col, this->movies, user);
        }
    // More cached matrices were found than there are users
    } else if ( present > this->users ) {
        std::ostringstream msg;
        msg << present << " indicator matrices found for " << this->users
            << " users" << std::endl;
        throw std::runtime_error(msg.str());
    }
#ifndef NDEBUG
    std::cout << "All indicator matrices cached" << std::endl;
#endif


    Col<data_t> hiddenProbabilities(this->hidden, fill::zeros);
    Col<ind_t> posHiddenStates(this->hidden, fill::zeros);
    Col<ind_t> posHiddenActivation(this->hidden, fill::zeros);    
    Col<ind_t> negHiddenStates(this->hidden, fill::zeros);
    Col<ind_t> negHiddenActivation(this->hidden, fill::zeros);

    Mat<data_t> visibleProbabilities(MAX_RATING, this->movies, fill::zeros);
    Mat<ind_t> posVisibleActivation(MAX_RATING, this->movies, fill::zeros);
    Mat<ind_t> negVisibleActivation(MAX_RATING, this->movies, fill::zeros);
    ind_t *softmax = new ind_t[this->movies];

    Cube<data_t> deltaCD(MAX_RATING, this->movies, this->hidden, fill::zeros);
    Mat<data_t> deltaVisibleBias(MAX_RATING, this->movies, fill::zeros);
    Col<data_t> deltaHiddenBias(this->hidden, fill::zeros);

    Cube<ind_t> posCD(MAX_RATING, this->movies, this->hidden, fill::zeros);
    Cube<ind_t> negCD(MAX_RATING, this->movies, this->hidden, fill::zeros);

#ifndef NDEBUG
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<float, std::ratio<60>> minutes_elapsed; 
#endif
    
    // For the specified number of epochs (should be while RMSE is decreasing)
    for ( int epoch = 0; epoch < EPOCHS; ++epoch ) {
#ifndef NDEBUG
        start = std::chrono::system_clock::now();
#endif
        // For each user
        for ( int user = 0; user < this->users; ++user ) {
            // Load this user's cached indicator matrix
            Mat<ind_t> indicator = loadIndicator(this->movies, user);
            // Find the indices of all columns (movies) that have a rating
            uvec ratedMovies = find(any(indicator, 0));
            // A buffer for storing contributions to the hidden units
            Mat<data_t> weightBuffer = conv_to<Mat<data_t>>::from(indicator);

            // Run a training iteration until we have to sample the visible
            // units

            // For each hidden unit
            for ( int j = 0; j < this->hidden; ++j ) {
                // Select the weights for set visible units
                weightBuffer %= this->weights.slice(j);
                // Accumulate the weights for this hidden unit
                hiddenProbabilities[j] = sum(nonzeros(weightBuffer));
            }
            // Zero the weight buffer
            weightBuffer.zeros();
            // Add bias contributions for hidden units
            hiddenProbabilities += this->hiddenBias;
            // Calculate P[h_j = 1 | V] (Eq. 9 of Salakhutdinov, Mnih,
            // & Hinton 2007)
            hiddenProbabilities = sigmoid<Mat<data_t>>(hiddenProbabilities);
            // Sample the hidden states after computing the activation
            // probabilities
            posHiddenStates = conv_to<Col<ind_t>>::from(
                randomBernoulli<data_t>(hiddenProbabilities));
            // Record activations for training purposes
            posHiddenActivation += posHiddenStates;

            // Add bias contributions for set visible units
            posVisibleActivation += indicator;
            // Buffer of (sampled) hidden states for use by CD-k algorithm
            Col<ind_t> hiddenStatesBuffer(posHiddenStates);
            // Run the desired number of contrastive divergence iterations
            for ( int k = CD_STEPS; k > 0; --k ) {
                
                // Begin reconstruction

                // For each movie that this user rated
                for ( urowvec::const_iterator it = ratedMovies.begin();
                      it != ratedMovies.end(); ++it ) {
                    // Fill the corresponding buffer column with ones
                    weightBuffer.col(*it).ones();
                }
                // Find the indices of all active sampled hidden units
                uvec indexBuffer = find(hiddenStatesBuffer);
                // For each activate hidden unit
                for ( urowvec::const_iterator it = indexBuffer.begin();
                      it != indexBuffer.end(); ++it ) {
                    // Select the weights for active softmax units
                    weightBuffer %= this->weights.slice(*it);
                    // Accumulate weights from this hidden unit
                    visibleProbabilities += weightBuffer;
                }
                // Add bias unit contributions
                visibleProbabilities += this->visibleBias;
                // Calculate P[v_q^k == 1 | h] (Eq. 10 of Salakhutdinov, Mnih,
                // & Hinton 2007)
                visibleProbabilities = 
                    normalise(sigmoid<Mat<data_t>>(visibleProbabilities));

                // Initialize uniform distribution
                std::uniform_real_distribution<float> uniform(0.0, 1.0);
                float random;
                // For each rated movie, sample the state of the softmax unit
                for ( urowvec::const_iterator it = ratedMovies.begin();
                      it != ratedMovies.end(); ++it ) {
                    // Generate a uniform random number
                    random = uniform(engine);
                    // Randomly pick a rating
                    int rating = 0;
                    random -= visibleProbabilities(rating, *it);
                    while ( random > 0 ) {
                        random -= visibleProbabilities(++rating, *it);
                    }
                    // Record the rating chosen for this movie
                    softmax[*it] = rating;
                    // Record activations for training purposes
                    negVisibleActivation(rating, *it) += 1;
                }

                // Zero the hidden unit probability buffer
                hiddenProbabilities.zeros();
                // For each hidden unit
                for ( int j = 0; j < this->hidden; ++j ) {
                    // Accumulate weights from all active visible units
                    for ( urowvec::const_iterator it = ratedMovies.begin();
                          it != ratedMovies.end(); ++it ) {
                            hiddenProbabilities[j] += 
                                this->weights(softmax[*it], *it, j);
                    }
                }
                // Add bias contributions for hidden units
                hiddenProbabilities += this->hiddenBias;
                // Calculate P[h_j = 1 | V] (Eq. 9 of Salakhutdinov, Mnih,
                // & Hinton 2007)
                hiddenProbabilities = 
                    sigmoid<Mat<data_t>>(hiddenProbabilities);
                // Sample the hidden states after computing the activation
                // probabilities
                negHiddenStates = conv_to<Col<ind_t>>::from(
                    randomBernoulli<data_t>(hiddenProbabilities));

                // If this is the last contrastive divergence step
                if ( k == 0 ) {
                    // Record activations for training purposes
                    negHiddenActivation += negHiddenStates;
                // If we are iterating again
                } else {
                    // Reset sampled probabilities
                    visibleProbabilities.zeros();
                    // Buffer the sampled hidden states
                    hiddenStatesBuffer = negHiddenStates;
                }
            }

            // Accumulate contrastive divergence contributions

            // For each hidden unit
            for ( int j = 0; j < this->hidden; ++j ) {
                // Accumulate contributions from Gibbs sampling
                for ( urowvec::const_iterator it = ratedMovies.begin();
                      it != ratedMovies.end(); ++it ) {
                    negCD(softmax[*it], *it, j) += negHiddenStates[j];
                }
                // If this hidden unit is active
                if ( posHiddenStates[j] == 1 ) {
                    // Accumulate contributions from the data
                    posCD.slice(j) += indicator;
                }
            }

            // Update weights & biases

            // For each hidden unit
            int pos, neg;
            for ( int j = 0; j < this->hidden; ++j ) {
                // For each rated movie
                for ( urowvec::const_iterator it = ratedMovies.begin();
                      it != ratedMovies.end(); ++it ) {
                    // For all ratings
                    for ( int r = 0; r < MAX_RATING; ++r ) {
                        pos = posCD(r, *it, j);
                        neg = negCD(r, *it, j);
                        // Ignore weights that were not effected
                        if ( (pos | neg) == 0.0 ) continue;
                        // Calculate the change in this weight
                        deltaCD(r, *it, j) = 
                            this->momentum * deltaCD(r, *it, j)
                            + this->rate
                                * ((data_t)(pos - neg)
                                   - DECAY * this->weights(r, *it, j));
                        // Update the weight matrix
                        this->weights(r, *it, j) += deltaCD(r, *it, j);
                    }
                }

                pos = posHiddenActivation[j];
                neg = negHiddenActivation[j];
                // Ignore hidden unit biases that were not effected
                if ( (pos | neg) == 0 ) continue;
                // Calculate the change in bias
                deltaHiddenBias[j] = this->momentum * deltaHiddenBias[j]
                    + this->rate * (data_t)(pos - neg);
                // Update the bias
                this->hiddenBias[j] += deltaHiddenBias[j];
            }

            // For each rated movie
            for ( urowvec::const_iterator it = ratedMovies.begin();
                  it != ratedMovies.end(); ++it ) {
                // For all ratings
                for ( int r = 0; r < MAX_RATING; ++r ) {
                    pos = posVisibleActivation(r, *it);
                    neg = negVisibleActivation(r, *it);
                    // Ignore visible unit biases that were not effected
                    if ( (pos | neg) == 0 ) continue;
                    // Calculate the change in bias
                    deltaVisibleBias(r, *it) = 
                        this->momentum * deltaVisibleBias(r, *it)
                        + this->rate * (data_t)(pos - neg); 
                    // Update the bias
                    this->visibleBias(r, *it) += deltaVisibleBias(r, *it);
                }
            }

            posCD.zeros();
            negCD.zeros();
            posHiddenActivation.zeros();
            negHiddenActivation.zeros();
            posVisibleActivation.zeros();
            negVisibleActivation.zeros();
        }
#ifndef NDEBUG
    end = std::chrono::system_clock::now();
    minutes_elapsed = end - start;
    cout << "Finished epoch " << (epoch + 1) << " of RBM training in " 
         << minutes_elapsed.count() << " minutes" << endl;
#endif
    }

    delete softmax;
}

float RBM::predict (int user, int item, int date) { return -1.0; }
