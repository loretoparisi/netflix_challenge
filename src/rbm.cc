#ifndef NDEBUG
#include <iostream>
#endif
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cmath>
#include <cstdint>

#include <dirent.h>
#include <sys/stat.h>

#include "rbm.hh"

#define CACHE_EXT ".mat"
#define DATA_DIR "data/rbmcached/"
#define INDICATOR_DATA_DIR DATA_DIR "users/"
#define RANK_PATH DATA_DIR "rank_prob" CACHE_EXT

RBM::RBM (int users, int movies, int hidden, float rate, float momentum) : 
    users(users), movies(movies), hidden(hidden), rate(rate), 
    momentum(momentum), weights(MAX_RATING, movies, hidden, fill::zeros), 
    visibleBias(MAX_RATING, movies, fill::zeros),
    hiddenBias(hidden, fill::zeros) { }

RBM::~RBM () { }

// Convert a probability matrix to a Bernoulli matrix
inline static void toBernoulli(fmat &probabilities) {
    // Generate a uniform random matrix w/ values in the range [0, 1]
    fmat uniform = randu<fmat>(probabilities.n_rows, probabilities.n_cols);
    // Indices of probabilities greater than a (uniform) random value
    uvec greater = find(probabilities > uniform);
    // Zero the probability matrix
    uniform.zeros();
    // Set the randomly selected elements to one
    uniform.elem(greater).ones();
}

// Compute & cache an indicator matrix
inline static void cacheIndicator(const fmat &data, int &col, int movies, 
                                  int user) {
    SpMat<uint8_t> indicator(MAX_RATING, movies);
    // While the current column of data is for this user
    while ( data.at(USER_ROW, col) == user ) {
        // Set the value of the user's indicator matrix to one for
        // this rating, movie pair
        indicator.at(data.at(RATING_ROW, col) - 1, 
                     data.at(MOVIE_ROW, col)) = 1;
        // Process the next column
        ++col;
    }
    // Stream buffer for building the path of this user's cache 
    std::ostringstream cachePath;
    // Cached matrices are stored with the user's id as a filename
    cachePath << INDICATOR_DATA_DIR << user << CACHE_EXT;
    // Cache the matrix
    indicator.save(cachePath.str());
}

void RBM::train(const fmat &data) {
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
    struct stat buffer;
    // If a cached ranking pmf matrix does not exist
    if ( stat(RANK_PATH, &buffer) != 0 ) {
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
        float total;
        // For each column in the visible bias matrix (movie)
        for ( int i = 0; i < this->movies; ++i ) {
            // Compute the total number of ratings for the movie
            total = std::accumulate(this->visibleBias.begin_col(i),
                                    this->visibleBias.end_col(i), 0.0);
            // Convert counts for each rating of this movie into probabilities
            for ( fmat::col_iterator it = this->visibleBias.begin_col(i);
                  it != this->visibleBias.end_col(i); ++it ) {
                *it /= total;
            }
        }
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
    DIR *directory;
    struct dirent *entity;
    // If the indicator data directory exists
    if ( (directory = opendir(INDICATOR_DATA_DIR)) ) {
        // For every entry in the directory, which is not hidden
        while ( (entity = readdir(directory)) && 
                entity->d_name[0] != '.' ) {
            // Convert the filename into a string
            std::string filename = std::string(entity->d_name);
            // Trim the file's extension (see CACHE_EXT macro)
            filename.resize(filename.size() - 4);
            // Mark this indicator matrix as cached
            found[std::stoi(filename)] = 1;
        }
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
        const std::function<int(fmat, int, int)> probe = 
        [&] (fmat data, int key, int index) {
            return ( key < data(USER_ROW, index) ) ? -1 : 
                   (( key > data(USER_ROW, index) ) ? 1 : 0 );
        };
        int col;
        // For each user
        for ( int user = 0; user < this->users; ++users ) {
            // If their indicator matrix is cached, skip them
            if ( found[user] ) continue;
            // Otherwise, serch for a rating by this user in the data
            col = binary_search<fmat, int>(data, user, probe, 0, data.n_cols);
            // Rewind past their first rating
            while ( col > 0 && data(USER_ROW, --col) == user ) { }
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


}

float RBM::predict (int user, int item, int date) { return -1.0; }
