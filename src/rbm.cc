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

#define DATA_DIR "data/rbmcached/"
#define INDICATOR_DATA_DIR DATA_DIR "users/"
#define INDICATOR_DATA_PATH "data/um/new_all.dta"
#define RANK_DATA_PATH "data/mu/new_all.dta"
#define RANK_PATH DATA_DIR "rank_prob.mat"

// Helper for RBM constructor
static inline void storePMF (int movie, float ratings [5], 
                             fmat &probabilities) {
    // Calculate the total number of ratings for this movie
    float total = ratings[0] + ratings[1] + ratings[2] + ratings[3] +
                  ratings[4];
    // Store the values of P[rating | movie] for this movie
    probabilities.at(movie, 0) = ratings[0] / total;
    probabilities.at(movie, 1) = ratings[1] / total;
    probabilities.at(movie, 2) = ratings[2] / total;
    probabilities.at(movie, 3) = ratings[3] / total;
    probabilities.at(movie, 4) = ratings[4] / total;
    // Reset the ratings counters
    ratings[0] = 0.0;
    ratings[1] = 0.0;
    ratings[2] = 0.0;
    ratings[3] = 0.0;
    ratings[4] = 0.0;
}

RBM::RBM (int users, int movies, int hidden, float rate) : 
    users(users), movies(movies), hidden(hidden), rate(rate),
    weights(movies, hidden, MAX_RATING, fill::zeros), 
    visibleBias(movies, MAX_RATING, fill::zeros),
    hiddenBias(hidden, fill::zeros) { }

RBM::~RBM () { }

void RBM::train() {
    // An array of per-user indicator matrices
    SpMat<uint8_t> *data = new SpMat<uint8_t>[this->users];

    // Mean & standard deviation of our normal distribution
    const float mean = 0.0;
    const float stddev = 0.01;
    std::default_random_engine generator;
    std::normal_distribution<float> normal(mean, stddev);
    // Initialize weight matrix using our normal distribution
    for ( int i = 0; i < movies; ++i ) {
        for ( int j = 0; j < hidden; ++j ) {
            this->weights.at(i, j, 0) = normal(generator);
            this->weights.at(i, j, 1) = normal(generator);
            this->weights.at(i, j, 2) = normal(generator);
            this->weights.at(i, j, 3) = normal(generator);
            this->weights.at(i, j, 4) = normal(generator);
        }
    }

#ifdef RANDOM
    // Randomly initialize the biases of the hidden units
    for ( int i = 0; i < hidden; ++i ) {
        this->hiddenBias[i] = normal(generator);
    }
#else
    // Zero the biases of the hidden units
    this->hiddenBias.zeros();
#endif

    struct stat buffer;
    // If a cached ranking pmf matrix does not exist
    if ( stat(RANK_PATH, &buffer) != 0 ) {
#ifndef NDEBUG
        std::cout << "Computing & caching rating pmf's for all movies" 
                  << std::endl;
#endif
        // Data file, sorted by movie ID
        std::ifstream dataFile(RANK_DATA_PATH);
        int user, currentMovie, previousMovie = 1, date, rating;
        // Rating counters
        float ratings [MAX_RATING] = {0.0, 0.0, 0.0, 0.0, 0.0};

        // Read a line from the data file
        while ( dataFile >> user >> currentMovie >> date >> rating ) {
            // Ignore points in the qual set (that have no rating)
            if ( rating == 0 ) continue;
            // If we just processed the last rating for a movie
            if ( currentMovie != previousMovie ) {
                // Store the probabilities for that movie
                storePMF(previousMovie, ratings, this->visibleBias);
            }
            // Increment the appropriate rating counter
            ++ratings[rating - 1];
            // Update last seen movie
            previousMovie = currentMovie;
        }
        // Store probabilities for the last movie
        storePMF(previousMovie, ratings, this->visibleBias);

        // Close the data file
        dataFile.close();
        // Cache the pmf matrix (initial biases of the visible units)
        this->visibleBias.save(RANK_PATH);
    } else {
        // Initialize biases of visible units
        this->visibleBias.load(RANK_PATH);
    }

    // Vector for storing user indices w/o cached rating indicator matrices
    std::vector<int> missing;
    // Vector for marking indicator matrices as found
    std::vector<bool> found(this->users, 0);
    DIR *directory;
    struct dirent *entity;
    // If the indicator data directory exists
    if ( (directory = opendir(INDICATOR_DATA_DIR)) ) {
        // For every entry in the directory
        while ( (entity = readdir(directory)) != NULL && 
                entity->d_name[0] != '.' ) {
            // Convert the filename into a string
            std::string filename = string(entity->d_name);
            std::cout << entity->d_name << std::endl;
            // Trim the file's .mat extension
            filename.resize(filename.size() - 4);
            // Mark this indicator matrix as cached
            found[std::stoi(filename)] = 1;
        }
    } 

    // TODO: do something if the directory doesn't exist

    // If any matrix is not found
    if ( std::accumulate(found.cbegin(), found.cend(), 0) != this->users ) {
        // For each user
        for ( std::vector<bool>::const_iterator it = found.cbegin();
              it != found.cend(); ++it ) {
            // If their indicator matrix was not cached
            if ( ! *it ) {
                // Mark it as missing
                missing.push_back(it - found.cbegin());
            }
        }
    }
    // The path of the user's cached rating indicator matrix
    std::string userDataPath;
    // If any users do not have cached indicator matrices
    if ( missing.size() > 0 ) {
        // Open the data file (sorted by user ID)
        std::ifstream dataFile(INDICATOR_DATA_PATH);
        // Initialize user to an invalid state (simplifies for loop)
        int user = -1, movie, date, rating;
        // For each user who is missing data
        for ( std::vector<int>::const_iterator it = missing.cbegin(); 
              it != missing.cend(); ++it ) {
#ifndef NDEBUG
            std::cout << "Computing & caching indicator matrix for user "
                      << *it << std::endl;
#endif
            // Binary indicator matrix for this user; element (i, j) == 1 iff
            // this user gave movie i (0-indexed) a rating of j + 1
            SpMat<uint8_t> userData(movies, MAX_RATING);
            // If we read an extra line on the last iteration (i.e., users w/
            // missing data are sequential)
            if ( user == *it ) {
                // Record this user's rating of this movie
                userData.at(movie, rating - 1) = 1;
            }
            // Read a line from the data file
            while ( dataFile >> user >> movie >> date >> rating ) {
                // If we have not reached the first user with missing data,
                // or there is no rating for this movie, keep reading lines
                if ( user < *it || rating == 0 ) continue;
                // If we have just finished a user, stop reading lines
                if ( user > *it ) break;
                // Record this user's rating of this movie
                userData.at(movie, rating - 1) = 1;
            }
            std::ostringstream pathBuffer;
            // Construct the path for this user's data
            pathBuffer << INDICATOR_DATA_DIR << *it << ".mat";
            // Get the generated path
            userDataPath = pathBuffer.str();
            userData.save(userDataPath);
        }
    }
#ifndef NDEBUG
    std::cout << "Loading cached indicator matrices" << std::endl;
#endif
    // For each user
    for ( int i = 0; i < this->users; ++i ) {
        std::ostringstream pathBuffer;
        // Construct the path for this user's data
        pathBuffer << INDICATOR_DATA_DIR << i << ".mat";
        // Get the generated path
        userDataPath = pathBuffer.str();
        // Load the data for this user
        data[i].load(userDataPath);
    }
}

void RBM::train(const imat &data) {

}

float RBM::predict(int user, int item) { return 0.0; }
