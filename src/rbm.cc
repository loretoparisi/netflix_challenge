#ifndef NDEBUG
#include <iostream>
#endif
#include <fstream>
#include <random>
#include <sys/stat.h>

#include "rbm.hh"

#define DATA_PATH "data/mu/all.dta"
#define RANK_PATH "data/rbmcached/rank_prob.mat"

using namespace std;

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
    weights(movies, hidden, MAX_RATING), visibleBias(movies, MAX_RATING),
    hiddenBias(hidden, fill::zeros) {
    // Initialize RNG
    default_random_engine gen;
    // Normal distribution with mean 0, standard deviation 0.01
    normal_distribution<float> dist(0.0, 0.01);
    // Initialize weight matrix
    for ( int i = 0; i < movies; ++i ) {
        for ( int j = 0; j < hidden; ++j ) {
            this->weights.at(i, j, 0) = dist(gen);
            this->weights.at(i, j, 1) = dist(gen);
            this->weights.at(i, j, 2) = dist(gen);
            this->weights.at(i, j, 3) = dist(gen);
            this->weights.at(i, j, 4) = dist(gen);
        }
    }

    struct stat buffer;
    // If the ranking pmf matrix does not exist
    if ( stat(RANK_PATH, &buffer) != 0 ) {
#ifndef NDEBUG
        cout << "Computing & caching rating pmf's for all movies" << endl;
#endif
        // Data file, sorted by movie ID
        ifstream dataFile(DATA_PATH);
        int user, currentMovie, previousMovie = 1, date, rating;
        // Rating counters
        float ratings [5] = {0.0, 0.0, 0.0, 0.0, 0.0};

        // Read a line from the data file
        while ( dataFile >> user >> currentMovie >> date >> rating ) {
            // Ignore points in the qual set (that have no rating)
            if ( rating == 0 ) continue;
            // If we just processed the last rating for a movie
            if ( currentMovie != previousMovie ) {
                // Store the probabilities for that movie
                storePMF(previousMovie - 1, ratings, this->visibleBias);
            }
            // Increment the appropriate rating counter
            ++ratings[rating - 1];
            // Update last seen movie
            previousMovie = currentMovie;
        }
        // Store probabilities for the last movie
        storePMF(previousMovie - 1, ratings, this->visibleBias);

        // Close the data file
        dataFile.close();
        // Save the probability matrix
        this->visibleBias.save(RANK_PATH);
    } else {
        // Initialize biases of visible units
        this->visibleBias.load(RANK_PATH);
    }

}

RBM::~RBM () { }

void RBM::train(const imat &data) { }

float RBM::predict(int user, int item) { return 0.0; }
