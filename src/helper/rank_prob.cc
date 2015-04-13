/* Calculate the P[rating | movie] for all movie, rating pairs.  The output
 * binary is sorted by movie ID, and the (0-indexed) ith column corresponds to
 * P[i + 1 | movie]
 */

#include <armadillo>
#include <fstream>

#include "netflix.hh"

// This code assumes that the data is already sorted by movie ID
#define DATA_PATH "data/mu/all.dta"
#define OUTPUT_PATH "data/rbmcached/rank_prob.mat"
#define SEP " "
#define PRECISION 4

using namespace arma;
using namespace netflix;
using namespace std;

inline void store_prob(int movie, float ratings [5], fmat &probabilities) {
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
    ratings[0] = 0;
    ratings[1] = 0;
    ratings[2] = 0;
    ratings[3] = 0;
    ratings[4] = 0;
}

int main() {
    ifstream dataFile(DATA_PATH);
    fmat::fixed<NUM_MOVIES, MAX_RATING> probabilities;

    int user = 0, currentMovie = 0, previousMovie = 1, date = 0, rating = 0;
    // Initialize rating counters
    float ratings [5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    // Read a line from the data file
    while ( dataFile >> user >> currentMovie >> date >> rating ) {
        // Ignore points in the qual set (that have no rating)
        if ( rating == 0 ) continue;
        // If we just processed the last rating for a movie
        if ( currentMovie != previousMovie ) {
            // Store the probabilities for that movie (convert to zero-indexed)
            store_prob(previousMovie - 1, ratings, probabilities);
        }
        // Increment the appropriate rating counter
        ++ratings[rating - 1];
        // Update last seen movie
        previousMovie = currentMovie;
    }
    // Store probabilities for the last movie (convert to zero-indexed)
    store_prob(previousMovie - 1, ratings, probabilities);
    // Close the data file
    dataFile.close();
    // Save the probability matrix
    probabilities.save(OUTPUT_PATH);

    return 0;
}