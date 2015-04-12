/* Calculate the P[rating | movie] for all movie, rating pairs.  The output
 * file is sorted by movie ID, and the (1-indexed) ith column corresponds to
 * P[i | movie] (columns are space-delimited)
 */

#include <fstream>
#include <iomanip>
#include <iostream>

// This code assumes that the data is already sorted by movie ID
#define DATA_PATH "data/mu/all.dta"
#define OUTPUT_PATH "data/rbmcached/rank_prob.dta"
#define SEP " "
#define PRECISION 4

using namespace std;

inline void store_prob(float *ratings, ofstream &outputFile) {
    // Calculate the total number of ratings for this movie
    float total = ratings[0] + ratings[1] + ratings[2] + ratings[3] +
                  ratings[4];
    // Store the values of P[rating | movie] for this movie
    outputFile << setprecision(PRECISION) 
        << ratings[0] / total << SEP << ratings[1] / total << SEP
        << ratings[2] / total << SEP << ratings[3] / total << SEP
        << ratings[4] / total << endl;
    // Reset the ratings counters
    ratings[0] = 0;
    ratings[1] = 0;
    ratings[2] = 0;
    ratings[3] = 0;
    ratings[4] = 0;
}

int main() {
    ifstream dataFile(DATA_PATH);
    ofstream outputFile(OUTPUT_PATH);

    int user = 0, currentMovie = 0, previousMovie = 1, date = 0, rating = 0;
    float ratings [5];
    // Initialize rating counters
    ratings[0] = 0;
    ratings[1] = 0;
    ratings[2] = 0;
    ratings[3] = 0;
    ratings[4] = 0;
    // Read a line from the data file
    while ( dataFile >> user >> currentMovie >> date >> rating ) {
        // Ignore points in the qual set (that have no rating)
        if ( rating == 0 ) continue;
        // If we just processed the last rating for a movie
        if ( currentMovie != previousMovie ) {
            // Store the probabilities for that movie
            store_prob(ratings, outputFile);
        }
        // Increment the appropriate rating counter
        ++ratings[rating - 1];
        // Update last seen movie
        previousMovie = currentMovie;
    }
    // Store probabilities for the last movie in the set
    store_prob(ratings, outputFile);
    
    dataFile.close();
    outputFile.close();

    return 0;
}