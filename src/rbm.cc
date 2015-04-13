#include <fstream>
#include <random>

#include "rbm.hh"

#define RANK_PATH "data/rbmcached/rank_prob.dta"

using namespace std;

RBM::RBM (float rate) {
    // Store learning rate
    this->rate = rate;
    // Initialize RNG
    default_random_engine gen;
    // Normal distribution with mean 0, standard deviation 0.01
    normal_distribution<float> dist(0.0, 0.01);
    // Initialize weight matrix
    for ( int i = 0; i < NUM_MOVIES; ++i ) {
        for ( int j = 0; j < HIDDEN; ++j ) {
            this->weights.at(i, j, 0) = dist(gen);
            this->weights.at(i, j, 1) = dist(gen);
            this->weights.at(i, j, 2) = dist(gen);
            this->weights.at(i, j, 3) = dist(gen);
            this->weights.at(i, j, 4) = dist(gen);
        }
    }

    // Open rank pmf data file (generated using rank_prob helper)
    ifstream rankPMF(RANK_PATH);
    // Initialize biases of visible units
    for ( int i = 0; i < NUM_MOVIES; ++i ) {
        // For each movie, initialize the bias of each rank to P[rank | movie]
        rankPMF >> this->visibleBias[i][0] >> this->visibleBias[i][1] 
                >> this->visibleBias[i][2] >> this->visibleBias[i][3]
                >> this->visibleBias[i][4];
    }
    // Close data file
    rankPMF.close();

    // Initialize biases of hidden units
    for ( int i = 0; i < HIDDEN; ++i ) {
        this->hiddenBias[i] = 0.0;
    }
}
