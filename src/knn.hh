/*
 * This class allows us to carry out KNN on the Netflix dataset. 
 */

#ifndef KNN_HH
#define KNN_HH

#include "singlealgorithm.hh"
#include "netflix.hh"
#include <armadillo>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <random>
#include <functional>
#include <iterator>
#include <array>
#include <math.h>
#include <sstream>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <queue>

// Minimum common neighbors required for decent prediction
#define MIN_COMMON 16

// Max weight elements to consider when predicting
#define MAX_W 10

// Idea of optimization, speed up is from:
// http://dmnewbie.blogspot.com/2009/06/calculating-316-million-movie.html

using namespace std;
using namespace arma;
using namespace netflix; // challenge-related constants/functions.

struct mu_pair {
    unsigned int user;
    unsigned char rating;
};

struct um_pair {
    unsigned short movie;
    unsigned char rating;
};

// Pearson intermediates, as described in dmnewbie's blog
struct s_inter {
    float x; // sum of ratings of movie i
    float y; // sum of ratings of movie j
    float xy; // sum (rating_i * rating_j)
    float xx; // sum (rating_i^2)
    float yy; // sum (rating_j^2)
    unsigned int n; // Num users who rated both movies
};

// To be stored in P
struct s_pear {
    float p;
    unsigned int common;
};

// Used during prediction
// As per the blogpost
struct s_neighbors {
    // Num users who watched both m and n
    unsigned int common;

    // Avg rating of m, n
    float m_avg;
    float n_avg;

    // Rating of n
    float n_rating;

    // Pearson coeff
    float pearson;

    float p_lower;
    float weight;
};

class KNN : public SingleAlgorithm
{
private:
    // Testing data set.
    const string &trainFilenameUM;
    const string &trainFilenameMU;
    const string &pFilename;
    const string &qualFilename;
    const string &outputFilename;

    // In test mode => test data has "answers."
    // For qual data, test mode is "false."
    bool test;

    // um: for every user, stores (movie, rating) pairs.
    vector<um_pair> um[NUM_USERS];

    // mu: for every movie, stores (user, rating) pairs.
    vector<mu_pair> mu[NUM_MOVIES];


    // Pearson coefficients for every movie pair
    // When accessing P[i][j], it must always be the case that:
    // i <= j (symmetry is assumed)
    s_pear P[NUM_MOVIES][NUM_MOVIES];
    float movieAvg[NUM_MOVIES];

    float predict(int user, int item, int date);

public:
    KNN(const string &trainFilenameUM, const string &trainFilenameMU,
        const string &pFilename, const string &qualFilename,
        const string &outputFilename, bool test);
    void train(const imat &data);
    ~KNN();
    void loadData();
    void calcP();
    void saveP();
    void loadP();
    void output();
};

#endif // KNN_HH
