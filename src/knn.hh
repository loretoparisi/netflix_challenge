/*
 * This class allows us to carry out KNN on the Netflix dataset. 
 */

#ifndef KNN_HH
#define KNN_HH

#include <algorithm>
#include <armadillo>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <math.h>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <netflix.hh>
#include <basealgorithm.hh>

#define EPSILON 0.0000000001

// Idea of optimization, speed up is from:
// http://dmnewbie.blogspot.com/2009/06/calculating-316-million-movie.html
using namespace arma;
using namespace netflix; // challenge-related constants/functions.

struct mu_pair
{
    unsigned int user;
    float rating;
};

struct um_pair
{
    unsigned short movie;
    float rating;
};

// Pearson intermediates, as described in dmnewbie's blog
struct s_inter 
{
    float x;  // sum of ratings of movie i
    float y;  // sum of ratings of movie j
    float xy; // sum (rating_i * rating_j)
    float xx; // sum (rating_i^2)
    float yy; // sum (rating_j^2)
    unsigned int n; // Num users who rated both movies
};

// To be stored in P
struct s_pear
{
    float p;
    unsigned int common;
};

// Used during prediction
// As per the blogpost
struct s_neighbors
{
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

class KNN : public BaseAlgorithm
{
    private:
        const int numUsers;
        const int numItems;

        // Whether or not to load P from file (i.e. from pFileName). If
        // false, we'll compute P instead.
        bool loadPFromFile;

        // Whether we want to save P to file after computing it.
        bool savePToFile;

        // Minimum common neighbors required for decent prediction.
        const int minCommon;

        // Max weight elements to consider when predicting.
        const unsigned int maxWeight;

        const std::string &pFilename;

        // um: for every user, stores (movie, rating) pairs.
        std::vector<std::vector<um_pair>> um;

        // mu: for every movie, stores (user, rating) pairs.
        std::vector<std::vector<mu_pair>> mu;

        // Pearson coefficients for every movie pair
        // When accessing P[i][j], it must always be the case that:
        // i <= j (symmetry is assumed)
        std::vector<std::vector<s_pear>> P;
        std::vector<float> movieAvg;

    public:
        KNN(const int numUsers, const int numItems, const int minCommon,
            const unsigned int maxWeight, bool loadPFromFile,
            bool savePToFile, const std::string &pFilename);
        void train(const fmat &data);
        float predict(int user, int item, int date, bool bound);
        void calcP();
        void saveP();
        void loadP();
        ~KNN();
};

#endif // KNN_HH
