/*
 * This class allows us to carry out Global Effect on the Netflix
 * dataset.
 */

#ifndef GLOBALS_HH
#define GLOBALS_HH

#include <algorithm>
#include <armadillo>
#include <array>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <netflix.hh>
#include <singlealgorithm.hh>

#define LEVEL1_ALPHA  25
#define LEVEL2_ALPHA  7
#define LEVEL3_ALPHA  550
#define LEVEL4_ALPHA  150
#define LEVEL5_ALPHA 4000
#define LEVEL6_ALPHA  500
#define LEVEL7_ALPHA  90
#define LEVEL8_ALPHA  90
#define LEVEL9_ALPHA  50
#define LEVEL10_ALPHA 50

using namespace arma;
using namespace netflix; // challenge-related constants/functions.

using std::cout;
using std::endl;

// This can technically be a subclass of SVDPP, but there's no real point
// in nesting the hierarchy that much. Especially since SVDPP's internals
// represent fairly different things.
class Globals : public SingleAlgorithm
{
private:
    fmat dataMU;
    int numUsers, numItems, level;
    float globalAverage;
     // The average of sqrt(Num of train data for a movie)
    float sqrtmoviecountaverage;
    float sqrtUserTimeUserAverage;
    float sqrtUserTimeMovieAverage;
    float sqrtMovieTimeMovieAverage;
    float sqrtMovieTimeUserAverage;
    std::vector<float> movieAverages;
    std::vector<float> userAverages;
    std::vector<float> movieVariances;
    std::vector<float> userVariances;
    std::vector<float> movieThetas;
    std::vector<float> userThetas;
    std::vector<float> monthThetas;
    std::vector<float> quarterThetas;
    std::vector<float> userTimeUserThetas;
    std::vector<float> userTimeMovieThetas;
    std::vector<float> movieTimeMovieThetas;
    std::vector<float> movieTimeUserThetas;
    std::vector<float> userMovieAverageThetas;
    std::vector<float> userMovieSupportThetas;
    std::vector<float> movieUserAverageThetas;
    std::vector<float> movieUserSupportThetas;
    std::vector<int> userFirstDates;
    std::vector<int> movieFirstDates;
    std::vector<int> userLastDates;
    std::vector<int> movieLastDates;
    float userAverageDates[NUM_USERS];
    float movieAverageDates[NUM_MOVIES];
    uint* user_first_dates;
    uint* user_last_dates;
    uint* movie_first_dates;
    uint* movie_last_dates;

    fcolvec numItemsTrainingSet;
    fcolvec numUsersTrainingSet;

    void initInternalData();
    void populateNumItemsTrainingSet(const fmat &data);
    void populateNumUsersTrainingSet(const fmat &data);
    bool setAverages(const fmat &dataUM);
    void setVariances(const fmat &dataUM);
    bool setThetas(const fmat &dataUM);


public:
    Globals(int numUsers, int numItems, int levels,
        const std::string &trainFileName);

    ~Globals();
    
    void train(const fmat &data);
    float predict(int user, int item, int date);
};

#endif // GLOBALS_HH
