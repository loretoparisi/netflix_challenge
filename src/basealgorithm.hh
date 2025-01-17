#ifndef __BASEALGORITHM_HH__
#define __BASEALGORITHM_HH__

#include <armadillo>

using namespace arma;

typedef float data_t;

class BaseAlgorithm 
{
public:
    /**
     * Note: data is assumed to be in column-major order. So, typically,
     * its shape will be 4 x NUM_TRAINING_PTS, where "4" is the number of
     * attributes in a "rating" (i.e. user, item, date, rating).
     */
    virtual void train (const Mat<data_t> &data) = 0;

    /**
     * This function also trains, but it works with file names specifying
     * the desired dataset. These file names are where we've stored "data"
     * -- the 4 x NUM_TRAINING_PTS array mentioned above. The data must be
     * stored in Armadillo's binary format.
     *
     * @param dataPath: The file where "data" is stored. This binary file
     *                  must hold matrix data in the format specified in
     *                  the train(const fmat &data) function.
     *
     */
    virtual void train (const std::string &dataPath) {
        Mat<data_t> data;
        data.load(dataPath);
        this->train(data);
    }

    /**
     * Note: Some algorithms do not use the date aspect, but this has been
     * added for consistency across all BaseAlgorithms.
     */
    virtual float predict(int user, int item, int date, bool bound) = 0;

    virtual ~BaseAlgorithm() {}
};

#endif // __BASEALGORITHM_HH__
