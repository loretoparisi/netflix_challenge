#ifndef SINGLEALGORITHM_HH
#define SINGLEALGORITHM_HH

#include <armadillo>

using namespace arma;

class SingleAlgorithm 
{
public:
    /**
     * Note: data is assumed to be in column-major order. So, typically,
     * its shape will be 4 x NUM_TRAINING_PTS, where "4" is the number of
     * attributes in a "rating" (i.e. user, item, date, rating).
     */
    virtual void train(const fmat &data) = 0;

    /**
     * This function also trains, but it works with file names specifying
     * the desired dataset. These file names are where we've stored "data"
     * -- the 4 x NUM_TRAINING_PTS array mentioned above. The data must be
     * stored in Armadillo's binary format.
     *
     */
    virtual void train(const std::string &fileNameData) = 0;

    /**
     * Note: Some algorithms do not use the date aspect, but this has been
     * added for consistency across all SingleAlgorithms.
     */
    virtual float predict(int user, int item, int date) = 0;
    
    virtual ~SingleAlgorithm() {}
};

#endif // SINGLEALGORITHM_HH
