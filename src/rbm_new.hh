#ifndef __RBM_NEW_HH__
#define __RBM_NEW_HH__

#include <netflix.hh>
#include <basealgorithm.hh>
#include <armadillo>
#include <iostream>

using namespace arma;
using namespace netflix;

class RBM_New : public BaseAlgorithm
{
    private:
        fmat dataUM;
        int numItems;
        int numUsers;
        int maxRating;
        int numFactors;
        int numIters;
        unsigned int CD_K;
        float globalAverage;
        float learningRate;

        // numUsers * numFactors * maxRating
        cube W;
        // maxRating * numUsers
        mat BV;
        // numFactors
        vec BH;

        fcolvec numItemsTrainingSet;
        fcolvec userStartIndex;

        float sigma(float num);
        void populateNumItemsTrainingSet();
        void singleUser(int user_id, int CD_K);

    public:
        RBM_New(int numUsers, int numItems, float globalAverage,
            int maxRating, int numFactors, float learningRate,
            int numIters);
        void train(const fmat &data);
        float predict(int user, int item, int date, bool bound);
        ~RBM_New();
};

#endif // __RBM_NEW_HH__

