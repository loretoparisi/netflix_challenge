#ifndef __RBM_NEW_HH__
#define __RBM_NEW_HH__

#include <netflix.hh>
#include <basealgorithm.hh>
#include <armadillo>
#include <iostream>
#include <assert.h>
#include <omp.h>

using namespace arma;
using namespace netflix;

/*
 * h = pvec = D * float
 * h0 = weight = D * float
 * h1 = weight+D = D * float
 */
struct rbm_user
{
  vec h;
  vec h0;
  vec h1;
  rbm_user()
  {

  }
  rbm_user(int D)
  {
    h.resize(D);
    h0.resize(D);
    h1.resize(D);
    h.zeros();
    h0.zeros();
    h1.zeros();
  }
  /*
    h = (float*)malloc(D * sizeof(float));
    h0 = (float*)malloc(D * sizeof(float));
    h1 = (float*)malloc(D * sizeof(float));
  }
  rbm_user(const vertex_data & vdata)
  {
    h = (float*)&vdata.pvec[0];
    h0 = h + 100;
    h1 = h0 + 100;
  }

  rbm_user & operator=(vertex_data & data)
  {
    h = (float*)&data.pvec[0];
    h0 = h + 100;
    h1 = h0 + 100;
    return * this;
  }
  */
};


/**
 * bi = rbm_bins * float 
 *  w = weight = rbm_bins * D * float
 */
struct rbm_movie{
  vec bi;
  vec w;
  rbm_movie()
  {

  }
  rbm_movie(int bins, int D)
  {
    bi.resize(bins);
    w.resize(D * bins);
    bi.zeros();
    w.zeros();
    /*
    for (unsigned i = 0; i < bi.size(); i++)
        bi[i] = 0;
    for(unsigned int i = 0; i < w.size(); i++)
        w[i] = 0;
    */
  }
  /*
  float * bi;
  float * w;


  rbm_movie(const vertex_data& vdata){
    bi = (float*)&vdata.pvec[0];
    w = bi + 6;
  }

  rbm_movie & operator=(vertex_data & data){
    bi = (float*)&data.pvec[0];
    w = bi + 6;
    return * this;
  }
  */
};

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

        /* New stuff here! */
        fcolvec numUsersTrainingSet;
        double rbm_alpha;
        double rbm_beta;
        int    rbm_bins;
        double rbm_scaling;
        double rbm_mult_step_dec;
        int D;
        //std::vector<vertex_data> latent_factors_inmem;
        std::vector<rbm_user> user_data;
        std::vector<rbm_movie> movie_data;

        void setRand2(int movieId, float c);
        float newer_dot(vec a, int as, vec b, int bs);

        float rbm_predict(const rbm_user &usr, const rbm_movie &mov, 
            const float rating, float & prediction);
        float predict1(const rbm_user & usr, 
            const rbm_movie & mov, 
            const float rating, 
            float & prediction);
        inline float sigmoid(float x);
        void rbm_init();
        /* End of new stuff */

    public:
        RBM_New(int numUsers, int numItems, float globalAverage,
            int maxRating, int numFactors, float learningRate,
            int numIters);
        void train(const fmat &data);
        float predict(int user, int item, int date, bool bound);
        float new_predict(int user, int movie, float rating);
        void new_train(const fmat &data);
        void update(int currIter);
        ~RBM_New();
};

#endif // __RBM_NEW_HH__

