#include <rbm_new.hh>

RBM_New::RBM_New(int numUsers, int numItems, float globalAverage,
    int maxRating, int numFactors, float learningRate, int numIters) :
    numUsers(numUsers), numItems(numItems), globalAverage(globalAverage),
    maxRating(maxRating), numFactors(numFactors), learningRate(learningRate),
    userStartIndex(numUsers), numItemsTrainingSet(numUsers), numIters(numIters),
    numUsersTrainingSet(numItems)
{
    W = randu<cube>(maxRating, numFactors, numUsers) / 8.0;
    BV = randu<mat>(maxRating, numUsers) / 8.0;
    BH = randu<vec>(numFactors) / 8.0;
    // BH = randu<mat>(maxRating, numFactors) / 8.0;
    CD_K = 1;
    userStartIndex.zeros();
    numItemsTrainingSet.zeros();
    numUsersTrainingSet.zeros();

    /* new stuff */
    rbm_alpha        = 0.001;
    rbm_beta         = 0.008;
    rbm_bins         = 6;
    rbm_scaling      = 1.0;
    rbm_mult_step_dec= 0.999;
    D = 100;
    /* end of new stuff */
}


float RBM_New::sigma(float num)
{
    return 1.0 / (1 + exp(-num));
}


void RBM_New::populateNumItemsTrainingSet()
{
    cout << "Begin populating numItems..." << endl;
    int userId = -1;
    for(unsigned int i = 0; i < dataUM.n_cols; i++)
    {
        // Based on the user that this rating was by, increment the
        // appropriate element of numItemsTrainingSet.
        int user = roundToInt(dataUM(USER_ROW, i));
        if (userId != user)
        {
            userStartIndex.at(user) = i;
            userId = user;
        }
        numItemsTrainingSet(user) ++;
    }
    cout << "Finished populating numItems..." << endl;
}


float RBM_New::predict(int user, int movie, int date, bool bound)
{
    // predicting stage
    int startIdx = userStartIndex[user];
    int user_size = numItemsTrainingSet[user];
    float predictedRating;
    if (user_size == 0)
        return globalAverage;

    vec Hu = zeros<vec>(numFactors);
    vec Vum(maxRating);
    ivec scores = linspace<ivec>(1, 5, 5);

    Hu = BH;

    for (int f = 0; f < numFactors; f++)
    {
        for (int u = startIdx; u < startIdx + user_size; u++)
        {
            int movieId = roundToInt(dataUM(MOVIE_ROW, u));
            int rating = roundToInt(dataUM(RATING_ROW, u));
            unsigned int k = rating - 1;
            float w = W(k, f, movieId);
            Hu(f) += w;
        }
    }

    Hu = 1.0 / (1 + exp(-Hu));
    date = date;
    bound = bound;
    // Negative phase to predict score
    Vum = normalise( exp(BV.col(movie) + W.slice(movie) * Hu), 1);
    predictedRating = dot(Vum, scores);
    if (bound)
    {
        if (predictedRating > 5)
            predictedRating = maxRating;
        else if (predictedRating < 1)
            predictedRating = MIN_RATING;
    }
    return predictedRating;
}


void RBM_New::singleUser(int user_id, int CD_K)
{
    int size = numItemsTrainingSet[user_id];
    int startIdx = userStartIndex[user_id];

    // Initialization
    mat V0 = zeros<mat>(maxRating, size);
    mat Vt = zeros<mat>(maxRating, size);
    vec H0 = zeros<vec>(numFactors);
    vec Ht = zeros<vec>(numFactors);

    // Set up V0 and Vt based on the input data.
    for (int i = 0; i < size; i++)
    {
        float score = dataUM(RATING_ROW, startIdx + i);
        V0(int(score)-1, i) = 1; // score - 1 is the index
        Vt(int(score)-1, i) = 1;
    }

    /*
    // Set up H0 by V -> H
    H0(j) = sigma( BH(j) + sum_ik ( W(k, j, movie) * V0(k, i) ))
    */

    H0 = BH;
    for (int i = 0; i < size; i++)
    {
        int curr_movie = roundToInt(dataUM(MOVIE_ROW, startIdx + i));
        H0 += W.slice(curr_movie).t() * V0.col(i);
    }
    H0 = 1.0 / (1 + exp(-H0));

    // Do the contrastive divergence
    for (int n = 0; n < CD_K; n++)
    {
        // Positive phase: V -> H
        Ht = BH;
        for (int i = 0; i < size; i ++) {
            int curr_movie = roundToInt(dataUM(MOVIE_ROW, startIdx + i));
            Ht += W.slice(curr_movie).t() * Vt.col(i);
        }
        Ht = 1.0 / (1 + exp(-Ht));

        // Negative phase: H -> V
        for (int i = 0; i < size; i++) {
            int curr_movie = roundToInt(dataUM(MOVIE_ROW, startIdx + i));
            Vt.col(i) = exp(BV.col(curr_movie) + W.slice(curr_movie) * Ht);
        }

        // Normalize Vt -> sum_k(Vt(k, i)) = 1
        Vt = normalise(Vt, 1);
    }

    // Update W
    for (int i = 0; i < size; i++)
    {
        int curr_movie = roundToInt(dataUM(MOVIE_ROW, startIdx + i));
        W.slice(curr_movie) += learningRate * (V0.col(i) * H0.t()
            - Vt.col(i) * Ht.t());
    }

    // Update BH
    BH += learningRate * (H0 - Ht);

    // Update BV
    for (int i = 0; i < size; i++)
    {
        int curr_movie = roundToInt(dataUM(MOVIE_ROW, startIdx + i));
        BV.col(curr_movie) += learningRate * (V0.col(i) - Vt.col(i));
    }
}


void RBM_New::train(const fmat &data)
{
    dataUM = data;
    populateNumItemsTrainingSet();

    // Training stage
    for (int iter_num = 0; iter_num < numIters; iter_num++)
    {
        // Customize CD_K based on the number of iteration
        if (iter_num < 15)
            CD_K = 1;
        else if (iter_num < 25)
            CD_K = 3;
        else if (iter_num < 35)
            CD_K = 5;
        else
            CD_K = 9;

        cout << "\n== Iteration " << iter_num << " ==" << endl;

        for (unsigned int i = 0; i < dataUM.n_cols; i++)
        {
            int user_id = roundToInt(dataUM(USER_ROW, i));
            singleUser(user_id, CD_K);
            if (i % 100000 == 0)
                cout << "Processed data: " << i << endl;
        }
    }
    cout << "\nFinished training!" << endl;
    cout << "Training data size: " << dataUM.n_cols << endl;
}

/* BEGIN NEW STUFF */

void RBM_New::setRand2(int movieId, float c)
{
    for(unsigned int i = 0; i < movie_data[movieId].w.size(); i++)
        movie_data[movieId].w[i] = ((drand48() - 0.5) * c);
}


float RBM_New::newer_dot(vec a, int as, vec b, int bs)
{
    float ret = 0;
    for(int i = 0; i < D; i++)
        ret += (double)a[as + i] * (double)b[bs + i];
    //std::cout << ret << std::endl;
    return ret;
}


float RBM_New::rbm_predict(const rbm_user &usr, 
    const rbm_movie &mov, const float rating, float &prediction)
{
    float ret = 0;
    float nn = 0;
    //if (mov.bi.size() == 0) return globalAverage;

    for(int r = 0; r < rbm_bins; ++r)
    {               
        float zz = exp(mov.bi[r] + newer_dot(usr.h, 0, mov.w, r * D));
        //cout << zz << endl;
        /*if (std::isinf(zz))
        {
            std::cout << " mov.bi[r] " << mov.bi[r] << " dot: " << newer_dot(usr.h, 0, mov.w, r * D) << std::endl;
        }*/
        ret += zz * (float)(r);
        //assert(!std::isnan(ret));
        nn += zz;
    }
    //std::cout << nn << std::endl;
    //assert(!std::isnan(ret));
    //assert(std::fabs(nn) > 1e-32);
    ret /= nn;
    if(ret < MIN_RATING) ret = MIN_RATING;
    else if(ret > maxRating) ret = maxRating;
    //std::cout << nn << " " << ret << " " << (ret != ret) << std:: endl;
    //assert(!std::isnan(ret));
    prediction = ret * rbm_scaling;
    //assert(!std::isnan(prediction));

    return pow(prediction - rating, 2);
}

void RBM_New::rbm_init()
{
    srand48(time(NULL));
    //latent_factors_inmem.resize(numUsers + numItems);
    user_data.resize(numUsers);
    movie_data.resize(numItems);
/*
#pragma omp parallel for
  for(int i = 0; i < (int)numItems; ++i)
  {
    vertex_data & movie = latent_factors_inmem[numUsers + i];
    movie.pvec = zeros(rbm_bins + D * rbm_bins);
    movie.bias = 0;
  }
*/
#pragma omp parallel for
  for(int i = 0; i < (int)numItems; ++i)
  {
    movie_data[i] = rbm_movie(rbm_bins, D);
  }
  for (int i = 0; i < (int)numUsers; ++i)
  {
    user_data[i] = rbm_user(D);
  }
}

float RBM_New::predict1(const rbm_user & usr, 
    const rbm_movie & mov, 
    const float rating, 
    float & prediction){

    vec zz = zeros(rbm_bins);
    float szz = 0;
    for(int r = 0; r < rbm_bins; ++r)
    {
        zz[r] = exp(mov.bi[r] + newer_dot(usr.h0, 0, mov.w, r * D));
        szz += zz[r];
    }
    float rd = drand48() * szz;
    szz = 0;
    int ret = 0;
    for(int r = 0; r < rbm_bins; ++r)
    {
        szz += zz[r];
        if(rd < szz)
        { 
            ret = r;
            break;
        }
    }
    prediction = ret * rbm_scaling;
    assert(!std::isnan(prediction));
    //std::cout << pow(prediction - rating, 2) << std::endl;
    return pow(prediction - rating, 2);
}

inline float RBM_New::sigmoid(float x)
{
    return 1 / (1 + exp(-1 * x));
}

void RBM_New::update(int currIter)
{
    if (currIter == 0)
    {
        // Go over users.
        for(int u = 0; u < numUsers; u++)
        {
            int startIdx = userStartIndex[u];
            int userId = roundToInt(dataUM(USER_ROW, startIdx));
            cout << userId << " " << u << " " << numItemsTrainingSet[u] << endl;
            assert(userId == u || (numItemsTrainingSet[u] == 0));
            //std::cout << u << " " << userId << std::endl;
            int currSize = numItemsTrainingSet[userId];
            for(int e = 0; e < currSize; e++)
            {
                //cout << u << " " << e << endl;
                int movieId = roundToInt(dataUM(MOVIE_ROW, startIdx + e));
                numUsersTrainingSet(movieId)++;
                //rbm_movie mov = latent_factors_inmem[idx];
                float observation = (dataUM(RATING_ROW, startIdx + e));
                int r = (int)(observation / rbm_scaling);
                assert(r < rbm_bins);
                /*
                if (movie_data[movieId].bi.size() == 0)
                    movie_data[movieId] = rbm_movie(rbm_bins, D);
                */
                //cout << movie_data[movieId].bi[r] << " ";
                movie_data[movieId].bi[r]++;
                //cout << movie_data[movieId].bi[r] << endl;
                //std::cout << latent_factors_inmem[idx].pvec[r] << std::endl;
                //latent_factors_inmem[idx].pvec = mov.bi;
            }
        }
        return;
    }
    else if (currIter == 1)
    {
        // Go over movies.
        for (int m = 0; m < numItems; m++)
        {
            int currSize = numUsersTrainingSet[m];
            if (currSize == 0) continue;
            //std::cout << movie_data[m].w[0] << " " << movie_data[m].w[1] << " " << movie_data[m].w[2] << std::endl;
            setRand2(m, 0.001);
            //std::cout << movie_data[m].w[0] << " " << movie_data[m].w[1] << " " << movie_data[m].w[2] << std::endl;
            for(int r = 0; r < rbm_bins; ++r)
            {
                movie_data[m].bi[r] /= (float)currSize;
                movie_data[m].bi[r] = log(1E-9 + movie_data[m].bi[r]);
                //cout << movie_data[m].bi[r] << endl;
                if (movie_data[m].bi[r] > 1000)
                {
                    std::cout << "Num. overflow!" << std::endl;
                    exit(1);
                }
            }
            //if (m == 0)
            //    std::cout << movie_data[m].bi[0] << std::endl;
        }
        return; //done with initialization
    }
    // go over all user nodes
    for(int u = 0; u < numUsers; u++)
    {
        user_data[u].h.zeros();
        user_data[u].h0.zeros();
        user_data[u].h1.zeros();
        //std::cout << user_data[u].h[0] << " " << user_data[u].h[0] << " " << user_data[u].h[0] << " ";
      if (u % 100000 == 0) std::cout << "At user: " << u << std::endl;
      int startIdx = userStartIndex[u];
      int movieRated = numItemsTrainingSet[u];
      vec v1 = zeros(movieRated); 
      //go over all ratings
      for(int e = 0; e < movieRated; e++)
      {
        //std::cout << user_data[u].h[0] << " " << user_data[u].h[0] << " " << user_data[u].h[0] << " " << std::endl;
      
        float observation = dataUM(RATING_ROW, startIdx + e);
        int movieId = roundToInt(dataUM(MOVIE_ROW, startIdx + e));     
        int r = (int)(observation / rbm_scaling);
        assert(r < rbm_bins);  
        for(int k = 0; k < D; k++)
        {
            //std::cout << user_data[u].h[k] << " ";
          user_data[u].h[k] += movie_data[movieId].w[D * r + k];
          //std::cout << user_data[u].h[k] << std::endl;
          assert(!std::isnan(user_data[u].h[k]));
        }
      }

      for(int k=0; k < D; k++)
      {
        user_data[u].h[k] = sigmoid(user_data[u].h[k]);
        if (drand48() < user_data[u].h[k]) 
          user_data[u].h0[k] = 1;
        else user_data[u].h0[k] = 0;
      }

      int i = 0;
      float prediction;
      for(int e = 0; e < movieRated; e++)
      {
        float observation = dataUM(RATING_ROW, startIdx + e);
        int movieId = roundToInt(dataUM(MOVIE_ROW, startIdx + e));     

        predict1(user_data[u], movie_data[movieId], observation, prediction);    
        int vi = (int)(prediction / rbm_scaling);
        v1[i] = vi;
        i++;
      }

      i = 0;
      for(int e=0; e < movieRated; e++) {
        int movieId = roundToInt(dataUM(MOVIE_ROW, startIdx + e));     
        int r = (int)v1[i];
        for (int k=0; k< D;k++)
        {
          user_data[u].h1[k] += movie_data[movieId].w[r * D + k];
        }
        i++;
      }

      for (int k=0; k < D; k++){
        user_data[u].h1[k] = sigmoid(user_data[u].h1[k]);
        if (drand48() < user_data[u].h1[k]) 
          user_data[u].h1[k] = 1;
        else user_data[u].h1[k] = 0;
      }

      i = 0;
      for(int e=0; e < movieRated; e++)
      {
        int movieId = roundToInt(dataUM(MOVIE_ROW, startIdx + e));
        float observation = dataUM(RATING_ROW, startIdx + e);
        float prediction;
        rbm_predict(user_data[u], movie_data[movieId], observation, prediction);
        //float pui = prediction / rbm_scaling;
        float rui = observation / rbm_scaling;
        //rmse_vec[omp_get_thread_num()] += (pui - rui) * (pui - rui);
        //nn += 1.0;
        int vi0 = (int)(rui);
        int vi1 = (int)v1[i];
        for (int k = 0; k < D; k++)
        {
          movie_data[movieId].w[D*vi0+k] += rbm_alpha * (user_data[u].h0[k] - rbm_beta * movie_data[movieId].w[vi0*D+k]);
          assert(!std::isnan(movie_data[movieId].w[D*vi0+k]));
          movie_data[movieId].w[D*vi1+k] -= rbm_alpha * (user_data[u].h1[k] + rbm_beta * movie_data[movieId].w[vi1*D+k]);
          assert(!std::isnan(movie_data[movieId].w[D*vi1+k]));
        }
        i++; 
      }
    }
    rbm_alpha *= rbm_mult_step_dec;
}


float RBM_New::new_predict(int user, int movie, float rating)
{
    float prediction;
    /*
    if (rating < 1)
      rating = globalAverage;
    */
    rbm_predict(user_data[user], movie_data[movie], 0, prediction);

    return (float)prediction;
}


void RBM_New::new_train(const fmat &data)
{
    dataUM = data;
    populateNumItemsTrainingSet();
    rbm_init();
}

/* END NEW STUFF */

RBM_New::~RBM_New()
{
    // Nothing to destroy here.
}
