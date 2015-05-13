#include <rbm_new.hh>

RBM_New::RBM_New(int numUsers, int numItems, float globalAverage,
    int maxRating, int numFactors, float learningRate, int numIters) :
    numUsers(numUsers), numItems(numItems), globalAverage(globalAverage),
    maxRating(maxRating), numFactors(numFactors), learningRate(learningRate),
    userStartIndex(numUsers), numItemsTrainingSet(numUsers), numIters(numIters)
{
    W = randu<cube>(maxRating, numFactors, numUsers) / 8.0;
    BV = randu<mat>(maxRating, numUsers) / 8.0;
    BH = randu<vec>(numFactors) / 8.0;
    // BH = randu<mat>(K, F) / 8.0;
    CD_K = 1;
    userStartIndex.zeros();
    numItemsTrainingSet.zeros();
}


float RBM_New::sigma(float num)
{
    return 1.0 / (1 + exp(-num));
}


void RBM_New::populateNumItemsTrainingSet()
{
    cout << "Begin populating numItems..." << endl;
    for(unsigned int i = 0; i < dataUM.n_cols; i++)
    {
        // Based on the user that this rating was by, increment the
        // appropriate element of numItemsTrainingSet.
        int user = roundToInt(dataUM(USER_ROW, i));
        if (numItemsTrainingSet[user] == 0)
            userStartIndex.at(user) = i;
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
    H0(j) = sigma( BH(j) + sum_ik ( W(k, j, r.movie) * V0(k, i) ))
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


RBM_New::~RBM_New()
{
    // Nothing to destroy here.
}
