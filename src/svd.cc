#ifndef SVD_CC_
#define SVD_CC_

#include "svd.hh"

namespace svd
{
    // use some global variables to store the parameter bu, bi, p, q
    float bu[USER_NUM] = {0};  // the user bias in the baseline predictor
    float bi[ITEM_NUM] = {0};  // the item bias in the baseline predictor

    int buNum[USER_NUM] = {0};  //ratings num of every user
    int biNum[ITEM_NUM] = {0};  //rating num of every item

    float p[USER_NUM][K_NUM] = {0};   //user character Matrix
    float q[ITEM_NUM][K_NUM] = {0};   //item character Matrix

    vector < vector<rateNode> > rateMatrix(USER_NUM);  // store training set
    vector<testSetNode> probeRow;                      // store test set

    // initialize the bias bu and bi
    // this is from the method in the page 2 of koren's TKDD'09 paper
    void initialBias()
    {
        using namespace svd;
        int i, j;
        for(i = 0; i < USER_NUM; i++)
        {
            int vSize = rateMatrix[i].size();
            for(j = 0; j < vSize; j++) {
                bi[rateMatrix[i][j].item] += \
                    (rateMatrix[i][j].rate - netflix::MEAN_RATING_TRAINING_SET);
                biNum[rateMatrix[i][j].item] += 1;
            }
        }

        for(i = 0; i < ITEM_NUM; i++)
        {
            if(biNum[i] >= 1) bi[i] = bi[i] / (biNum[i] + 25);
            else bi[i] = 0.0;
        }

        for(i = 0; i < USER_NUM; i++)
        {
            int vSize = rateMatrix[i].size();
            for(j = 0; j < vSize; j++)
            {
                bu[i] += (rateMatrix[i][j].rate - \
                    netflix::MEAN_RATING_TRAINING_SET - bi[rateMatrix[i][j].item]);
                buNum[i] += 1;
            }
        }

        for(i = 0; i < USER_NUM; i++)
        {
            if(buNum[i]>= 1)bu[i] = bu[i] / (buNum[i] + 10);
            else bu[i] = 0.0;
        }
    }
    
    // intialize the matrix of user character(P)
    // and the matrix of item character(Q)
    void initialPQ(int itemNum, int userNum, int dim)
    {
        using namespace svd;
        for(int i = 0; i < itemNum; i++)
        {
            setRand(q[i], dim, 0);   
        }

        for(int i = 0; i < userNum; i++)
        {
            setRand(p[i], dim, 0);   
        }
    }

    void model(int dim, float alpha1, float alpha2, float beta1, float beta2, 
               int maxStep = 60, float slowRate = 1)
    {
        cout << "begin initialization: " << endl;
        loadRating(TRAINING_SET, rateMatrix, RATE_SP);  // load training set
        loadProbe(PROBE_SET, probeRow, RATE_SP);   // load test set
        int i, u, k;
        
        srand((unsigned)time(0)); 
        //initialBias(); //initialize bu and bi
        
        initialPQ(ITEM_NUM, USER_NUM, K_NUM); //intialize the matrix of user
        // character(P) and the matrix of item character(Q) 
        cout <<"initialization end!"<<endl<< "begin iteration: " << endl;
        
        float pui = 0.0 ; // the predict value of user u to item i
        float preRmse = 1000000000000.0;
        // used to record the previous rmse of test set and make as the
        // terminal condition if the rmse of test begin to increase, then break
        float nowRmse = 0.0;
        cout <<"begin testRMSEProbe(): " << endl;
        RMSEProbe(probeRow, K_NUM);
        // main loop
        for(int step = 0; step < maxStep; ++step)
        {  // only iterate maxStep times
            long double rmse = 0.0;
            int n = 0;
            for( u = 0; u < USER_NUM; ++u) {   // process every user
                int RuNum = rateMatrix[u].size(); // the num of items rated by user
                   
                for(i = 0; i < RuNum; i++)
                { // process every item rated by user
                    int itemI = rateMatrix[u][i].item;
                    short rui = rateMatrix[u][i].rate; // real rate
                    // pui = predictRate(u, itemI, netflix::MEAN_RATING_TRAINING_SET, bu, bi, p[u], q[itemI], dim);
                    pui = predictRate(u, itemI, dim);
                    
                    float eui = rui - pui;
                    
                    if( isnan(eui) )
                    { // fabs(eui) >= 4.2 || 
                        cout << u << '\t' << i << '\t' << pui <<'\t';
                        cout << rui<< '\t' << bu[u] << '\t';
                        cout << bi[itemI] << '\t' <<  netflix::MEAN_RATING_TRAINING_SET << endl;
                        // printArray(q[itemI], p[u], K_NUM+1);
                        exit(1);
                    }
                    rmse += eui * eui; ++n;
                    if(n % 10000000 == 0)
                    {
                        cout << "step: " << step << " n: " << n;
                        cout << " dealed!" << endl;
                    }
                    
                    // Update bias.
                    bu[u] += alpha1 * (eui - beta1 * bu[u]);
                    bi[itemI] += alpha1 * (eui - beta1 * bi[itemI]);
                    
                    for( k = 0; k< K_NUM; ++k)
                    {
                        // float tempPu = p[u][k];
                        p[u][k] += alpha2 * (eui*q[itemI][k] - beta2*p[u][k]);
                        q[itemI][k] += alpha2 * (eui*p[u][k] - beta2*q[itemI][k]);
                    }
                } 
            }
            nowRmse = sqrt( rmse / n);
            //if the rmse of test set begin to increase, then break
            if( nowRmse >= preRmse && step >= 3) break;
            else
                preRmse = nowRmse;
            RMSEProbe(probeRow, K_NUM);  // check rmse of test set 
            // gradually reduce the learning rate
            alpha1 *= slowRate;    
            alpha2 *= slowRate;
        }
        RMSEProbe(probeRow, K_NUM);  // check rmse of test set 
        // OUTPUT
        string line;
        char c_line[20];
        int userId;
        int movieId;
        float rating;
        stringstream OFname;

        cout << "Generating output" << endl;

        OFname << OUT_FILE;

        ifstream qual (QUAL_SET);
        ofstream outputFile (OFname.str().c_str(), ios::trunc); 
        if (qual.fail() || outputFile.fail()) {
            cout << "qual.dta: Open failed.\n";
            exit(-1);
        }
        while (getline(qual, line)) {
            memcpy(c_line, line.c_str(), 20);
            userId = atoi(strtok(c_line, " "));
            movieId = (short) atoi(strtok(NULL, " "));
            rating = predictRate(userId, movieId, K_NUM);
            outputFile << rating << '\n';
        }

        cout << "Output generated" << endl;
        // END OUTPUT
        return;
    }
};

/**
 * predict the rate
 */
float predictRate(int user, int item, int dim)
{
    using namespace svd;
    int RuNum = rateMatrix[user].size(); // the num of items rated by user
    float ret; 
    if(RuNum > 1)
        ret = netflix::MEAN_RATING_TRAINING_SET + bu[user] + bi[item] +  new_dot(p[user], q[item], dim);
    else
        ret = netflix::MEAN_RATING_TRAINING_SET + bu[user] + bi[item];

    if(ret < 1.0) ret = 1;
    if(ret > 5.0) ret = 5;

    return ret;
}

#endif // SVD_CC_ 
