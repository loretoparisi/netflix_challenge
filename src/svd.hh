#ifndef SVD_HH_
#define SVD_HH_

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <ctime>
#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <malloc.h>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

struct rateNode
{
    short item;
    short rate;
};

// use different struct between test set
// and training set to save memory space
struct testSetNode
{
    int user;
    short item;
    short rate;
};

float new_dot(float* p, float* qLocal, int dim)
{
    float result = 0.0;
    for (int i = 0; i < dim; i++)
    {
        result += p[i] * qLocal[i];
    }
    return result;
}

float get_rand(int dim)
{
    return 0.1 * (rand()/(float)RAND_MAX) / sqrt(dim);
}

// set the vector to random values
void setRand(float  p[], int dim, float base)
{
    for(int i = 0; i < dim; i++)
    {
        float temp = base + get_rand(dim);
        p[i] = temp;
    }
}

float predictRate(int user, int item, int dim);

// compute the rmse of test set
float RMSEProbe(vector<testSetNode>& probeSet, int dim)
{
    int probeSize = probeSet.size();
    float pRate, err;
    long double rmse = 0;

    for(int i = 0; i < probeSize; i++)
    {
        pRate = predictRate(probeSet[i].user, \
            probeSet[i].item, dim); // predict rate
        err = pRate-probeSet[i].rate;
        rmse += err*err;
    }
    rmse = sqrt( rmse / probeSize);
    cout<<"RMSE: "<< rmse <<" probeSize: "<< probeSize << endl;
    return rmse;
}

// load training set
void loadRating(char * fileName, vector< vector<rateNode> >& rateMatrixLocal, \
	const char* separator)
{
    char rateStr[256];
    char* pch;    
    vector<string> rateDetail;
    std::ifstream from (fileName);
    if (!from.is_open())
    {
        cout << "can't open: operation failed!\n";
        exit(1);
    }

    int itemId = -1, userId = -1, rate = 0;
    while(from.getline(rateStr, 256))
    {
        string strTemp(rateStr);
        if(strTemp.length() < 3) continue;

        int i = 0;
        pch = strtok (rateStr, separator);
        while (pch != NULL)
        {
            if(0 == i) userId = atoi(pch);
            else if(1 == i) itemId = atoi(pch);
            else if(3 == i) rate = atoi(pch);
            else if(i > 4) break;
            ++i;
            pch = strtok (NULL, separator);
        }
        if(-1 == itemId || -1 == userId || 0 == rate )
        {
            cout << strTemp << "#########userId: " << userId;
            cout <<" itemId: "<< itemId <<" rate: "<< rate << endl;
            exit(1);
        }

        // initialization rateMatrix
        try
        {
            rateNode tmpNode;
            tmpNode.item = itemId;
            tmpNode.rate = (short)rate;
            rateMatrixLocal[userId].push_back(tmpNode);
        }
        catch (bad_alloc& ba)
        {
            cerr << "bad_alloc caught: " << ba.what() << endl;
            cout << "Can't allocate the momery!" << endl; exit(1);
        }
    }
    from.close();
    cout<<"read file sucessfully!"<<endl;
    return;
}

// load test set of netflix dataset
void loadProbe(char * fileName, vector<testSetNode>& probeSet, \
	const char* separator)
{
    ifstream in(fileName);
    if (!in.is_open())
    {
        cout << "can't open test set file!\n";
        exit(1);
    }
    char rateStr[256];
    char* pch ; // store a token of a string

    string strTemp;
    int rateValue = 0, itemId = -1, userId = -1, probeNum = 0;
    
    while(in.getline(rateStr, 256))
    {
        strTemp = rateStr;
        if(strTemp.length() < 4) continue;
        int i = 0;
        pch = strtok (rateStr, separator);
        while (pch != NULL)
        {
            if(1 == i) itemId = atoi(pch);
            else if(0 == i) userId = atoi(pch);
            else if(3 == i) rateValue = atoi(pch);
            else if(i > 3) break;
            ++i;
            pch = strtok (NULL, separator);
        }
        try
        {
            testSetNode tmpNode;
            tmpNode.item = itemId;
            tmpNode.rate = (short)rateValue;
            tmpNode.user = userId;
            probeSet.push_back(tmpNode);
            ++probeNum;
        }
        catch (bad_alloc& ba)
        {
            cerr << "bad_alloc caught: " << ba.what() << endl;
            cout << "Can't allocate the momery!" << endl;
            exit(1);
        }
    }
    cout << "Load " << probeNum << " test ratings successfully!"<< endl;
    in.close(); 
}

#endif // SVD_HH_
