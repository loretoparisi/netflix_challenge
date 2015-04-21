#include "netflix.hh"
#define USER_NUM netflix::NUM_USERS
#define ITEM_NUM netflix::NUM_MOVIES
#define K_NUM  200
#define TRAINING_SET "../../netflix_challenge/data/train.dta"
#define PROBE_SET "../../netflix_challenge/data/probe.dta"
#define QUAL_SET "../../netflix_challenge/data/um/new_qual.dta"
#define OUT_FILE "output_svd.dta"
#define RATE_SP " "
#define ITERATION 60

#include "svd.cc"

int main(int argc, char ** argv)
{
    time_t start,end;
    struct tm* startInfo;
    struct tm* endInfo;
    float duration;
    start = time(NULL);
    startInfo = localtime(&start);
    string startStr = asctime(startInfo);
    float alpha1 = 0.008; // according suggestion of xlvector
    float alpha2 = 0.008; // according suggestion of xlvector
    float beta1  = 0.01;  //according suggestion of xlvector
    float beta2  = 0.01;  //according suggestion of xlvector  

    svd::model(K_NUM, alpha1, alpha2, beta1, beta2, ITERATION, 0.9);

    end = time(NULL);
    duration = end-start;
    endInfo =   localtime(&end);
    cout << "start at: " << startStr << " ";
    cout <<" end at: "<< asctime(endInfo) <<endl;
    cout << "duration: "<< duration <<" s." <<endl;
    return 0;
}
