#include "rbm.hh"

int main() {
    RBM rbm(NUM_USERS, NUM_MOVIES, HIDDEN, EPSILON);
    rbm.train();
    
    return 0;
}