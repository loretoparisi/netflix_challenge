#include <iostream>

#include <rbm.hh>

// The indices of the dataset to use for training.
const std::set<int> TRAINING_SET_INDICES = {BASE_SET};

int main() {
#ifndef NDEBUG
    std::cout << "Intializing RBM" << std::endl;
#endif
    RBM rbm(NUM_USERS, NUM_MOVIES, HIDDEN, EPSILON, MOMENTUM);
#ifndef NDEBUG
    std::cout << "Training RBM" << std::endl;
#endif
    rbm.train(BASE_BIN);
    
    return 0;
}