#include <iostream>

#include "rbm.hh"

#define INDEX_PATH "data/um/all.idx"
#define DATA_PATH "data/um/new_all.dta"

// The indices of the dataset to use for training.
const std::set<int> TRAINING_SET_INDICES = {BASE_SET};

int main() {
#ifndef NDEBUG
    std::cout << "Loading data" << std::endl;
#endif
    imat data = parseData(INDEX_PATH, DATA_PATH, TRAINING_SET_INDICES);
#ifndef NDEBUG
    std::cout << "Intializing RBM" << std::endl;
#endif
    RBM rbm(NUM_USERS, NUM_MOVIES, HIDDEN, EPSILON);
#ifndef NDEBUG
    std::cout << "Training RBM" << std::endl;
#endif
    rbm.train(data);
    
    return 0;
}