#include <cassert>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>

#ifndef NDEBUG
#include <iostream>
#endif

#include <rbm.hh>

// The indices of the dataset to use for training.
const std::set<int> TRAINING_SET_INDICES = {BASE_SET};

// Sig-figs for output file.
const int RATING_SIG_FIGS = 4;

// The name of the output file to use (for predictions on "qual").
const std::string OUTPUT_FN = "data/rbm_predictions.mat";

/*
 * Goes through the test file and saves the prediction results to the
 * specified output file.
 *
 */
// void testOnDataFile(BaseAlgorithm &predAlgo, const std::string &testFileName,
//                     const std::string &outputFileName)
// {
//     std::ifstream qualDataFile(testFileName);
//     std::ofstream outputFile(outputFileName); 
    
//     if (qualDataFile.fail())
//     {
//         throw std::runtime_error("Couldn't find test file at " + testFileName);
//     }

//     if (outputFile.fail())
//     {
//         throw std::runtime_error("Couldn't open output file at " 
//                                  + outputFileName);
//     }

//     std::string qualDataLine;
    
//     std::cout << "\nTesting on data in " << testFileName << "..." << std::endl;

//     while (std::getline(qualDataFile, qualDataLine))
//     {
//         // Read the line and split it.
//         std::vector<int> thisLineVec;
//         splitIntoInts(qualDataLine, DELIMITER,
//                       thisLineVec);

//         if (thisLineVec.size() != 3)
//         {
//             throw std::logic_error("The line \"" + qualDataLine + "\" did not "
//                                    "contain three delimiter-separated "
//                                    "entries!");
//         }
        
//         // The first entry is the user ID, the second entry is the item ID,
//         // and the third entry is the date. All of these should be
//         // zero-indexed!
//         int user = thisLineVec[0];
//         int item = thisLineVec[1];
//         int date = thisLineVec[2];
        
//         // Output the prediction to file.
//         float prediction = predAlgo.predict(user, item, date);
//         outputFile << std::setprecision(RATING_SIG_FIGS) << prediction << std::endl;
//     }
    
//     outputFile.close();
    
//     cout << "\nOutputted predictions on " << testFileName << " to the " 
//             "output file " << outputFileName << endl;
// }

double computeRMSE(const Mat<data_t> &data, const Mat<data_t> &predictions) {
    if ( data.n_cols != predictions.n_cols ) {
        std::ostringstream msg;
        msg << "Missing predictions for " << data.n_cols - predictions.n_cols
            << " entries";
        throw std::logic_error(msg.str());
    }

    double rmse = 0.0;

    for ( unsigned col = 0; col < data.n_cols; ++col ) {
        double delta = pow((double)data.at(RATING_ROW, col) - 
                           (double)predictions.at(2, col), 2.0);
        rmse += delta;
        if ( col % 100000 == 0 ) {
            std::cout << delta << ", " << rmse << std::endl;
            std::cout << (double)data.at(RATING_ROW, col) << " " 
                      << (double)predictions.at(2, col)
                      << std::endl;
        }
    }
    rmse *= (1.0 / data.n_cols);

    return sqrt(rmse);
}

int main() {
    std::cout << "Intializing RBM" << std::endl;
    RBM rbm(NUM_USERS, NUM_MOVIES, HIDDEN, EPSILON, MOMENTUM);
    std::cout << "Training RBM" << std::endl;
    fmat data;
    data.load(BASE_BIN);
    rbm.train(data);
    std::cout << "Generating predictions on probe set" << std::endl;
    Mat<data_t> probe;
    probe.load(PROBE_BIN);
    Mat<data_t> predictions = rbm.predict(probe);
    predictions.save(OUTPUT_FN);
    double rmse = computeRMSE(probe, predictions);
    std::cout << "RBM achieved RMSE of " << rmse << " on probe set" 
              << std::endl;
    
    return 0;
}