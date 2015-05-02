#include <iomanip>
#include <fstream>
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
const std::string OUTPUT_FN = "data/rbm_predictions.dta";

/*
 * Goes through the test file and saves the prediction results to the
 * specified output file.
 *
 */
void testOnDataFile(SingleAlgorithm &predAlgo, const std::string &testFileName,
                    const std::string &outputFileName)
{
    std::ifstream qualDataFile(testFileName);
    std::ofstream outputFile(outputFileName); 
    
    if (qualDataFile.fail())
    {
        throw std::runtime_error("Couldn't find test file at " + testFileName);
    }

    if (outputFile.fail())
    {
        throw std::runtime_error("Couldn't open output file at " 
                                 + outputFileName);
    }

    std::string qualDataLine;
    
    std::cout << "\nTesting on data in " << testFileName << "..." << std::endl;

    while (std::getline(qualDataFile, qualDataLine))
    {
        // Read the line and split it.
        std::vector<int> thisLineVec;
        splitIntoInts(qualDataLine, DELIMITER,
                      thisLineVec);

        if (thisLineVec.size() != 3)
        {
            throw std::logic_error("The line \"" + qualDataLine + "\" did not "
                                   "contain three delimiter-separated "
                                   "entries!");
        }
        
        // The first entry is the user ID, the second entry is the item ID,
        // and the third entry is the date. All of these should be
        // zero-indexed!
        int user = thisLineVec[0];
        int item = thisLineVec[1];
        int date = thisLineVec[2];
        
        // Output the prediction to file.
        float prediction = predAlgo.predict(user, item, date);
        outputFile << std::setprecision(RATING_SIG_FIGS) << prediction << std::endl;
    }
    
    outputFile.close();
    
    cout << "\nOutputted predictions on " << testFileName << " to the " 
            "output file " << outputFileName << endl;
}

int main() {
#ifndef NDEBUG
    std::cout << "Intializing RBM" << std::endl;
#endif
    RBM rbm(NUM_USERS, NUM_MOVIES, HIDDEN, EPSILON, MOMENTUM);
#ifndef NDEBUG
    std::cout << "Training RBM" << std::endl;
#endif
    rbm.train(BASE_BIN);
#ifndef NDEBUG
    std::cout << "Generating predictions on qual set" << std::endl;
#endif
    testOnDataFile(rbm, QUAL_DATA_FN, OUTPUT_FN);
    
    return 0;
}