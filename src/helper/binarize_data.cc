/**
 * This script just turns a few common files into Armadillo matrices, and
 * then saves those matrices in binary format. No command-line arguments
 * are taken.
 *
 */

#include <iostream>
#include <netflix.hh>

using namespace std;
using namespace arma;
using namespace netflix;

// Note: All constants are specified in the netflix namespace.

int main(void)
{
    // This code is relatively lazy; it just runs the parseData() helper
    // function in separate scopes. While this is slow, it's also less
    // likely to eat up all your RAM, since we have to make around 7
    // different matrices.

    // Separate scopes to make memory management easier...
    {
        // Only base data
        cout << "Starting to parse base data..." << endl;
        fmat baseData = parseData(INDEX_PATH, DATA_PATH, BASE_IDX);
        baseData.save(BASE_BIN, arma_binary);

        cout << "Saved base data to " << BASE_BIN << ".\n" << endl;
    }

    {
        // Only hidden data
        cout << "Starting to parse hidden data..." << endl;
        fmat hiddenData = parseData(INDEX_PATH, DATA_PATH, HIDDEN_IDX);
        hiddenData.save(HIDDEN_BIN, arma_binary);

        cout << "Saved hidden data to " << HIDDEN_BIN << ".\n" << endl;
    }

    {
        // Only "valid" data
        cout << "Starting to parse valid data..." << endl;
        fmat validData = parseData(INDEX_PATH, DATA_PATH, VALID_IDX);
        validData.save(VALID_BIN, arma_binary);

        cout << "Saved valid data to " << VALID_BIN << ".\n" << endl;
    }

    {
        // Only probe data
        cout << "Starting to parse probe data..." << endl;
        fmat probeData = parseData(INDEX_PATH, DATA_PATH, PROBE_IDX);
        probeData.save(PROBE_BIN, arma_binary);

        cout << "Saved probe data to " << PROBE_BIN << ".\n" << endl;
    }

    {
        // Base and hidden data
        cout << "Starting to parse base and hidden data..." << endl;
        fmat baseHiddenData = parseData(INDEX_PATH, DATA_PATH,
                                        BASE_HIDDEN_IDX);
        baseHiddenData.save(BASE_HIDDEN_BIN, arma_binary);

        cout << "Saved base and hidden data to " << BASE_HIDDEN_BIN 
            << ".\n" << endl;
    }

    {
        // Base MU for Global Effect
        cout << "Starting to parse BASE MU data..." << endl;
        fmat trainMUHiddenData = parseData(INDEX_PATH_MU,
            DATA_PATH_MU, BASE_IDX);
        trainMUHiddenData.save(MU_BASE_BIN, arma_binary);

        cout << "Saved base data to " << MU_BASE_BIN 
            << ".\n" << endl;
    }

    {
        // Base, hidden, and valid data
        cout << "Starting to parse base, hidden, and valid data..." << 
            endl;
        fmat baseHiddenValidData = parseData(INDEX_PATH, DATA_PATH,
                                             BASE_HIDDEN_VALID_IDX);
        baseHiddenValidData.save(BASE_HIDDEN_VALID_BIN, arma_binary);

        cout << "Saved base, hidden, and valid data to " << 
            BASE_HIDDEN_VALID_BIN << ".\n" << endl;
    }

    {
        // All training data, i.e. base, hidden, valid, and probe.
        cout << "Starting to parse all training data..." << endl;
        fmat allTrainData = parseData(INDEX_PATH, DATA_PATH,
                                      ALL_TRAIN_IDX);
        allTrainData.save(ALL_TRAIN_BIN, arma_binary);

        cout << "Saved all training data to " << ALL_TRAIN_BIN << 
            ".\n" << endl;
    }

    {
        // All training data in MU order.
        cout << "Starting to parse all MU data..." << endl;
        fmat allTrainDataMu = parseData(INDEX_PATH_MU, DATA_PATH_MU,
                                        ALL_TRAIN_IDX);
        allTrainDataMu.save(MU_ALL_TRAIN_BIN, arma_binary);

        cout << "Saved all MU training data to " << MU_ALL_TRAIN_BIN 
            << ".\n" << endl;
    }

    cout << "\nSaved all desired data in Armadillo binary format." << endl;
}
