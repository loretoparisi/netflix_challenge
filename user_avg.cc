#include <assert.h>
#include <fstream>
#include <iomanip>
#include <iostream>

#define IDX_PATH "test/um/test.idx"
#define DTA_PATH "test/um/test.dta"
#define OUTPUT_PATH "test/um/test.out"
#define PRECISION 4 // 4 Sig figs to output file

using namespace std;

int main() {
    // Input/output file specification
    ifstream indexFile(IDX_PATH);
    ifstream dataFile(DTA_PATH);
    ofstream outputFile(OUTPUT_PATH, ios::out);

    // Variables by user
    int idx, currentUser = 1, sumRating = 0, countRating = 0, countQual = 0;
    bool newUser = false;
    // Variables by line
    int currUser, currMovie, currDate, currRate;
    float avg;

    while (indexFile >> idx) {
        dataFile >> currUser >> currMovie >> currDate >> currRate;

        // Not sure if we should use data with score above 3?
        if (idx >= 4)
            continue;
        // This is the "qual" data case
        if (currRate == 0) {
            countQual++;
            continue;
        }

        newUser = (currUser != currentUser);
        // Before switching user, output prev user's avg to file
        if (newUser) {
            if (countQual) {
            	// Check that at least each user has some training data
                assert(countRating != 0);
                avg = (float)sumRating / countRating;
                while(countQual) {
                    outputFile << setprecision(PRECISION) << avg << endl;
                    countQual--;
                }
            }
            sumRating = 0;
            countRating = 0;
            currentUser = currUser;
        }
        sumRating += currRate;
        countRating++;
    }

    // If last line happens to be a "qual" data, then we still need
    // to output reusult
    if (countQual != 0) {
    	// Check that at least each user has some training data
        assert(countRating != 0);
        avg = (float)sumRating / countRating;
        while(countQual) {
            outputFile << avg << endl;
            countQual--;
        }
    }
    outputFile.close();

    return 0;
}
