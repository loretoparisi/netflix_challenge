#include <two_algo.hh>


Two_Algo::Two_Algo(const std::string &trainingSet,
        const std::string &intermediatePredFileName,
        const int ratingSigFig) :
    intermediatePredFileName(intermediatePredFileName),
    ratingSigFig(ratingSigFig)
{
    currentTrain.load(trainingSet, arma_binary);

#ifndef NDEBUG
    cout << "Set up Two_Algo by loading data from " <<
        trainingSet << endl;
#endif
}


void Two_Algo::trainFirst(SingleAlgorithm &firstAlgo)
{
#ifndef NDEBUG
    cout << "\nStarted training first model." << endl;
#endif

    firstAlgo.train(currentTrain);

#ifndef NDEBUG
    cout << "Finished training first model." << endl;
#endif
}


void Two_Algo::saveFirstQualPredictions(SingleAlgorithm &firstAlgo,
        const std::string &qualFileName)
{
    // Output predictions to intermediatePredFileName
    std::ifstream qualDataFile(qualFileName);
    std::ofstream outputFile(intermediatePredFileName); 

    if (qualDataFile.fail())
    {
        throw std::runtime_error("Couldn't find qual file at "
            + qualFileName);
    }

    if (outputFile.fail())
    {
        throw std::runtime_error("Couldn't open output file at " 
            + intermediatePredFileName);
    }

    std::string qualDataLine;

    while (getline(qualDataFile, qualDataLine))
    {
        // Read the line and split it.
        std::vector<int> thisLineVec;
        splitIntoInts(qualDataLine, DELIMITER, thisLineVec);

        if (thisLineVec.size() != 3)
        {
            throw std::logic_error("The line \"" + qualDataLine + "\" did not "
                              "contain three delimiter-separated "
                              "entries!");
        }

        // The first entry is the user ID, the second entry is the item ID,
        // and the third entry is the date.
        int user = thisLineVec[0];
        int item = thisLineVec[1];
        int date = thisLineVec[2];

        // Output the prediction of the first algorithm to
        // intermediatePredFileName. Don't bound predictions since we want
        // the second algorithm to correct on where the first went awry.
        float prediction = firstAlgo.predict(user, item, date, false);
        outputFile << std::setprecision(ratingSigFig) << prediction << endl;
    }

    outputFile.close();

#ifndef NDEBUG
    cout << "Outputted first algorithm's qual predictions to the temporary"
        << " file " << intermediatePredFileName << "." << endl;
#endif

}


void Two_Algo::computeFirstResiduals(SingleAlgorithm &firstAlgo)
{
    int user, item, date;
    float actualRating, predictedRating;
    unsigned int i;
    for(i = 0; i < currentTrain.n_cols; i++)
    {
        user = roundToInt(currentTrain(USER_ROW, i));
        item = roundToInt(currentTrain(MOVIE_ROW, i));
        date = roundToInt(currentTrain(DATE_ROW, i));
        actualRating = currentTrain(RATING_ROW, i);

        predictedRating = firstAlgo.predict(user, item, date, false);
        currentTrain.at(RATING_ROW, i) = actualRating - predictedRating;
    }
    
#ifndef NDEBUG
    cout << "Finished computing residuals of first model." << endl;
#endif
}


float Two_Algo::getAverage()
{
    float sum = 0;
    unsigned int i;
    for(i = 0; i < currentTrain.n_cols; i++)
        sum += currentTrain(RATING_ROW, i);
    return sum / currentTrain.n_cols;
}


void Two_Algo::saveResiduals(const std::string residualsFile)
{
    unsigned int i;
    float residual;
    std::ofstream outputResidual(residualsFile);
    for (i = 0; i < currentTrain.n_cols; i++)
    {
        residual = currentTrain(RATING_ROW, i);
        outputResidual << residual << "\n";
    }
}


void Two_Algo::trainSecond(SingleAlgorithm &secondAlgo)
{
#ifndef NDEBUG
    cout << "\nStarted training second model." << endl;
#endif

    secondAlgo.train(currentTrain);

#ifndef NDEBUG
    cout << "Finished training second model." << endl;
#endif

}


/**
 * Note: This actually saves the combined predictions of the two algorithms
 * to file, and then deletes the temp file at intermediatePredFileName.
 */
void Two_Algo::saveSecondQualPredictions(SingleAlgorithm &secondAlgo,
    const std::string &qualFileName,
    const std::string &outputFileName)
{
    // Open up the first algorithm's qual predictions too.
    std::ifstream qualDataFile(qualFileName);
    std::ifstream firstAlgoPredFile(intermediatePredFileName);
    std::ofstream outputFile(outputFileName); 

    if (qualDataFile.fail())
    {
        throw std::runtime_error("Couldn't find qual file at "
            + qualFileName);
    }

    if (firstAlgoPredFile.fail())
    {
        throw std::runtime_error("Couldn't find first algorithm's "
                "predictions at " + intermediatePredFileName);
    }

    if (outputFile.fail())
    {
        throw std::runtime_error("Couldn't open output file at " 
            + outputFileName);
    }

    std::string qualDataLine;
    float firstAlgoPred;

    while (getline(qualDataFile, qualDataLine))
    {
        // Store the previous algorithm's rating in firstAlgoPred.
        firstAlgoPredFile >> firstAlgoPred;

        // Read a line from the qual dataset and split it.
        std::vector<int> thisLineVec;
        splitIntoInts(qualDataLine, DELIMITER, thisLineVec);
        
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
        
        // Combine the second algorithm's prediction with the first
        // algorithm's prediction. Bound the sum and then save that to
        // file.
        float secondAlgoPred = secondAlgo.predict(user, item, date, false);
        float comboPred = firstAlgoPred + secondAlgoPred;

        if (comboPred > MAX_RATING)
        {
            comboPred = MAX_RATING;
        }
        if (comboPred < MIN_RATING)
        {
            comboPred = MIN_RATING;
        }

        outputFile << std::setprecision(ratingSigFig) << comboPred << endl;
    }

    qualDataFile.close();
    firstAlgoPredFile.close();
    outputFile.close();

#ifndef NDEBUG
    cout << "\nOutputted combined algorithm's qual predictions to " <<
        outputFileName << "." << endl;
#endif

    // Delete the temporary file at intermediatePredFileName
    if (std::remove(intermediatePredFileName.c_str()) != 0)
    {
#ifndef NDEBUG
        cerr << "Unable to delete temporary file at " << 
            intermediatePredFileName << endl;
#endif
    }
    else
    {
#ifndef NDEBUG
        cout << "Deleted temporary file at " << intermediatePredFileName <<
            endl;
#endif
    }
}


Two_Algo::~Two_Algo()
{
    // No dynamically allocated resources to free at the moment.
}
