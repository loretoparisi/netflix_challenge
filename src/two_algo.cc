#include <two_algo.hh>

Two_Algo::Two_Algo(const std::string &trainingSet,
    const int ratingSigFig) :
    ratingSigFig(ratingSigFig)
{
    currentTrain.load(trainingSet, arma_binary);
}

void Two_Algo::trainFirst(SingleAlgorithm &predAlgo)
{
    cout << "Start training on first model: Global effects..." << endl;
    predAlgo.train(currentTrain);
    cout << "Finished training on global effects." << endl;
}

void Two_Algo::firstResiduals(SingleAlgorithm &predAlgo)
{
    cout << "Start outputting residuals of first model..." << endl;
    int user, item, date;
    float actualRating, predictedRating;
    unsigned int i;
    for(i = 0; i < currentTrain.n_cols; i++)
    {
        user = roundToInt(currentTrain(USER_ROW, i));
        item = roundToInt(currentTrain(MOVIE_ROW, i));
        date = roundToInt(currentTrain(DATE_ROW, i));
        actualRating = currentTrain(RATING_ROW, i);

        predictedRating = predAlgo.predict(user, item, date);
        currentTrain.at(RATING_ROW, i) = actualRating - predictedRating;
    }
    cout << "Finished outputting residuals of first model." << endl;
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

void Two_Algo::trainSecond(SingleAlgorithm &predAlgo)
{
    predAlgo.train(currentTrain);
}

void Two_Algo::outputQual(SingleAlgorithm &predAlgo,
    const std::string &testFileName,
    const std::string &previousOutputName,
    const std::string &newOutputFileName)
{
    std::ifstream testDataFile(testFileName);
    std::ifstream previousOutputFile(previousOutputName);
    std::ofstream outputFile(newOutputFileName); 

    if (testDataFile.fail())
    {
        throw std::runtime_error("Couldn't find test file at "
            + testFileName);
    }

    if (previousOutputFile.fail())
    {
        throw std::runtime_error("Couldn't find test file at "
            + previousOutputName);
    }

    if (outputFile.fail())
    {
        throw std::runtime_error("Couldn't open output file at " 
            + newOutputFileName);
    }

    std::string testDataLine;
    float originalRating;
    cout << "Testing on data in " << testFileName << "..." << endl;

    while (getline(testDataFile, testDataLine))
    {
        previousOutputFile >> originalRating;
        // Read the line and split it.
        std::vector<int> thisLineVec;
        splitIntoInts(testDataLine, DELIMITER, thisLineVec);

        if (thisLineVec.size() != 3)
        {
            throw std::logic_error("The line \"" + testDataLine + "\" did not "
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
        prediction = originalRating + prediction;
        if (prediction > 5)
            prediction = 5;
        if (prediction < 1)
            prediction = 1;
        outputFile << std::setprecision(ratingSigFig) << prediction << endl;
    }

    outputFile.close();

    cout << "Outputted predictions on " << testFileName << " to the " 
        "output file " << newOutputFileName << endl;
}

Two_Algo::~Two_Algo()
{
    // No dynamically allocated resources to free at the moment.
}
