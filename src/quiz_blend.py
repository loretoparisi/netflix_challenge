#!/usr/bin/env python3

# This script carries out quiz blending on the predictions stored in the
# data/quiz_blend directory (which were presumably generated by predictors
# trained on all of quiz). For more details, refer to:
#
#   http://www.netflixprize.com/assets/GrandPrize2009_BPC_BigChaos.pdf
#
# This script should be executed at the top level of this repo. It takes
# the output file's name as a required argument, and also takes a "-v" or
# "--verbose" optional argument.
#
# NOTE: Files must include their quiz RMSE in their names! See the README
# in data/quiz_blend for more information.
#

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import math
import sys

plt.rcParams.update({'figure.autolayout': True})
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7

# Constants

# The blending directory (relative to this repo's topmost directory).
QBLEND_DIR = "data/quiz_blend/"

# The extension of predictor data.
PRED_EXT = ".dta"

# The string that comes before RMSEs in file names.
BEFORE_RMSE = "_QRMSE_"

# Regularization constant for our linear regression. This value came from
# BigChaos's 2009 paper.
LAMBDA = 0.0014

# The number of ratings in the qual set. This is used to standardize X
# before regressing. Note that this is referred to as N_L below.
NUM_QUAL_RAT = 2749898

# The mean rating in the quiz set.
QMEAN = 3.674

# The variance of the quiz set, i.e. 1/N_Q sum_u (y_u - yMean)^2
QVAR = 1.274

# The minimum and maximum ratings that we'll allow in the outputs. These
# were chosen by just trying various combinations and submitting them to
# the scoreboard.
MIN_BLEND_RATING = 1.0
MAX_BLEND_RATING = 5.0

# Separate note: The number of ratings in quiz is referred to as N_Q in the
# code below and the comments above.


# Prints a usage statement for this program, and then exits.
def usage(progName):
    # This program either takes one or two argument(s). If it is given two
    # argument, one argument must be a verbose flag.
    print("usage: python3 " + progName + " [-v] outputFile\n\n" +
          "optional arguments:\n" +
          "  -v, --verbose\tprint additional status outputs");
    sys.exit(1)


# Plots the correlation matrix, to see if any predictors are too highly
# correlated. Note that each column of X represents a prediction. The name
# of the prediction corresponding to the ith column in X will be given by
# predNames[i].
def plotCorrHeatmap(X, predNames):
    # Each column represents a prediction, so rowvar = 0.
    corr = np.corrcoef(X, rowvar = 0)

    # Plot the heat map. Black = low correlation (i.e. 0.85 or below);
    # white = high correlation (1.00).
    fig = plt.figure(facecolor='w')
    plt.imshow(corr, interpolation="nearest", cmap=plt.cm.hot,
               vmin=0.85, vmax=1.00)

    plt.xticks(np.arange(len(predNames)), predNames, rotation=90)
    plt.yticks(np.arange(len(predNames)), predNames)
    plt.title("Predictors' Correlation Matrix")
    plt.colorbar()

    plt.show()


# The main quiz blending function. Our approach is as follows:
#
# We want to solve for beta in y = X * beta, where y is an N_Q x 1 array of
# actual *quiz* ratings minus the mean quiz rating, and then divided by
# sqrt(N_Q) (to standardize). Also, X is an N_Q x p matrix holding the
# predictions of "p" predictors on the quiz set with the mean quiz set
# rating subtracted, and then divided by sqrt(N_Q). And, lastly, beta is a
# p x 1 weight matrix.
#
# If we add a regularization parameter LAMBDA to this least-squares
# minimization problem, we then have:
#
#   beta = (X^T * X + LAMBDA * I)^(-1) * (X^T * y)
#
# Of course, this is impossible to solve directly since we don't know which
# ratings are in quiz. So, we'll instead take a different approach. We'll
# approximate X^T * X by using the array X of predictors' predictions on
# the entire "qual" set (i.e. X is an N_L x p matrix). For each predictor,
# we'll subtract off the mean quiz rating and then divide by sqrt(N_L) (to
# standardize each predictor), so we should get the same result.
#
# To estimate X^T * y, which is a p x 1 column vector, we use the fact that
# the jth element in this column vector (j in [0, p-1]) is given by:
#
#   sum_{u} x_{uj} y_{u}
#   = 1/2 [sum_{u} y_u^2 + sum_{u} x_{uj}^2 - sum_{u} (y_u - x_{uj})^2]
#
# Where the sum over u extends from u = 0 to u = N_Q. However, in our
# example, we've actually standardized y by subtracting off its (known)
# mean and then dividing by sqrt(N_Q). So this means that the above sum is:
#
#   1/2 [QVAR + sum_{u} x_{uj}^2 - MSE_OF_PREDICTOR_J]
#
# And we can estimate sum_{u} x_{uj}^2 with the standardized X matrix
# generated by qual predictions (hence the standardization).
#
# Once beta is found, the final prediction is then given by standardized y =
# standardized X * beta. Here, standardized y is (y - QMEAN) / sqrt(N_L),
# so we need to scale up accordingly.
#
def main():

    # This script only takes exactly one or two arguments.
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        usage(sys.argv[0])

    # Verbose flag
    verbose = False
    verboseFlagFound = False

    # The output file's name
    outputFileName = None

    # If there's one arg, then it's the output file's name. If there's two
    # args, one should be the verbose flag.
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "-v" or sys.argv[i] == "--verbose":
            verbose = True
            verboseFlagFound = True
        else:
            outputFileName = sys.argv[i]

    if outputFileName == None or \
        (len(sys.argv) == 3 and not verboseFlagFound):
        # No output file found.
        usage(sys.argv[0])

    # The number of predictors, or "p".
    numPredictors = 0

    if verbose:
        print("Using predictors in", QBLEND_DIR)

    # Find how many prediction files there are in the quiz blend directory,
    # and store their names (in alphabetical order).
    predFiles = glob.glob(QBLEND_DIR + "*" + PRED_EXT)
    predFiles = sorted(predFiles)

    predNames = list()

    for predFile in predFiles:
        # Ignore the part before the QRMSE in getting this predictor's
        # name, as well as the part before the QBLEND_DIR (including the
        # "/").
        afterDirStart = predFile.rindex(QBLEND_DIR) + len(QBLEND_DIR)
        beforeRMSEStart = predFile.rindex(BEFORE_RMSE)
        predName = predFile[afterDirStart:beforeRMSEStart]

        predNames.append(predName)
        numPredictors += 1

    if verbose:
        print("Found", numPredictors, "predictor(s) in", QBLEND_DIR)

    # This is an array of all of our predictors' qual predictions. It will
    # be N_L x p in shape. We will standardize the qual predictions by
    # subtracting off the mean quiz rating and dividing by sqrt(N_L).
    X = np.zeros((NUM_QUAL_RAT, numPredictors))

    # An array of each predictor's RMSE. The jth element will be the RMSE
    # of predictor j.
    predQRMSEs = np.zeros(numPredictors)

    # We'll store sum_{u} x_{uj}^2 as we load data into X. The jth element
    # of the following array will be sum_{u} x_{uj}^2 for predictor j.
    sumXColSquared = np.zeros(numPredictors)

    # Parse all of the prediction files in order to load up X and predQRMSEs
    if verbose:
        print("\nParsing predictor files...")

    for predInd in range(numPredictors):
        fileName = predFiles[predInd]

        # Get the RMSE for this predictor from its file name. If this is
        # not possible, an error will be thrown.
        beforeRMSEStart = fileName.rindex(BEFORE_RMSE)
        rmseStart = beforeRMSEStart + len(BEFORE_RMSE)
        rmseEnd = len(fileName) - len(PRED_EXT)

        thisQuizRMSE = float(fileName[rmseStart:rmseEnd])
        predQRMSEs[predInd] = thisQuizRMSE

        # Also get the predictor's name, as stored earlier.
        predName = predNames[predInd];

        # Parse the data in this prediction file. Make sure the number of
        # lines equals NUM_QUAL_RAT as expected.
        lineNum = 0

        # First, just store the entries, before standardization.
        with open(fileName, "r") as predFile:

            for line in predFile:
                thisRating = float(line)
                X[lineNum, predInd] = thisRating

                lineNum += 1

        assert lineNum == NUM_QUAL_RAT

        # Standardize this column by subtracting off the quiz mean and
        # dividing by sqrt(N_L)
        X[:, predInd] -= QMEAN
        X[:, predInd] /= math.sqrt(NUM_QUAL_RAT)

        # Store sum_{u} x_{uj}^2 for this predictor, post-standardization.
        sumXColSquared[predInd] = np.sum(np.square(X[:, predInd]))

        if verbose:
            print("Finished parsing data for predictor " + predName +
                  ", which had a QRMSE of " + str(thisQuizRMSE))

    if verbose:
        print("Finished parsing predictor files.")
        print("\nDisplaying predictors' correlation heatmap.")
        plotCorrHeatmap(X, predNames)

    # Compute (X^T * X + LAMBDA * I)
    xTransXPlusLam = (X.T).dot(X) + LAMBDA * np.identity(numPredictors)

    # Estimate X^T * y (which is a p-element column vector), by using the
    # fact that the jth entry is 1/2 [QVAR + sum_{u} x_{uj}^2 -
    # MSE_OF_PREDICTOR_J (i.e. the MSE, not the RMSE)].
    xTransYEst = np.zeros((numPredictors, 1))

    for j in range(numPredictors):
        xTransYEst[j, 0] = 0.5 * (QVAR + sumXColSquared[j] -
                                  predQRMSEs[j]**2.0)

    # Compute beta = (X^T * X + LAMBDA * I)^(-1) * (X^T * y)
    beta = (np.linalg.inv(xTransXPlusLam)).dot(xTransYEst)

    if verbose:
        print("\nComputed an un-normalized beta. NOT normalizing.")
        print("Sum of elements in un-normalized beta:", np.sum(beta))

    if verbose:
        print("\nElements in un-normalized beta:")

        for i in range(len(beta)):
            print("    * " + predNames[i] + ": " + \
                  str('%0.3f' % beta[i, 0]))

    # The final list of standardized predictions is given by standardized y
    # = standardized X * beta. Here, standardized y is (y - QMEAN) /
    # sqrt(N_L), so we need to transform accordingly to get y.
    blendedPred = X.dot(beta)   # standardized prediction
    blendedPred *= math.sqrt(NUM_QUAL_RAT)
    blendedPred += QMEAN

    # Clip the blended prediction values
    blendedPred = np.clip(blendedPred, MIN_BLEND_RATING, MAX_BLEND_RATING)

    if verbose:
        print("\nClipped blended predictions to range [" +
              str(MIN_BLEND_RATING) + ", " + str(MAX_BLEND_RATING) + "]")


    # Save to the specified output file.
    np.savetxt(outputFileName, blendedPred, '%0.3f')

    print("Saved blended predictions to", outputFileName)


if __name__ == "__main__":
    main()
