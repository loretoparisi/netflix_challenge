#!/usr/bin/env python2

# This script creates a "bins.dta" file that specifies a bin each rating
# belongs to. This is chosen based on how many movies that given user
# rated. The output of this code is used for quiz bin blending.

import matplotlib.pyplot as plt
import numpy as np
import sys

# Constants

# The input data file for all the data
ALL_DTA = "../../data/um/new_all.dta"

# The input data file for qual-only data.
QUAL_DTA = "../../data/um/new_qual.dta"

# The file to output to.
BINS_FILE = "../../data/quiz_bin_blend/bins.dta"

# The number of users in the dataset
NUM_USERS = 458293

# The number of ratings in new_qual.dta
NUM_QUAL_RAT = 2749898

# The number of bins to use, and the edges of each boundary (in units of
# number of movies rated). These are chosen based on visualizing
# histograms.
NUM_BINS = 15

# This must have NUM_BINS + 1 entries. The upper endpoint of each boundary
# is *not* included.
#
# Statistics of dataset (15 bins):
#       * Min number of ratings by a user: 18.0
#       * Average number of ratings by a user: 223.473424207
#       * Median number of ratings by a user: 111.0
#       * Max number of ratings by a user: 3496.0
#       * Percentile 0.0: 18.0
#       * Percentile 6.66666666667: 26.0
#       * Percentile 13.3333333333: 34.0
#       * Percentile 20.0: 43.0
#       * Percentile 26.6666666667: 52.0
#       * Percentile 33.3333333333: 64.0
#       * Percentile 40.0: 79.0
#       * Percentile 46.6666666667: 99.0
#       * Percentile 53.3333333333: 125.0
#       * Percentile 60.0: 158.0
#       * Percentile 66.6666666667: 202.0
#       * Percentile 73.3333333333: 260.0
#       * Percentile 80.0: 342.0
#       * Percentile 86.6666666667: 467.0
#       * Percentile 93.3333333333: 697.0
#       * Percentile 100.0: 3496.0
#
# With 5 bins:
#       * Percentile 0.0: 18.0
#       * Percentile 20.0: 43.0
#       * Percentile 40.0: 79.0
#       * Percentile 60.0: 158.0
#       * Percentile 80.0: 342.0
#       * Percentile 100.0: 3496.0
#       * Clear rightwards skew (towards higher values)
#

# Even 15 bins:
BIN_BOUNDARIES = [0.0, 26.0, 34.0, 43.0, 52.0, 64.0, 79.0, 99.0, 125.0,
                  158.0, 202.0, 260.0, 342.0, 467.0, 697.0, 3497.0]

# Even 5 bins:
# BIN_BOUNDARIES = [0.0, 43.0, 79.0, 158.0, 342.0, 3497.0]

# Uneven 5 bins:
# BIN_BOUNDARIES = [0.0, 80.0, 170.0, 340.0, 680.0, 3497.0]

# If True, plot a histogram instead of outputting the actual bins. This is
# useful for finding boundaries.
DEBUG = False


# Plot the rating distribution histogram.
def plotHistogram(ratingDist):
    assert DEBUG

    fig = plt.figure(facecolor = "white")

    hist, bins = np.histogram(ratingDist, bins = 150)
    center = (bins[:-1] + bins[1:]) / 2.0
    width = 0.7 * (bins[1] - bins[0])

    plt.bar(center, hist, align = "center", width = width)
    plt.show()


# Main function. Reads in qual data, looks at number of movies each user
# rated, and splits up ratings in qual based on that.
def main():

    if DEBUG:
        print "In DEBUG mode; plotting histograms instead.\n"
    else:
        assert len(BIN_BOUNDARIES) == NUM_BINS + 1

        print "NOT in DEBUG mode; splitting data into", NUM_BINS, "bins", \
              "based on boundaries:", BIN_BOUNDARIES, "\n"

    # This array has NUM_USERS elements. The ith element is the number of
    # ratings user "i" made in new_all.dta
    userIDToTotalNumRatings = np.zeros(NUM_USERS)
    lineNum = 0

    with open(ALL_DTA, "r") as allFile:

        for line in allFile:
            thisLineSplit = line.split()

            userID = int(thisLineSplit[0])
            userIDToTotalNumRatings[userID] += 1

            lineNum += 1

            if lineNum % 1000000 == 0:
                print "Parsed", lineNum, "lines so far."

    print "\nPopulated userIDToTotalNumRatings"

    if DEBUG:
        print "\nStatistics:"
        print "Min number of ratings by a user:", \
                np.amin(userIDToTotalNumRatings)
        print "Average number of ratings by a user:", \
                np.mean(userIDToTotalNumRatings)
        print "Median number of ratings by a user:", \
                np.median(userIDToTotalNumRatings)
        print "Max number of ratings by a user:", \
                np.amax(userIDToTotalNumRatings)

        # Print out the relevant percentiles, based on the NUM_BINS.
        for perc in np.arange(0.0, 100.0 + 0.001, 100.0 / NUM_BINS):
            print "Percentile " + str(perc) + ": " +\
                    str(np.percentile(userIDToTotalNumRatings, perc))

        # Make a histogram and plot it.
        print "\nPlotting histogram..."
        plotHistogram(userIDToTotalNumRatings)

    else:
        # Divide users into bin boundaries. This array will contain a bin
        # number (between 0 and NUM_BINS - 1) for each user (where the ith
        # element is the user "i").
        userBins = np.digitize(userIDToTotalNumRatings, BIN_BOUNDARIES) - 1

        # Start outputting to BINS_FILE by reading entries in QUAL_DTA,
        # finding which bin each user belongs to, and outputting that.
        fout = open(BINS_FILE, "w")

        with open(QUAL_DTA, "r") as qualFile:

            for line in qualFile:
                thisLineSplit = line.split()
                userID = int(thisLineSplit[0])

                thisUserBin = int(userBins[userID])

                assert thisUserBin >= 0 and thisUserBin < NUM_BINS

                fout.write(str(thisUserBin))

                if lineNum != NUM_QUAL_RAT - 1:
                    fout.write("\n")
                elif lineNum >= NUM_QUAL_RAT:
                    print "\nToo many ratings in", QUAL_DTA
                    sys.exit(1)

        fout.close()

        print "\nOutputted users' bins to", BINS_FILE


if __name__ == "__main__":
    main()
