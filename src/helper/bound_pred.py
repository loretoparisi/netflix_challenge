#!/usr/bin/env python3

# This program takes a dta file and bounds its predictions to between 1 and
# 5 (and outputs another file in the process). This is useful for combo
# algorithm runs (e.g. kNN on Time-SVD++) where the first model's
# predictions aren't bounded.
#
# Usage (from top-level directory, ideally):
#       python3 src/helper/bound_pred.py <path to data file>
#               <path to output file>
#

import sys

# Constants

MIN_RATING = 1.0
MAX_RATING = 5.0

# This function takes the input and output file names as its arguments.
def bound(inFileName, outFileName):

    fout = open(outFileName, 'w');

    with open(inFileName) as inputFile:
        print("Reading rating data from", inFileName)
        print("Writing bounded rating data to", outFileName)

        for line in inputFile:
            thisLine = line.strip().split()

            # Each line should only contain one number: its rating.
            assert len(thisLine) == 1

            rating = float(thisLine[0])

            if rating >= MIN_RATING and rating <= MAX_RATING:
                # Just output this rating without changing its precision.
                fout.write(thisLine[0] + "\n")
            elif rating < MIN_RATING:
                fout.write(str(MIN_RATING) + "\n")
            else:
                fout.write(str(MAX_RATING) + "\n")

    print("\nFinished writing bounded rating data to", outFileName)

    fout.close()


# Print a usage statement and exit.
def usage(progName):
    print("Usage: python3", progName, "<path to data file>",
          "<path to output file>")

    print("\nExiting...")
    sys.exit(1)


if __name__ == "__main__":

    # This script takes the input file and output file as its arguments.
    if len(sys.argv) != 3:
        usage(sys.argv[0])

    bound(sys.argv[1], sys.argv[2])
