#!/usr/bin/env python3

# This helper function computes f_{ut} for a user u at day t. This is just
# the floor of the log base A_CONST of the number of ratings user u gave
# on day t.
#
# We output f_{ut} for each user and time pair to a plain-text file, which
# has lines of the form:
#
#   <USER ID> <DATE ID> <f_{ut} FOR THAT USER/DAY COMBO>
#
# We also keep track of the minimum and maximum values of f_{ut} and the
# maximum number of ratings by any given user on a given day, and output
# these at the end. For A_CONST = 6.76, these are:
#
#   Max number of ratings on a given day is 2651.0 which was achieved by
#   user 16765 on date 1970
#
#   Minimum fUT is 0 (many users have this because they rate < 7 times/day)
#   Maximum fUT is 4 for user 16765 and date 1970
#

import math

# Constants

# Delimiter to use between values in the output file.
DELIMITER = " "

# The constant "a" to use as the base of our logarithm.
A_CONST = 6.76

# The input data file.
ALL_DTA = "../../data/um/new_all.dta"

# The file to output to.
OUTPUT_F_U_T = "../../data/f_u_t.dta"


# A mapping from (user ID, date ID) tuples to f_{ut} for that user/date
# combo. Initially this will just hold the number of ratings on a given
# date for that user.
userDateToFUT = dict();

# Track the maximum number of ratings on a given day by a user.
userDateMaxRatings = None
maxRatings = -float('inf')


# Iterate through all entries in the "all" data file, and find the number
# of ratings for each user/date combo.
with open(ALL_DTA, "r") as allFile:

    for line in allFile:
        thisLineSplit = line.split()

        userID = int(thisLineSplit[0])
        dateID = int(thisLineSplit[2])

        userDate = (userID, dateID)

        newValue = 1.0

        # See if there's already an entry in the dict for this (user, date)
        # combination, and update the number of ratings accordingly.
        if userDate in userDateToFUT:
            newValue = userDateToFUT[userDate] + 1.0
            userDateToFUT[userDate] = newValue;
        else:
            userDateToFUT[userDate] = newValue;

        if newValue > maxRatings:
            maxRatings = newValue
            userDateMaxRatings = userDate

print("Finished adding rating counts for each user/date combo.")
print("Max number of ratings on a given day is", maxRatings, "which was",
      "achieved by user", userDateMaxRatings[0], "on date",
      userDateMaxRatings[1])

# Track the minimum and maximum f_{ut}, and which user/date they correspond
# to.
userDateMinFUT = None
minFUT = float('inf')
userDateMaxFUT = None
maxFUT = -float('inf')

# Take the floor(log_a(F_{ut})) for all F_{ut} (numbers of ratings) in
# userDateToFUT, in order to get f_{ut}.
for userDate in userDateToFUT:
    numRatings = userDateToFUT[userDate];
    fUT = int(math.log(numRatings)/math.log(A_CONST))

    if fUT < minFUT:
        minFUT = fUT
        userDateMinFUT = userDate
    if fUT > maxFUT:
        maxFUT = fUT
        userDateMaxFUT = userDate

    userDateToFUT[userDate] = int(math.log(numRatings)/math.log(A_CONST))

print("\nFinished populating userDateToFUT with f_{ut} values.")
print("Minimum fUT is", minFUT, "for user", userDateMinFUT[0], "and date",
      userDateMinFUT[1])
print("Maximum fUT is", maxFUT, "for user", userDateMaxFUT[0], "and date",
      userDateMaxFUT[1])

# Output f_{ut} in the format specified at the top of this script.
fout = open(OUTPUT_F_U_T, "w")

# Output the users and dates in sorted order.
for userDate in sorted(userDateToFUT.keys()):
    userID = userDate[0]
    dateID = userDate[1]
    fUT = int(userDateToFUT[userDate])

    fout.write(str(userID) + DELIMITER)
    fout.write(str(dateID) + DELIMITER)
    fout.write(str(fUT))
    fout.write("\n")

print("\nOutputted f_{ut} file to", OUTPUT_F_U_T)
