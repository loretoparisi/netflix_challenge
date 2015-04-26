#!/usr/bin/env python3

# This helper function computes hat{dev_u(t)} for a user u's rating at time
# t. First, this involves computing dev_u(t) for each of user u's ratings
# (at different times t), which is given by:
#
#   dev_u(t) = sign(t - t_u) * |t - t_u|^{beta}
#
# Where t_u is the mean date of rating by user u (note that dates in the
# zero-indexed Netflix dataset range from 0 to 2242, where Day 0 is
# Thursday, Nov 11, 1999, and Day 2242 is Saturday, December 31, 2005). The
# constant beta was determined to be 0.4 by BellKor.
#
# I'm assuming that we're using the entire dataset to come up with
# dev_u(t). This is useful since it lets us take into account the user's
# entire viewing history. We also don't have to recompute dev_u(t).
#
# For each user u, we then center all of these time deviations dev_u(t) by
# subtracting off the **mean** dev_u(t) for that user. This gives us
# "centered" values hat{dev_u(t)}.
#
# We then output these centered deviations to a file, which has lines of
# the form:
#
#   <USER ID> <DATE ID> <hat{dev_u(t)} FOR THAT RATING>
#

import numpy as np

# Constants

# Delimiter to use between values in the output file.
DELIMITER = " "

# The exponent used in computing dev_u(t). This is taken from the BellKor
# grand prize paper.
BETA = 0.4

# The input data files.
ALL_DTA = "../../data/um/new_all.dta"

# The file to output to
OUTPUT_HAT_DEV_U_FN = "../../data/hat_dev_u_t.dta"


# A mapping from a user ID to a list of all of the dates that that user has
# rated a movie.
userToRatingDates = dict()

# A mapping from a user ID to the mean rating date t_u for that user.
userToMeanRatingDate = dict()

# A mapping from a user ID to the mean dev_u(t) for that user.
userToMeanDevUT = dict()

# A mapping from a user ID and date ID to dev_u(t) for that rating. This
# will later become a mapping from (user, date) to hat{dev_u(t)}, after we
# subtract off the mean dev_u(t) for that user.
userDateToDevUT = dict()


# Iterate through all entries in the "all" data file, so as to first
# populate userToRatingDates.
with open(ALL_DTA, "r") as allFile:

    for line in allFile:
        thisLineSplit = line.split()

        userID = int(thisLineSplit[0])
        dateID = int(thisLineSplit[2])

        # See if there's already an entry in the dict for this userID.
        if userID in userToRatingDates:
            userToRatingDates[userID].append(dateID)
        else:
            userToRatingDates[userID] = list([dateID])

print("Finished populating userToRatingDates.")

# Get the mean rating date for each user.
for userID in userToRatingDates:
    thisUserRatingDates = userToRatingDates[userID]

    userToMeanRatingDate[userID] = np.mean(thisUserRatingDates)

print("Finished populating userToMeanRatingDate.")
#print(userToMeanRatingDate[2]) # for debugging purposes

# For each user ID and date ID, get dev_u(t) for that rating.
for userID in userToRatingDates:
    # Convert the list of rating dates into a set, since otherwise we'll be
    # repeating work unnecessarily.
    thisUserRatingDates = set(userToRatingDates[userID])
    thisUserMeanRatingDate = userToMeanRatingDate[userID]

    # Iterate over date IDs
    for dateID in thisUserRatingDates:
        userDateToDevUT[(userID, dateID)] = \
                np.sign(dateID - thisUserMeanRatingDate) *\
                (abs(dateID - thisUserMeanRatingDate)**BETA)

print("Finished populating userDateToDevUT.")
#print(userDateToDevUT[(2, 2073)])

# Find the mean dev_u(t) for each user.
for userID in userToRatingDates:
    # Note: this isn't converted into a set since we do want to allow for
    # repeating dates in computing the mean.
    thisUserRatingDates = userToRatingDates[userID]

    # This is where we'll store all of the dev_u(t) values for this user.
    thisUserDevUTs = list()

    for dateID in thisUserRatingDates:
        thisDevUT = userDateToDevUT[(userID, dateID)]
        thisUserDevUTs.append(thisDevUT)

    # Take the mean and store it in userToMeanDevUT
    userToMeanDevUT[userID] = np.mean(thisUserDevUTs)

print("Finished populating userToMeanDevUT.")
#print(userToMeanDevUT[2])

# Subtract off the mean from the entries in userDateToDevUT.
for userID in userToRatingDates:
    # This is a set since we only want to update entries in userDateToDevUT
    # once for each (user, date) tuple.
    thisUserRatingDates = set(userToRatingDates[userID])
    thisUserMeanDevUT = userToMeanDevUT[userID]

    for dateID in thisUserRatingDates:
        currentDevUT = userDateToDevUT[(userID, dateID)]
        newDevUT = currentDevUT - thisUserMeanDevUT

        userDateToDevUT[(userID, dateID)] = newDevUT

print("Centered dev_u(t) values for each user.")
#print(userDateToDevUT[(2, 2073)])

# Output hat{dev_u(t)} in the format specified at the top of this script.
fout = open(OUTPUT_HAT_DEV_U_FN, "w")

# Output the user IDs in sorted order.
for userID in sorted(userToRatingDates.keys()):
    # Use a set for the rating dates for this user, since we don't want to
    # repeat entries in this file. In other words, (user, date) should be a
    # primary key in the output file.
    thisUserRatingDates = set(userToRatingDates[userID])

    # Output the date IDs in sorted order too.
    for dateID in sorted(thisUserRatingDates):
        fout.write(str(userID) + DELIMITER)
        fout.write(str(dateID) + DELIMITER)

        fout.write(str("%.6f" % userDateToDevUT[(userID, dateID)]))
        fout.write("\n")

print("\nOutputted hat{dev_u(t)} file to", OUTPUT_HAT_DEV_U_FN)
