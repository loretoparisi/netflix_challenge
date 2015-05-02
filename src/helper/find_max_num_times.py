#!/usr/bin/env python3

# This helper function just finds the maximum number of distinct days that
# a single user has cast ratings on. This search is carried out across the
# entire dataset.
#
# Output: User 353571 had the largest distinct number of rating dates: 1169
#

# Constants

# The input data file.
ALL_DTA = "../../data/um/new_all.dta"


# A mapping from user ID to a set of days that the user has rated movies
# on.
userToRatingDates = dict()

# Iterate through all entries in the "all" data file, so as to first
# populate userToRatingDates.
with open(ALL_DTA, "r") as allFile:

    for line in allFile:
        thisLineSplit = line.split()

        userID = int(thisLineSplit[0])
        dateID = int(thisLineSplit[2])

        # See if there's already an entry in the dict for this userID.
        if userID in userToRatingDates:
            userToRatingDates[userID].add(dateID)
        else:
            userToRatingDates[userID] = set([dateID])

print("Finished populating userToRatingDates.")

# Find the user with the most distinct rating dates
userMostDistinctRatingDates = -1
mostDistinctRatingDates = 0

for userID in userToRatingDates:
    numDistinctRatingDates = len(userToRatingDates[userID])

    if numDistinctRatingDates > mostDistinctRatingDates:
        userMostDistinctRatingDates = userID
        mostDistinctRatingDates = numDistinctRatingDates

print("\nUser", userMostDistinctRatingDates, "had the largest distinct",
      "number of rating dates:", mostDistinctRatingDates)
