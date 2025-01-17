#!/usr/bin/env python3

# This script creates the N(u) data file for all users in the dataset. For
# a given user u, N(u) is defined as the set of items that "u" showed an
# implicit preference for. In the context of the Netflix challenge, this is
# the set of movies that the user voted for across *all* datasets.
#
# Each line of the output file will contain the following information:
#
#       <user ID> <item 0 in N(u)> ... <item X in N(u)>
#
# Where all of these numbers are integers, and we use a delimiter of " ".
#


# Constants

# Delimiter to use between values
DELIMITER = " "

# The input data files.
ALL_DTA = "../../data/um/new_all.dta"

# The file to output to
OUTPUT_N_FILE_NAME = "../../data/N.dta"


# A mapping from a user ID to a list of items in "all" that that user has
# voted on (and hence has some implicit opinion on).
userToImpPrefItems = dict()

# Iterate through all entries in the "all" data file.
with open(ALL_DTA, "r") as allFile:

    for line in allFile:
        thisLineSplit = line.split()

        userID = int(thisLineSplit[0])
        itemID = int(thisLineSplit[1])

        # See if there's already an entry in the dict for this userID.
        if userID in userToImpPrefItems:
            userToImpPrefItems[userID].append(itemID)
        else:
            userToImpPrefItems[userID] = list([itemID])


# Output the dict to file in the format specified.
fout = open(OUTPUT_N_FILE_NAME, "w")

# Output the user IDs in sorted order.
for userID in sorted(userToImpPrefItems.keys()):

    # The first thing to output is the user ID
    fout.write(str(userID) + DELIMITER)

    # Output the items in N(u)
    impPrefItems = userToImpPrefItems[userID]

    for ind in range(len(impPrefItems)):
        itemID = impPrefItems[ind]

        if ind != len(impPrefItems) - 1:
            fout.write(str(itemID) + DELIMITER)
        else:
            fout.write(str(itemID))

    fout.write("\n")


print("Outputted N data file to", OUTPUT_N_FILE_NAME)
