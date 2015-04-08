# Helper function to get global mean
# Total lines processed: 99666408
# Sum of all ratings: 359747514
# Global average movie rating: 3.60951619727

from numpy import *

# Set up
# Does not matter if we choose mu or um data
idx = open('../../data/um/all.idx', 'r')
total, n = 0.0, 0

# Iterate through the data file one line at a time
with open('../../data/um/all.dta', 'r') as f:
    for s in f:
        # Determine if the index is valid
        index = int(idx.next().strip())
        if index > 4:
            continue

        l = s.strip().split()

        # Sum to total
        total += int(l[3])
        n += 1

        # Print progress in terminal
        if ((n % 1000000 == 0)):
            print 'Finished processing line ' + str(n)

#  Close file not opened by with open...
idx.close()

print 'Total lines processed: ' + str(n) # 99666408
print 'Sum of all ratings: ' + str(total) # 359747514
print 'Global average movie rating: ' + str(total / n) # 3.60951619727
