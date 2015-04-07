# Helper function to output each movie's offset
# from global mean
from numpy import *

# Set up input and output files
avg = open('../data/user_offset.dta', 'w')
idx = open('../data/um/all.idx', 'r')

# Assume we start with Movie 1
current = 0
total, n = 0.0, 0
temp = 1
GLOBAL_AVG = 3.60951619727 # Computed in global_mean.py
PRECISION = 4 # How many decimal points?

# Iterate through the data file one line at a time
with open('../data/um/all.dta', 'r') as f:
    for s in f:
        # Only use valid indices
        index = int(idx.next().strip())
        if index > 4:
            continue

        l = s.strip().split()

        current = int(l[0])
        # Save result in output file
        if (temp != current):
            result = '{0}\n'.format(round(total / n - GLOBAL_AVG, PRECISION))
            avg.write(result)
            total = 0.0
            n = 0
            temp = current
            # Output to terminal for progress
            if (current % 1000 == 0):
                print "Processed user number " + str(current)

        # Sum to get total
        total += int(l[3])
        n += 1

    # Save result for last movie end of loop
    result = '{0}\n'.format(round(total / n - GLOBAL_AVG, PRECISION ))
    avg.write(result)

# Close files not opened by with open...
idx.close()
avg.close()
