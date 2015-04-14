## Destination for our blended result
DEST = "../../data/blended_output.dta"
QUAL_SIZE = 2749898

## Open output files here:
x = open("<filename>", "r")
## y = ...

## Create a list of output files:
files = [x]

## Write out to:
out = open(DEST, "w")

## Weight calculated by calc_weight.py
w_x = 0.20
## w_y = ...

## Create a list of weights:
weights = [w_x]

## Checkpoint
assert(len(weights) == len(files))

for i in xrange(QUAL_SIZE):
    j = 0
    k = 0
    for f in files:
        val = f.readline().rstrip()
        k += float(val) * weights[j]
        j += 1
    k = round(k, 4)
    out.write(str(k) + "\n")

## Close the opened files:
x.close()
