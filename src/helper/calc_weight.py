import numpy as np
from math import sqrt
# Reference to this article: arxiv.org/pdf/0911.0460.pdf

PROBE = "../../data/probe.dta"
PROBE_SIZE = 1374739

## blender
# Pass in all learned prediction functions and the data used for blending
# (in numpy array format). All prediction functions passed to this function 
# should be in the following format:
# g(idx corresponding to (user_id, movie_id)) = rating. 
def blender(blend_dta, *funcs):
    # Unpack tuple
    funcs = funcs[0]
    X = np.ndarray(shape=(np.shape(blend_dta)[0], len(funcs)))
    y = np.ndarray(shape=(np.shape(blend_dta)[0], 1))

    # Initialize X
    for i in xrange(np.shape(blend_dta)[0]):
        row = blend_dta[i]
        y[i][0] = row
        for j in xrange(len(funcs)):
            X[i][j] = funcs[j](i)

    # Calculate w[i][0]
    # and inv of X
    # BE AWARE: Time consuming!
    X_pinv = np.linalg.pinv(X)
    w = np.dot(X_pinv, y)

    # Print RMSE to output
    for f in xrange(len(funcs)):
        err = 0.0
        for i in xrange(PROBE_SIZE):
            err += (X[i][f] - blend_dta[i])**2

        print "RMSE of ", funcs[f].__name__, " = ", str(sqrt(err/PROBE_SIZE))

    err = 0.0
    for i in xrange(PROBE_SIZE):
        b = 0
        for f in xrange(len(funcs)):
            b += w[f][0] * X[i][f]
        err += (b - blend_dta[i])**2
    
    print "After blending = ", str(sqrt(err/PROBE_SIZE))
    print "======================"

    # Print weights
    for i in xrange(len(funcs)):
        print "w" + funcs[i].__name__ + " = " +  str(w[i][0])

def main():
    probe = open(PROBE, "r")
    blend_dta = np.array([0 for i in xrange(PROBE_SIZE)], dtype=np.uint8)
    i = 0

    # Load blend_dta
    for line in probe:
        l = line.split()
        rating = int(l[3].rstrip())
        blend_dta[i] = rating
        i += 1

    probe.close()

    # Create function's data
    _funct1 = np.array([0 for i in xrange(PROBE_SIZE)], dtype=np.float32)
    # ... more

    # Open probe files according to function
    funct1_probe = open("<probe_file_path>", "r")
    # ... more

    for i in xrange(PROBE_SIZE):
        _funct1[i] = float(funct1_probe.readline().rstrip())
        # ... more

    # Close the files
    funct1_probe.close()
    # ... more

    # Set up access array functions here:
    def funct1(x):
        return _funct1[x]
    # ... more

    # Compile these return functions as a list:
    funcs = [funct1]
    # ... more

    blender(blend_dta, funcs)

if __name__ == '__main__':
    main()
