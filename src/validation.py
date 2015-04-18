#!/usr/bin/env python3

# This script carries out parameter tuning on machine learning algorithms,
# using the following method(s):
#   * Regular validation on a set of parameters over a range of values.
#   * APT2
#
# TODO: Describe usage.

from interface import *

# Laksh: I'll work on this later, hopefully when there are more algorithms
# we've written and can actually validate on.

# TODO: Expand algorithms' constructors (mostly SVD++ right now?) so that
# they take constants as constructor parameters (for non-caching
# constructors). Also change the wrapper classes so that they take **kwargs
# that can be properly interpreted by their constructors.

# TODO: For algorithms that use seeds, add a constructor parameter (for the
# non-caching constructors) to set the seed for the purposes of
# cross-validation.

# TODO: Use argparse to take the following command-line args for each run:
#       * --config <CONFIG_FILE>
#
# The config file will determine the algorithm, which tuning method to try,
# the training set indices, the validation set indices, and the relevant
# parameters for that tuning method. For instance, for regular validation,
# you want a range of values and a step size for each parameter. For APT2,
# you want an initial guess for each parameter. If a parameter isn't
# specified, the default value for it is used instead.
#
#
# Example config file contents:
#       SVD++
#       APT2
#       1 2 3 // training set indices
#       4 // validation set index
#       SVDPP_LAM_B_I 0.005
#       SVDPP_GAMMA_B_I 0.003
#       ...
#
# Where each line sets the initial values. Alternatively, if we're using
# regular validation on a set of parameters:
#       SVD++
#       VALID
#       1 2 3 // training set indices
#       4 // validation set index
#       SVDPP_LAM_B_I 0.004 0.006 0.001
#       SVDPP_GAMMA_B_I 0.002 0.004 0.001
#       ...
#
# Where each parameter line tells us the range of values to try (i.e. start
# value, end value, step size).
#


# TODO: Implement parsing the config file.

# TODO: Implement regular validation over a range of values.

# TODO: Implement APT2 for tuning parameters.


svdpp = PySVDPP(1, 2, 3.0, 4, 5, "data/N.dta")
