This directory contains data cached for use by our restricted boltzmann
machines.  The helper program rank_prob (src/helper/rank_prob.cc) outputs the 
per-movie pmf for ratings (P[rating | movie] for all movies) here.  The file is
called rank_prob.mat, and is used to initialize the biases of the visible units 
of our RBM's; see the helper's source for details.
