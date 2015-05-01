This folder contains cached internal data for TimeSVDPP objects. This is
intended to avoid constantly re-training.

Our mapping from internal variables to files (in this folder) is:
    * bItemConst -> b_item_const.mat
    * bItemTimewise -> b_item_timewise.mat
    * bUserConst -> b_user_const.mat
    * bUserAlpha -> b_user_alpha.mat
    * bUserTime -> b_user_time.mat
    * itemFacMat -> item_fac.mat
    * userFacMat -> user_fac.mat
    * userFacMatAlpha -> user_fac_alpha.mat
    * userFacMatTime -> user_fac_time.dta
    * sumMovieWeights -> user_sum_y.mat
    * yMat -> y.mat
