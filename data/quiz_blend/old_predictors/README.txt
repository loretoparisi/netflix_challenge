This is a folder that contains old predictors' qual data. We no longer want
to use these predictors for quiz blending, which is why we've put them here
instead.

===========================================================================
OLD PREDICTIONS
===========================================================================


TIME-SVD++ (overfit)
---------------------


TIME-SVD++
-----------

TIMESVDPP_FAC_240_EPOCH_40_QRMSE_0.87662: See TIMESVDPP_QUAL_7.860 in
"good_predictions". This run did use userFacMatTime. 

TIMESVDPP_FAC_60_EPOCH_40_QRMSE_0.87911: See TIMESVDPP_QUAL_7.598 in
"good_predictions". This run did use userFacMatTIme.

TIMESVDPP_FAC_110_EPOCH_25_QRMSE_0.87768: This is an extension to
SVD++^(3), with frequency-dependent biases and extra bin-wise item factors.
This was generated using the parameters specified in "good_predictions" for
TIMESVDPP_QUAL_7.749.dta. Note that this predictor has been trained on
probe as well, and did use userFacMatTime.

TIMESVDPP_FAC_130_EPOCH_30_QRMSE_0.88515: this is SVD++^(3), generated
using the parameters specified in "good_predictions" for
TIMESVDPP_QUAL_6.963.dta. Note that this has not been trained on probe.

TIMESVDPP_FAC_500_EPOCH_30_QRMSE_0.88572: this is SVD++^(1), generated
using the parameters specified in "good_predictions" for
TIMESVDPP_QUAL_6.904.dta. Note that this has not been trained on probe.


SVD++
------

SVDPP_FAC_60_EPOCH_40_QRMSE_0.88947: See SVDPP_QUAL_6.509 in
"good_predictions".

SVDPP_FAC_200_EPOCH_25_QRMSE_0.89204: this is SVD++, generated using the
parameters specified in good_predictions.dta for SVDPP_QUAL_6.239.dta.
Note that this has not been trained on probe.


SVD (overfit)
--------------

SVD_FAC_2000_EPOCH_80_QRMSE_0.89078: See SVD_QUAL_6.372 in
"good_predictions".


Regular RBM (UToronto)
-----------------------

RBM_FAC_200_EPOCH_36_QRMSE_0.91109: See RBM_QUAL_4.237 in
"good_predictions".


Residual kNN
-------------

GLOBALS_KNN_GE_6_MC_30_MW_80_QRMSE_0.94725: For more details, see
KNN_ON_GLOBALS_QUAL_0.436 in "good_predictions".

KNN_ON_TIMESVDPP_MC_24_MW_200_FAC_30_EPOCH_80_QRMSE_0.90103: For more
details, see KNN_ON_TIMESVDPP_QUAL_5.294 in "good_predictions".

KNN_ON_TIMESVDPP_MC_30_MW_50_FAC_30_EPOCH_80_QRMSE_0.90389: For more
details, see KNN_ON_TIMESVDPP_QUAL_4.994 in "good_predictions".

KNN_ON_TIMESVDPP_MC_30_MW_50_FAC_60_EPOCH_80_QRMSE_0.89256: For more
details, see KNN_ON_TIMESVDPP_QUAL_6.185 in "good_predictions".

KNN_ON_TIMESVDPP_MC_24_MW_200_FAC_60_EPOCH_40_QRMSE_0.88249: For more
details, see KNN_ON_TIMESVDPP_QUAL_7.243 in "good_predictions".
