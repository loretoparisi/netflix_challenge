This is a folder that contains old predictors' qual data. We no longer want
to use these predictors for quiz blending, which is why we've put them here
instead.

===========================================================================
OLD PREDICTIONS
===========================================================================


TIME-SVD++ (overfit)
---------------------

TIMESVDPP_FAC_100_EPOCH_60_QRMSE_0.88112: See TIMESVDPP_QUAL_7.387 in
"good_predictions".

TIMESVDPP_NO_UFMT_FAC_750_EPOCH_80_QRMSE_0.87758: See TIMESVDPP_QUAL_7.759
in "good_predictions".

TIMESVDPP_NO_UFMT_FAC_650_EPOCH_80_QRMSE_0.87842: See TIMESVDPP_QUAL_7.671
in "good_predictions".

TIMESVDPP_FAC_150_EPOCH_80_QRMSE_0.88563: See TIMESVDPP_QUAL_6.913 in
"good_predictions".


TIME-SVD++ (not overfit)
-------------------------

TIMESVDPP_NO_UFMT_FAC_100_EPOCH_40_QRMSE_0.8796: See TIMESVDPP_QUAL_7.547
in "good_predictions". This run did not use userFacMatTime.

TIMESVDPP_FAC_110_EPOCH_40_QRMSE_0.87752: See TIMESVDPP_QUAL_7.765 in
"good_predictions". This run did use userFacMatTime.

TIMESVDPP_FAC_240_EPOCH_40_QRMSE_0.87662: See TIMESVDPP_QUAL_7.860 in
"good_predictions". This run did use userFacMatTime. 

TIMESVDPP_FAC_60_EPOCH_40_QRMSE_0.87911: See TIMESVDPP_QUAL_7.598 in
"good_predictions". This run did use userFacMatTIme.

TIMESVDPP_FAC_110_EPOCH_25_QRMSE_0.87768: This is an extension to
SVD++^(3), with frequency-dependent biases and extra bin-wise item factors.
This was generated using the parameters specified in "good_predictions" for
TIMESVDPP_QUAL_7.749.dta. Note that this predictor has been trained on
probe as well, and did use userFacMatTime.


SVD++
------

SVDPP_FAC_200_EPOCH_40_QRMSE_0.88698: See SVDPP_QUAL_6.771 in
"good_predictions".

SVDPP_FAC_60_EPOCH_40_QRMSE_0.88947: See SVDPP_QUAL_6.509 in
"good_predictions".


SVD (overfit)
--------------

SVD_FAC_1000_EPOCH_80_QRMSE_0.89161: See SVD_QUAL_6.284 in
"good_predictions".


SVD (not overfit)
------------------




Regular RBM (UToronto)
-----------------------

RBM_FAC_400_EPOCH_120_QRMSE_0.91224: See RBM_QUAL_4.116
in "good_predictions".


Residual kNN
-------------

KNN_ON_RBM_FAC_50_EPOCH_60_MC_50_MW_800_QRMSE_0.91957: For more details,
see KNN_ON_RBM_QUAL_3.346 in "good_predictions".

KNN_ON_RBM_FAC_400_EPOCH_120_MC_30_MW_50_QRMSE_0.91653: For more details,
see KNN_ON_RBM_QUAL_3.665 in "good_predictions".

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
