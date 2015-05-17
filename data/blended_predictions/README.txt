
===========================================================================
INTRODUCTION
===========================================================================

This is where we'll store some of our blended predictions, as well as the
quiz RMSEs they achieved (added on in the same format as in the
"quiz_blend" folder). Additional details on each blend are below.


===========================================================================
ADDITIONAL DETAILS
===========================================================================

BLEND_MAY_17_QRMSE_0.87093: Added a lot of predictors back to the
quiz_blend folder, and ran quiz-blending again. This gave the following
coefficients:
    * SVDPP_FAC_200_EPOCH_40: -0.137
    * SVDPP_FAC_500_EPOCH_40: -0.064
    * TIMESVDPP_FAC_110_EPOCH_40: -0.011
    * TIMESVDPP_FAC_110_EPOCH_80: -0.073
    * TIMESVDPP_FAC_20_EPOCH_40: 0.132
    * TIMESVDPP_NO_UFMT_FAC_100_EPOCH_40: 0.084
    * TIMESVDPP_NO_UFMT_FAC_200_EPOCH_40: 0.129
    * TIMESVDPP_NO_UFMT_FAC_20_EPOCH_40: -0.011
    * TIMESVDPP_NO_UFMT_FAC_500_EPOCH_40: 0.198
    * GLOBALS_KNN_GE_10_MC_24_MW_30: 0.010
    * TIMESVDPP_FAC_60_EPOCH_80: -0.027
    * SVDPP_FAC_100_EPOCH_80: -0.147
    * SVDPP_FAC_1000_EPOCH_80: 0.016
    * TIMESVDPP_FAC_345_EPOCH_40: 0.127
    * TIMESVDPP_FAC_300_EPOCH_80: 0.250
    * SVDPP_FAC_2000_EPOCH_80: 0.280
    * KNN_ON_TIMESVDPP_MC_24_MW_200_FAC_60_EPOCH_40: 0.062
    * KNN_ON_TIMESVDPP_MC_24_MW_400_FAC_60_EPOCH_40: 0.024
    * SVD_FAC_1000_EPOCH_80: 0.037
    * RBM_FAC_200_EPOCH_60: 0.116
    * RBM_FAC_400_EPOCH_60: 0.084
    * SVD_FAC_2000_EPOCH_80: 0.060
    * TIMESVDPP_FAC_200_EPOCH_80: 0.117
    * TIMESVDPP_FAC_100_EPOCH_60: -0.122
    * KNN_ON_TIMESVDPP_MC_30_MW_50_FAC_60_EPOCH_80: 0.071
    * TIMESVDPP_FAC_60_EPOCH_80: -0.176


BLEND_MAY_17_QRMSE_0.87131: After fixing some bugs in the quiz blending
script, I ran a blend on the same predictors as below. This gave the
following coefficients:
    * SVDPP_FAC_200_EPOCH_40: -0.129
    * SVDPP_FAC_500_EPOCH_40: -0.042
    * TIMESVDPP_FAC_110_EPOCH_40: -0.013
    * TIMESVDPP_FAC_110_EPOCH_80: -0.103
    * TIMESVDPP_FAC_20_EPOCH_40: 0.149
    * TIMESVDPP_NO_UFMT_FAC_100_EPOCH_40: 0.047
    * TIMESVDPP_NO_UFMT_FAC_200_EPOCH_40: 0.105
    * TIMESVDPP_NO_UFMT_FAC_20_EPOCH_40: -0.002
    * TIMESVDPP_NO_UFMT_FAC_500_EPOCH_40: 0.187
    * TIMESVDPP_FAC_60_EPOCH_80: -0.046
    * SVDPP_FAC_100_EPOCH_80: -0.160
    * SVDPP_FAC_1000_EPOCH_80: 0.000
    * TIMESVDPP_FAC_345_EPOCH_40: 0.159
    * TIMESVDPP_FAC_300_EPOCH_80: 0.278
    * SVDPP_FAC_2000_EPOCH_80: 0.292
    * SVD_FAC_1000_EPOCH_80: 0.091
    * RBM_FAC_200_EPOCH_60: 0.123
    * RBM_FAC_400_EPOCH_60: 0.088


BLEND_MAY_15_QRMSE_0.87278: This came from a quiz blend of the following
predictors, all of which were trained on the entire dataset. Blending
coefficients have been included as well.
    * SVDPP_FAC_200_EPOCH_40: -0.113
    * SVDPP_FAC_500_EPOCH_40: -0.124
    * TIMESVDPP_FAC_110_EPOCH_40: -0.107
    * TIMESVDPP_FAC_110_EPOCH_80: -0.044
    * TIMESVDPP_FAC_20_EPOCH_40: 0.259
    * TIMESVDPP_NO_UFMT_FAC_100_EPOCH_40: 0.077
    * TIMESVDPP_NO_UFMT_FAC_200_EPOCH_40: 0.087
    * TIMESVDPP_NO_UFMT_FAC_20_EPOCH_40: -0.014
    * TIMESVDPP_NO_UFMT_FAC_500_EPOCH_40: 0.081
    * TIMESVDPP_FAC_60_EPOCH_80: -0.018
    * SVDPP_FAC_100_EPOCH_80: -0.090
    * SVDPP_FAC_1000_EPOCH_80: 0.102
    * TIMESVDPP_FAC_345_EPOCH_40: -0.214
    * TIMESVDPP_FAC_300_EPOCH_80: 0.541
    * SVDPP_FAC_2000_EPOCH_80: 0.174
    * SVD_FAC_1000_EPOCH_80: 0.057
    * RBM_FAC_200_EPOCH_60: 0.165
    * RBM_FAC_400_EPOCH_60: 0.180

The overfitted SVD and the RBMs seem to be improving the blend noticeably.
I also removed the old 200-factor RBM that was trained on just base.


BLEND_MAY_14_QRMSE_0.87358: This came from a quiz blend of the following
predictors. All but the RBM were trained on the entire dataset:
    * RBM_FAC_200_EPOCH_36_QRMSE_0.91109.dta
    * SVDPP_FAC_1000_EPOCH_80_QRMSE_0.88479.dta
    * SVDPP_FAC_100_EPOCH_80_QRMSE_0.88763.dta
    * SVDPP_FAC_2000_EPOCH_80_QRMSE_0.88415.dta
    * SVDPP_FAC_200_EPOCH_40_QRMSE_0.88698.dta
    * SVDPP_FAC_500_EPOCH_40_QRMSE_0.88639.dta
    * TIMESVDPP_FAC_110_EPOCH_40_QRMSE_0.87752.dta
    * TIMESVDPP_FAC_110_EPOCH_80_QRMSE_0.87817.dta
    * TIMESVDPP_FAC_20_EPOCH_40_QRMSE_0.88668.dta
    * TIMESVDPP_FAC_300_EPOCH_80_QRMSE_0.88275.dta
    * TIMESVDPP_FAC_345_EPOCH_40_QRMSE_0.87629.dta
    * TIMESVDPP_FAC_60_EPOCH_80_QRMSE_0.87924.dta
    * TIMESVDPP_NO_UFMT_FAC_100_EPOCH_40_QRMSE_0.8796.dta
    * TIMESVDPP_NO_UFMT_FAC_200_EPOCH_40_QRMSE_0.8789.dta
    * TIMESVDPP_NO_UFMT_FAC_20_EPOCH_40_QRMSE_0.88863.dta
    * TIMESVDPP_NO_UFMT_FAC_500_EPOCH_40_QRMSE_0.87834.dta

The RBM improved prediction performance, but could use additional
tweaking.


BLEND_MAY_14_QRMSE_0.87374: This came from a quiz blend of the following
predictors, all of which were trained on the entire dataset:
    * KNN_ON_TIMESVDPP_MC_24_MW_200_FAC_60_EPOCH_40_QRMSE_0.88249.dta
    * SVDPP_FAC_1000_EPOCH_80_QRMSE_0.88479.dta
    * SVDPP_FAC_100_EPOCH_80_QRMSE_0.88763.dta
    * SVDPP_FAC_2000_EPOCH_80_QRMSE_0.88415.dta
    * SVDPP_FAC_200_EPOCH_40_QRMSE_0.88698.dta
    * SVDPP_FAC_500_EPOCH_40_QRMSE_0.88639.dta
    * TIMESVDPP_FAC_110_EPOCH_40_QRMSE_0.87752.dta
    * TIMESVDPP_FAC_110_EPOCH_80_QRMSE_0.87817.dta
    * TIMESVDPP_FAC_20_EPOCH_40_QRMSE_0.88668.dta
    * TIMESVDPP_FAC_300_EPOCH_80_QRMSE_0.88275.dta
    * TIMESVDPP_FAC_345_EPOCH_40_QRMSE_0.87629.dta
    * TIMESVDPP_FAC_60_EPOCH_80_QRMSE_0.87924.dta
    * TIMESVDPP_NO_UFMT_FAC_100_EPOCH_40_QRMSE_0.8796.dta
    * TIMESVDPP_NO_UFMT_FAC_200_EPOCH_40_QRMSE_0.8789.dta
    * TIMESVDPP_NO_UFMT_FAC_20_EPOCH_40_QRMSE_0.88863.dta
    * TIMESVDPP_NO_UFMT_FAC_500_EPOCH_40_QRMSE_0.87834.dta

Note that the new, 2000-factor SVD++ is contributing a noticeable
improvement. The old 1000-factor SVD++ is not contributing as much now.


BLEND_MAY_14_QRMSE_0.87393: This came from a quiz blend of the following
predictors, all of which were trained on the entire dataset:
    * KNN_ON_TIMESVDPP_MC_24_MW_200_FAC_60_EPOCH_40_QRMSE_0.88249.dta
    * SVDPP_FAC_1000_EPOCH_80_QRMSE_0.88479.dta
    * SVDPP_FAC_100_EPOCH_80_QRMSE_0.88763.dta
    * SVDPP_FAC_200_EPOCH_40_QRMSE_0.88698.dta
    * SVDPP_FAC_500_EPOCH_40_QRMSE_0.88639.dta
    * TIMESVDPP_FAC_110_EPOCH_40_QRMSE_0.87752.dta
    * TIMESVDPP_FAC_110_EPOCH_80_QRMSE_0.87817.dta
    * TIMESVDPP_FAC_20_EPOCH_40_QRMSE_0.88668.dta
    * TIMESVDPP_FAC_300_EPOCH_80_QRMSE_0.88275.dta
    * TIMESVDPP_FAC_345_EPOCH_40_QRMSE_0.87629.dta
    * TIMESVDPP_FAC_60_EPOCH_80_QRMSE_0.87924.dta
    * TIMESVDPP_NO_UFMT_FAC_100_EPOCH_40_QRMSE_0.8796.dta
    * TIMESVDPP_NO_UFMT_FAC_200_EPOCH_40_QRMSE_0.8789.dta
    * TIMESVDPP_NO_UFMT_FAC_20_EPOCH_40_QRMSE_0.88863.dta
    * TIMESVDPP_NO_UFMT_FAC_500_EPOCH_40_QRMSE_0.87834.dta 

The new 300-factor, overfitted Time-SVD++ significantly changed the
characteristics of the blend, which was interesting since it's QRMSE was
fairly low.


BLEND_MAY_13_QRMSE_0.87418: This came from a quiz blend of the following
predictors, all of which were trained on the entire dataset:
    * SVDPP_FAC_1000_EPOCH_80_QRMSE_0.88479.dta
    * SVDPP_FAC_100_EPOCH_80_QRMSE_0.88763.dta
    * SVDPP_FAC_200_EPOCH_40_QRMSE_0.88698.dta
    * SVDPP_FAC_500_EPOCH_40_QRMSE_0.88639.dta
    * TIMESVDPP_FAC_110_EPOCH_40_QRMSE_0.87752.dta
    * TIMESVDPP_FAC_110_EPOCH_80_QRMSE_0.87817.dta
    * TIMESVDPP_FAC_20_EPOCH_40_QRMSE_0.88668.dta
    * TIMESVDPP_FAC_60_EPOCH_80_QRMSE_0.87924.dta
    * TIMESVDPP_NO_UFMT_FAC_100_EPOCH_40_QRMSE_0.8796.dta
    * TIMESVDPP_NO_UFMT_FAC_200_EPOCH_40_QRMSE_0.8789.dta
    * TIMESVDPP_NO_UFMT_FAC_20_EPOCH_40_QRMSE_0.88863.dta
    * TIMESVDPP_NO_UFMT_FAC_500_EPOCH_40_QRMSE_0.87834.dta

The biggest contributor to this blend was the 1000-factor overfitted SVD++
prediction; it had a weight of around 59%. The next biggest contributor was
TIMESVDPP_FAC_20_EPOCH_40, interestingly enough.


BLEND_MAY_11_QRMSE_0.87552: This came from a quiz blend of the following
qual prediction data, all of which were trained on the entire dataset:
    * SVDPP_FAC_200_EPOCH_40_QRMSE_0.88698.dta
    * SVDPP_FAC_500_EPOCH_40_QRMSE_0.88639.dta
    * TIMESVDPP_FAC_110_EPOCH_40_QRMSE_0.87752.dta
    * TIMESVDPP_FAC_110_EPOCH_80_QRMSE_0.87817.dta
    * TIMESVDPP_FAC_20_EPOCH_40_QRMSE_0.88668.dta
    * TIMESVDPP_FAC_60_EPOCH_40_QRMSE_0.87911.dta
    * TIMESVDPP_NO_UFMT_FAC_100_EPOCH_40_QRMSE_0.8796.dta
    * TIMESVDPP_NO_UFMT_FAC_200_EPOCH_40_QRMSE_0.8789.dta
    * TIMESVDPP_NO_UFMT_FAC_20_EPOCH_40_QRMSE_0.88863.dta
    * TIMESVDPP_NO_UFMT_FAC_500_EPOCH_40_QRMSE_0.87834.dta 

The improvement over the previous set of blended predictions is only
marginal. Possible reason: negative coefficients were produced for
TIMESVDPP_NO_UFMT_FAC_200_EPOCH_40, TIMESVDPP_NO_UFMT_FAC_100_EPOCH_40
(huge negative coefficient), and TIMESVDPP_FAC_60_EPOCH_40. Perhaps these
didn't work as well with the other data...


BLEND_MAY_07_QRMSE_0.87651: This came from a quiz blend that added on to the
previous blend. The following qual prediction data was used:
    * SVDPP_FAC_200_EPOCH_25_QRMSE_0.89204.dta
    * TIMESVDPP_FAC_110_EPOCH_25_QRMSE_0.87768.dta (new)
    * TIMESVDPP_FAC_130_EPOCH_30_QRMSE_0.88515.dta
    * TIMESVDPP_FAC_500_EPOCH_30_QRMSE_0.88572.dta


Note that only TIMESVDPP_FAC_110_EPOCH_25_QRMSE_0.87768.dta was trained on
the entire dataset, while the others were trained on just base, hidden, and
valid. Also, TIMESVDPP_FAC_130_EPOCH_30_QRMSE_0.88515.dta actually had a
negative weight in this blend (indicating possible issues with blending too
many Time-SVD++ models with similar factors?)


BLEND_MAY_03_QRMSE_0.88427: This came from a quiz blend (that I (Laksh)
tried as a test) on the following qual prediction data:
    * SVDPP_FAC_200_EPOCH_25_QRMSE_0.89204.dta
    * TIMESVDPP_FAC_130_EPOCH_30_QRMSE_0.88515.dta
    * TIMESVDPP_FAC_500_EPOCH_30_QRMSE_0.88572.dta

Note that none of these data files were trained on all of the dataset, so
the quiz blend didn't give as good of a result as one would expect.


