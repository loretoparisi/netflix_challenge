
===========================================================================
INTRODUCTION
===========================================================================

This is where we'll store some of our blended predictions, as well as the
quiz RMSEs they achieved (added on in the same format as in the
"quiz_blend" folder). Additional details on each blend are below.


===========================================================================
ADDITIONAL DETAILS
===========================================================================

BLEND_MAY_14_QRMSE_0.87393: This came from a quiz blend of the following
predictors, all of which were trained on the entire dataset:
    * 

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


