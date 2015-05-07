
===========================================================================
INTRODUCTION
===========================================================================

This is where we'll store some of our blended predictions, as well as the
quiz RMSEs they achieved (added on in the same format as in the
"quiz_blend" folder). Additional details on each blend are below.


===========================================================================
ADDITIONAL DETAILS
===========================================================================

BLEND_MAY_6_QRMSE_0.87658.dta: This came from a quiz blend that added on to
the previous blend. The following qual prediction data was used:
    * SVDPP_FAC_200_EPOCH_25_QRMSE_0.89204.dta
    * TIMESVDPP_FAC_110_EPOCH_25_QRMSE_0.87772.dta (new)
    * TIMESVDPP_FAC_130_EPOCH_30_QRMSE_0.88515.dta
    * TIMESVDPP_FAC_500_EPOCH_30_QRMSE_0.88572.dta

Note that only TIMESVDPP_FAC_110_EPOCH_25_QRMSE_0.87772.dta was trained on
the entire dataset, while the others were trained on just base, hidden, and
valid. Also, TIMESVDPP_FAC_130_EPOCH_30_QRMSE_0.88515.dta actually had a
negative weight in this blend (indicating possible issues with blending too
many Time-SVD++ models with similar factors?)


BLEND_MAY_3_QRMSE_0.88427.dta: This came from a quiz blend (that I (Laksh)
tried as a test) on the following qual prediction data:
    * SVDPP_FAC_200_EPOCH_25_QRMSE_0.89204.dta
    * TIMESVDPP_FAC_130_EPOCH_30_QRMSE_0.88515.dta
    * TIMESVDPP_FAC_500_EPOCH_30_QRMSE_0.88572.dta

Note that none of these data files were trained on all of the dataset, so
the quiz blend didn't give as good of a result as one would expect.


