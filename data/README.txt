This folder contains the data we have used in our project. Note that the
raw data has been hidden from the GitHub repository.

Additional data (add descriptions to this as necessary):
movie_offset.dta     -  This file has 17,770 lines (total number of movies)
                        and contains the offset of each movie's average
                        from the global mean.  Precision set to 4 decimal
                        points. 
user_offset.dta      -  This file has 458,293 lines (total number of users)
                        and contains the offset of each user's average
                        rating from the global mean.  Precision set to 4
                        decimal points. 
N.dta                -  A mapping from user IDs (which have been
                        zero-indexed) to the item IDs (also zero-indexed)
                        that that user has left implicit feedback for
                        (specifically, by rating those items but giving an
                        unknown rating).
svdpppredictions.dta -  The SVD++ rating predictions for all the entries in
                        the "qual" dataset.
