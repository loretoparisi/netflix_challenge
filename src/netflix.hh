/**
 * This file contains various constants and convenience functions related
 * to the Netflix challenge and the format of our data. All of these have
 * been gathered into a "netflix" namespace that should be used by other
 * files.
 *
 * Note that we're using data files where entries are in (user, movie)
 * order, **not** (movie, user) order.
 *
 */

#ifndef NETFLIX_NAMESPACE_HH
#define NETFLIX_NAMESPACE_HH

#include <armadillo>
#include <array>
#include <set>
#include <string>
#include <vector>

using namespace arma;

namespace netflix
{
    /* Constants */

    // Minimum and maximum possible ratings in this dataset.
    constexpr int MIN_RATING = 1;
    constexpr int MAX_RATING = 5;

    // Represents an entry with no rating (or rather, an unknown rating).
    constexpr int NO_RATING = 0;

    // The total number of users, movies, and dates in the dataset.
    constexpr int NUM_USERS = 458293;
    constexpr int NUM_MOVIES = 17770;
    constexpr int NUM_DATES = 2243;
    
    // The mean rating in the training set.
    constexpr float MEAN_RATING_TRAINING_SET = 3.60951619727;
    
    // Name of the file containing all of the data. Note that we are using
    // the version where user IDs, item IDs, and time IDs are all
    // zero-indexed. 
    const std::string DATA_PATH = "data/um/new_all.dta";
    const std::string DATA_PATH_MU = "data/mu/new_all.dta";
    
    // Name of the file containing corresponding set indexes for "all.dta"
    // (and "new_all.dta" too).
    const std::string INDEX_PATH = "data/um/all.idx";
    const std::string INDEX_PATH_MU = "data/mu/all.idx";
    
    // Name of the file containing "qual" set data only. The user IDs, item
    // IDs, and time IDs in this have also been zero-indexed.
    const std::string QUAL_DATA_FN = "data/um/new_qual.dta";
    
    // Name of the file containing the "N" matrix.
    const std::string N_FN = "data/N.dta";
    
    // Name of the file containing the hat{dev_u(t)} mapping.
    const std::string HAT_DEV_U_T_FN = "data/hat_dev_u_t.dta";

    // These indices represent the different kinds of data in all.dta (and
    // in "new_all.dta" too).
    constexpr int BASE_SET = 1;
    constexpr int VALID_SET = 2;
    constexpr int HIDDEN_SET = 3;
    constexpr int PROBE_SET = 4;
    constexpr int QUAL_SET = 5;
    
    // Various subsets of the training dataset (i.e. not including
    // "qual").
    const std::set<int> BASE_IDX                = {BASE_SET};
    const std::set<int> HIDDEN_IDX              = {HIDDEN_SET};
    const std::set<int> VALID_IDX               = {VALID_SET};
    const std::set<int> PROBE_IDX               = {PROBE_SET};
    const std::set<int> BASE_HIDDEN_IDX         = {BASE_SET, HIDDEN_SET};
    const std::set<int> BASE_HIDDEN_VALID_IDX   = {BASE_SET, HIDDEN_SET,
                                                   VALID_SET};
    const std::set<int> ALL_TRAIN_IDX           = {BASE_SET, HIDDEN_SET,
                                                   VALID_SET, PROBE_SET};


    // These are the file names of where we're storing various subsets of
    // the dataset in Armadillo's binary format (as fmats). We don't
    // include "qual" among these subsets since qual should not be used for
    // testing. The contents of these matrices should be self-explanatory.
    // Run the helper code in "binarize_data.cc" to create them once.
    const std::string BASE_BIN                  = "data/um/base.mat";
    const std::string HIDDEN_BIN                = "data/um/hidden.mat";
    const std::string VALID_BIN                 = "data/um/valid.mat";
    const std::string PROBE_BIN                 = "data/um/probe.mat";
    const std::string BASE_HIDDEN_BIN           = "data/um/base_"
                                                  "hidden.mat";
    const std::string BASE_HIDDEN_VALID_BIN     = "data/um/base_hidden_"
                                                  "valid.mat";
    const std::string ALL_TRAIN_BIN             = "data/um/base_hidden_valid_"
                                                  "probe.mat";
    const std::string MU_BASE_BIN               = "data/base-mu.mat";
    
    // The number of columns in the data files (not including qual).
    constexpr int COLUMNS = 4;

    // These are row indices in the data matrix (passed to MLAlgorithm::train)
    constexpr int USER_ROW = 0;
    constexpr int MOVIE_ROW = 1;
    constexpr int DATE_ROW = 2;
    constexpr int RATING_ROW = 3;

    // The delimiter used in our data files (e.g. in the data file containing
    // N).
    const std::string DELIMITER = " ";
    
    /* Convenience functions */
    void splitIntoInts(const std::string &str, const std::string &delimiter,
                       std::vector<int> &output);
    fmat parseData(const std::string &indexPath, const std::string &dataPath, 
                   const std::set<int> &indices);
    
    /**
     * Rounds a float to an int without truncating. Used to convert user
     * IDs, item IDs, and date IDs in our fmats into integers. For
     * instance, an item ID of 16.9999 really should be recorded as item ID
     * 17, not item ID 16.
     *
     */
    inline int roundToInt(float x)
    {
        return int(x + 0.5);
    }
}


#endif // NETFLIX_NAMESPACE_HH
