from libcpp.string cimport string

cdef extern from "svdpp.hh":
    cdef cppclass SVDPP:
        
        # Constructor without cached files.
        SVDPP(int numUsers, int numItems, float meanRating,
              int numFactors, int numIterations,
              const string &fileNameN)

        # Constructor with cached files.
        SVDPP(int numUsers, int numItems, float meanRating,
              int numFactors, int numIterations, const string &fileNameN,
              const string &fileNameBUser, const string &fileNameBItem,
              const string &fileNameUserFacMat,
              const string &fileNameItemFacMat,
              const string &fileNameYMat,
              const string &fileNameSumMovieWeights)
        
        void train(const string &fileNameData)
        void trainAndCache(const string &fileNameData,
                           const string &fileNameBUser,
                           const string &fileNameBItem,
                           const string &fileNameUserFacMat,
                           const string &fileNameItemFacMat,
                           const string &fileNameYMat,
                           const string &fileNameSumMovieWeights)
    
        float predict(int user, int item, int date)


# Cython wrapper class for SVDPP.
cdef class PySVDPP:

    # The C++ instance being wrapped
    cdef SVDPP *obj


    # We can't overload __cinit__, so we just have it take all of the
    # parameters needed. If the file names (beyond that for N.dta) are
    # None, then we use the non-caching constructor.
    def __cinit__(self, int numUsers, int numItems, float meanRating,
                  int numFactors, int numIterations,
                  fileNameN, fileNameBUser = None,
                  fileNameBItem = None, fileNameUserFacMat = None,
                  fileNameItemFacMat = None, fileNameYMat = None,
                  fileNameSumMovieWeights = None):
        
        if fileNameBUser == None or fileNameBItem == None or \
            fileNameUserFacMat == None or fileNameItemFacMat == None or \
            fileNameYMat == None or fileNameSumMovieWeights == None:

            self.obj = new SVDPP(numUsers, numItems, meanRating,
                                 numFactors, numIterations, fileNameN)

        else:
            
            self.obj = new SVDPP(numUsers, numItems, meanRating,
                                 numFactors, numIterations, fileNameN,
                                 fileNameBUser, fileNameBItem,
                                 fileNameUserFacMat, fileNameItemFacMat,
                                 fileNameYMat, fileNameSumMovieWeights)
    
    
    def __dealloc__(self):
        del self.obj
    
    
    def train(self, string fileNameData):
        self.obj.train(fileNameData)


    def trainAndCache(self, string fileNameData,
                      string fileNameBUser, string fileNameBItem,
                      string fileNameUserFacMat, string fileNameItemFacMat,
                      string fileNameYMat, string fileNameSumMovieWeights):
        self.obj.trainAndCache(fileNameData, fileNameBUser, fileNameBItem,
                               fileNameUserFacMat, fileNameItemFacMat,
                               fileNameYMat, fileNameSumMovieWeights)
    
    
    def predict(self, int user, int item, int date):
        return self.obj.predict(user, item, date)
