globals_README
// Compare to result from:
// http://webcache.googleusercontent.com/search?q=cache%3AV7SkSLYh16sJ%3Aalgorithmsanalyzed.blogspot.com%2F2008%2F05%2Fbellkor-algorithm-global-effects.html+&cd=1&hl=en&ct=clnk&gl=us
Probe RMSE trained on BASE (idx == 1 only)
    10 Levels                    Probe RMSE           Alpha
 0: Global Average                (12% below water)    -
 1: Movie Effect                  1.0498               25
 2: User Effect                   0.981093             7
 3: User  * Time(User)            0.978626             550
 4: User  * Time(Movie)           0.978626             150
 5: Movie * Time(Movie)           0.976903             4000
 6: Movie * Time(User)            0.975854             500
 7: User  * Movie Average         0.97187              90
 8: User  * Movie Support         0.969727             90
 9: Movie * User(Average)         0.969552             50
10: Movie * User(Support)         0.966201             50
