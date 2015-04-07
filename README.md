# Netflix Challenge
This repository contains our team's Netflix Challenge code for CS 156b (a
Caltech class).

Team name: VC Dementors.  
Team members: Laksh Bhasin, Walker Mills, and Sharon Yang.

**Collaboration policy:** If you are currently taking CS 156b at Caltech,
or are planning on taking it at some point, you *may not* view the contents
of this repository unless you've received express written permission from
one of the team members (listed above). Any violation of this policy will
be deemed a breach of the Caltech Honor Code, and will be handled
accordingly.


Update 4/7/2015
===

Current has a simple solution by each user's average ranking. To create an
output file, modify the constant for input/output file paths, and do:
```
make
./user_avg
```

Helper functions for SVD added as well. This is in helper/. The functions
compute global average and offset of each movie's average from the global
average. Output is in stats/. More to come...
