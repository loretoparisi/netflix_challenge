#include "netflix_namespace.hh"

namespace netflix
{
    /* Convenience functions */

    /**
     * This function splits a string around an input delimiter string. The
     * parts of the string between delimiters are converted into ints, and are
     * returned in a vector of ints.
     *
     * @param str: The input string.
     * @param delimiter: The delimiter string that separates data.
     * @param output: A vector containing the ints that were in the original
     *                string, after separating around "delimiter".
     *
     */
    void splitIntoInts(const string &str, const string &delimiter,
                       vector<int>& output)
    {
        string::size_type start = 0;
        string::size_type delimPos = str.find(delimiter);
        string::size_type length = 0;

        // Keep adding elements to the vector as long as the delimiter is found.
        while (delimPos != string::npos)
        {
            length = delimPos - start;
            
            // Convert the substring to an integer and add to the vector.
            output.push_back(stoi(str.substr(start, length)));
            
            // Go look for the next delimiter after this last one.
            start = delimPos + 1;
            delimPos = str.find(delimiter, start);
            
            // If there are no more delimiters left, add in the remainder of
            // the string (from "start" to the end of the string).
            if (delimPos == string::npos)
            {
                output.push_back(stoi(str.substr(start, str.length())));
            }
        }
    }
}
