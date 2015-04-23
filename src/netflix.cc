#include <fstream>
#ifndef NDEBUG
#include <iostream>
#endif

#include <netflix.hh>

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
    void splitIntoInts(const std::string &str, const std::string &delimiter,
                       std::vector<int> &output)
    {
        std::string::size_type start = 0;
        std::string::size_type delimPos = str.find(delimiter);
        std::string::size_type length = 0;

        // Keep adding elements to the vector as long as the delimiter is found.
        while (delimPos != std::string::npos)
        {
            length = delimPos - start;
            
            // Convert the substring to an integer and add to the vector.
            output.push_back(std::stoi(str.substr(start, length)));
            
            // Go look for the next delimiter after this last one.
            start = delimPos + 1;
            delimPos = str.find(delimiter, start);
            
            // If there are no more delimiters left, add in the remainder of
            // the string (from "start" to the end of the string).
            if (delimPos == std::string::npos)
            {
                output.push_back(std::stoi(str.substr(start, str.length())));
            }
        }
    }

    fmat parseData(const std::string &indexPath, const std::string &dataPath, 
                   const std::set<int> &indices) {
        // Open the index file
        std::ifstream indexFile(indexPath);

        if (indexFile.fail())
        {
            throw std::runtime_error("Couldn't find index file at " +
                                     indexPath);
        }

        // Line buffer
        std::string line;
        // Line count
        int lines = 0;
        // Count the number of lines in the data file
        while ( std::getline(indexFile, line) != 0 ) ++lines;

        // Clear the eofbit
        indexFile.clear();
        // Reset the index file's stream to the beginning of the file
        indexFile.seekg(0, indexFile.beg);

        // Open the data file
        std::ifstream dataFile(dataPath);
        
        if (dataFile.fail())
        {
            throw std::runtime_error("Couldn't find data file at " +
                                     dataPath);
        }
        
        // Armadillo matrix for storing data
        fmat data(COLUMNS, lines, fill::zeros);
        // Start loading the 0th column
        int col = 0;
        // Temporary variables for reading index & data of a row
        int index, user, movie, date;
        float rating;
        // For each row, read the index
        while ( indexFile >> index ) {
            // Get the data for this row
            dataFile >> user >> movie >> date >> rating;
            // Skip the row if its index is not in our set of desired indices
            if ( indices.count(index) == 0 ) continue;
            // Store it in our data matrix as floats
            data.at(USER_ROW, col) = (float) user;
            data.at(MOVIE_ROW, col) = (float) movie;
            data.at(DATE_ROW, col) = (float) date;
            data.at(RATING_ROW, col) = rating;
            // Write to the next column
            ++col;
            
#ifndef NDEBUG
            if (col % 1000000 == 0)
            {
                std::cout << "Finished adding entry " << col << "." <<
                    std::endl;
            }
#endif

        }
#ifndef NDEBUG
        std::cout << col << " columns added to data" << std::endl;
#endif
        // Remove unused columns
        data.shed_cols(col, lines - 1);

        indexFile.close();
        dataFile.close();

        return data;
    }

}
