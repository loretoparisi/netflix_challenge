#!/usr/bin/env python3

# Convert one-indexed to zero-indexed
#
# > import this file
# > usage:
#   conversion(<one-indexed filename>, <zero-indexed filename>)
#   conversion('../../data/um/test.dta', '../../data/um/new_test.dta')

def conversion(filename, output_name):
    with open(filename) as f:
        for line in f:
            first_line = line.strip().split()
            break
    # Make sure that we are not opening an idx file.
    assert len(first_line) > 1
    output = open(output_name, 'w')
    # Counter to output progress
    count = 0
    with open(filename) as input_file:
        for line in input_file:
            curr_line = line.strip().split()
            # First slot is user number, second slot is movie number, and
            # third slot is date number.
            user_num = int(curr_line[0]) - 1
            movie_num = int(curr_line[1]) - 1
            date_num = int(curr_line[2]) - 1
            curr_line[0] = str(user_num)
            curr_line[1] = str(movie_num)
            curr_line[2] = str(date_num)
            output_line = ' '.join(curr_line) + '\n'
            output.write(output_line)
            count += 1
            # Print progress
            if (count % 10000 == 0):
                print('Finished processing line ' + str(count))
    output.close()

if __name__ == "__main__":
    conversion('../../data/um/all.dta', '../../data/um/new_all.dta')