""" This script takes the different tol_grid grid search dictionaries and
    puts each method into its own dictionary for ease of data visualization.

    Parameters
    ----------
        method_name : str
            The name of the method to zip into a dictionary. This is the begining
            part of the names of the individual dictionaries. e.g - 'tols_svd'
            would put together all the files of the form 'tols_svd_[.*]'.
        
        output_name : str
            The name of the file to output to. This is the file that contains
            the pickled 'stitched together' dictionary.

    Outputs
    -------
        output_name.pkl : Pickle object containing the dictionary of the data
                          consolidated into one dictionary.
"""
import pickle
import sys
import numpy as np
import os
from glob import glob
import re

if __name__ == '__main__':
    # Get the variables from argv
    if len(sys.argv) != 3:
        raise Exception("'zip_dicts.py' requires 2 arguments -- method name and output file name.")
    method_name = sys.argv[1]
    output_name = sys.argv[2]

    # Get the file names
    file_names = sorted(glob('longertimetol_' + method_name + "*"))

    print(file_names)
    dict_list = list()
    degs = list()
    # Compile regex for finding the degrees
    deg_pattern = re.compile('.*_([0-9]+)_cond\.pkl')

    # Read each of the individual file dictionaries and store
    # Also get the degree list
    for file_name in file_names:
        dict_list.append(np.load(file_name, allow_pickle=True))
        degs.append(int(re.findall(deg_pattern, file_name)[0]))

    # Put the first dictionary into the final dict
    final_dict = {deg:dictionary for deg, dictionary in zip(degs, dict_list)}

    # Format the output name properly
    if output_name[-4:] != '.pkl':
        output_name += '.pkl'

    # Pickle the final, consilidated dictionary
    with open(output_name, 'w+b') as f:
        pickle.dump(final_dict, f, pickle.HIGHEST_PROTOCOL)

    print("Process executed successfully. See the results in {}.".format(output_name))
