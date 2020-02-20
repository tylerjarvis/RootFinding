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

if __name__ == '__main__':
    # Get the variables from argv
    method_name = sys.argv[1]
    output_name = sys.argv[2]

    # Get the file names
    file_names = [method_name + '_[3, 5].pkl', method_name + '_[9, 12, 16].pkl', \
                  method_name + '_[20, 25].pkl']

    dict_list = list()
    # Read each of the individual file dictionaries and store
    for file_name in file_names:
        dict_list.append(np.load(file_name, allow_pickle=True))
    
    # Put the first dictionary into the final dict
    final_dict = {n:dict_list[0][n] for n in range(len(dict_list[0]))}

    # Put the next dictionary into the final dict
    for i in range(len(dict_list[1])):
        final_dict[len(dict_list[0]) + i] = dict_list[1][i]

    # Put the third dictionary into the final dict
    for i in range(len(dict_list[2])):
        final_dict[len(dict_list[0]) + len(dict_list[1]) + i] = dict_list[2][i]

    # Print the final length (Debuging purposes)
    print('The resulting dictionary has the appropriate length: {}' \
           .format(len(final_dict) == sum(len(d) for d in dict_list)))

    # Format the output name properly
    if output_name[-4:] != '.pkl':
        output_name += '.pkl'

    # Pickle the final, consilidated dictionary
    with open(output_name, 'w+b') as f:
        pickle.dump(final_dict, f, pickle.HIGHEST_PROTOCOL)

    print("Process executed successfully. See the results in {}.".format(output_name))