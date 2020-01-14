from itertools import product
import grid_search2
import numpy as np 

"(rel_approx_tol, abs_approx_tol, trim_zero_tol, max_cond_num, macaulay_zero_tol, good_zeros, good_zero_factor)"
rel_approx_tol = [10.**-i for i in range(4,8)] # 4
abs_approx_tol = [10.**-i for i in range(9,14)] # 5
trim_zero_tol = [10.**-i for i in range(9,14)] # 5
max_cond_num = [10.**i for i in range(5,11)] # 6
macaulay_zero_tol = [10.**-i for i in range(9,14)] # 5
good_zeros_tol = [10.**-i for i in range(4,7)] # 3
# good_zero_factor = [i for i in range(100,210,10)] # 10
good_zero_factor = [100] # 1

# Temp tols to test.
rel_approx_tol = [10.**-i for i in range(8,10)] # 4
abs_approx_tol = [10.**-i for i in range(12,14)] # 5
trim_zero_tol = [10.**-i for i in range(10,11)] # 5
max_cond_num = [10.**i for i in range(6,7)] # 6
macaulay_zero_tol = [10.**-i for i in range(12,13)] # 5
good_zeros_tol = [10.**-i for i in range(5,6)] # 3
# good_zero_factor = [i for i in range(100,210,10)] # 10
good_zero_factor = [100] # 1

tols_to_test = [rel_approx_tol, abs_approx_tol, trim_zero_tol,
                max_cond_num, macaulay_zero_tol, good_zeros_tol, 
                good_zero_factor]

total_tols_to_test = np.prod([len(t) for t in tols_to_test])


scores = list()
possible_tols = list(product(rel_approx_tol, abs_approx_tol, trim_zero_tol, 
                     max_cond_num, macaulay_zero_tol, good_zeros_tol, 
                     good_zero_factor))
with open('grid_search_results.txt', 'a') as fi:
    i = 0
    fi.write('\n')
    try:
        for tol_set in possible_tols:
            score = grid_search2.get_score(*tol_set)
            scores.append(score)
            i += 1
            fi.write(str((i/total_tols_to_test, score, tol_set)) + '\n')

    finally:
        max_score = np.max(scores)

        best_tols_indices = np.array(possible_tols)[np.where(np.array(scores) == max_score)[0]]
        import time

        # TODO Report the date and time that this is done. Append to the end 
        # of the file.
        fi.write("=========================================================\n")
        fi.write("Searched through " + str(i) + " possibilities out of " + \
                str(total_tols_to_test) + '\n')
        fi.write("Top score: " + str(max_score))
        fi.write("\nTolerances where we scored that:\n(rel_approx_tol," + 
                " abs_approx_tol, trim_zero_tol, max_cond_num," + 
                " macaulay_zero_tol, min_good_zeros_tol, good_zero_factor)\n")
        for tol_list in best_tols_indices:
            fi.write(str(tol_list) + '\n')

        fi.write("=========================================================\n")
