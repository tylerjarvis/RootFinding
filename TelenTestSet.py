from numalgsolve.polyroots import solve
from numalgsolve.polynomial import MultiPower
from scipy.io import loadmat
from numalgsolve.ProjectiveSpace import common_root_at_inf, roots_at_inf, pad_with_zeros
from glob import glob
from numpy import set_printoptions, isclose
import numpy as np
import traceback

#functions for adding things to the right pile
def solved(filename, solver, error):
    difficulty = filename.split('/')[1]
    if difficulty == 'Easy':
        i = 0
    elif difficulty == 'Medium':
        i = 1
    elif difficulty == 'Hard':
        i = 2
    else:
        i = 3
    if solver == 'mult':
        mult_solved[i].append((filename, error))
    elif solver == 'multrand':
        multrand_solved[i].append((filename, error))
    else:
        div_solved[i].append(filename)

def not_solved(filename, solver, error):
    difficulty = filename.split('/')[1]
    if difficulty == 'Easy':
        i = 0
    elif difficulty == 'Medium':
        i = 1
    elif difficulty == 'Hard':
        i = 2
    else:
        i = 3
    if solver == 'mult':
        mult_not_solved[i].append((filename, error))
    elif solver == 'multrand':
        multrand_not_solved[i].append((filename, error))
    else:
        div_not_solved[i].append((filename, error))

def right_roots(filename, solver, roots, percent):
    difficulty = filename.split('/')[1]
    if difficulty == 'Easy':
        i = 0
    elif difficulty == 'Medium':
        i = 1
    elif difficulty == 'Hard':
        i = 2
    else:
        i = 3
    if solver == 'mult':
        mult_right[i].append((filename, percent))
    elif solver == 'multrand':
        multrand_right[i].append((filename,percent))
    else:
        div_right[i].append((filename,percent))

def wrong_roots(filename, solver, roots, percent):
    difficulty = filename.split('/')[1]
    if difficulty == 'Easy':
        i = 0
    elif difficulty == 'Medium':
        i = 1
    elif difficulty == 'Hard':
        i = 2
    else:
        i = 3
    if solver == 'mult':
        mult_wrong[i].append((filename,percent))
    elif solver == 'multrand':
        multrand_wrong[i].append((filename,percent))
    else:
        div_wrong[i].append((filename,percent))

def percent_good_roots(roots, polys, ignore_out_of_range, tolerance):
    #function for calculating what percent of the roots were good
    if len(roots) == 0:
        return 0
    correct = 0
    outOfRange = 0
    for root in roots:
        good = True
        for poly in polys:
            if not np.isclose(0, poly(root), rtol = tolerance):
                good = False
                if (np.abs(root) > 1).any():
                    outOfRange += 1
                break
        if good:
            correct += 1

    if ignore_out_of_range:
        return correct/(len(roots)-outOfRange)
    else:
        return correct/(len(roots))

if __name__ == "__main__":
    np.set_printoptions(suppress=True, linewidth=1000)

    #A ton of lists to sort test cases into
    total_count = 0
    #difficulty is indicated by which sublist it's in. Easy in 0, medium in 1, hard in 2, impossible in 3
    mult_solved = [list(),list(),list(),list()]
    mult_not_solved = [list(),list(),list(),list()]
    mult_right = [list(),list(),list(),list()]
    mult_wrong = [list(),list(),list(),list()]

    multrand_solved = [list(),list(),list(),list()]
    multrand_not_solved = [list(),list(),list(),list()]
    multrand_right = [list(),list(),list(),list()]
    multrand_wrong = [list(),list(),list(),list()]

    div_solved = [list(),list(),list(),list()]
    div_not_solved = [list(),list(),list(),list()]
    div_right = [list(),list(),list(),list()]
    div_wrong = [list(),list(),list(),list()]

    for filename in glob('./*/*.mat'):
        total_count += 1

        dct = loadmat(filename)
        p = MultiPower(dct['p'])
        q = MultiPower(dct['q'])

        error_message = str()

        try:
            assert np.allclose(np.fliplr(pad_with_zeros(p.coeff)), np.triu(np.fliplr(pad_with_zeros(p.coeff)))), "p's coefficients are not upper left triangular. \n p.coef = \n{}\nq.coeff = \n{}".format(p.coeff, q.coeff)
            assert np.allclose(np.fliplr(pad_with_zeros(q.coeff)), np.triu(np.fliplr(pad_with_zeros(q.coeff)))), "q's coefficients are not upper left triangular. \n p.coef = \n{}\nq.coeff = \n{}".format(p.coeff, q.coeff)
            assert not np.any(np.isclose(np.diag(np.fliplr(pad_with_zeros(p.coeff))), 0)), "p has highest term coefficients close to zero. \n p.coef = \n{}\nq.coeff = \n{}".format(p.coeff, q.coeff)
            assert not np.any(np.isclose(np.diag(np.fliplr(pad_with_zeros(q.coeff))), 0)), "p has highest term coefficients close to zero. \n p.coef = \n{}\nq.coeff = \n{}".format(p.coeff, q.coeff)
        except AssertionError as e:
            error_message += '\nNot Upper Triangular:' + str(e)
        finally:
            try:
                if common_root_at_inf([p,q]) != False:
                    error_message += '\nCommon root at infinity:' + str(common_root_at_inf([p,q])[1])
            except Exception as e:
                error_message += '\nFailed finding roots at Infinity:' + str(e)
            finally:
                try:
                    mult_solutions = solve([p,q], MSmatrix=1)
                except Exception as e:
                    mult_error = '\n' + str(e)
                    not_solved(filename, 'mult', error_message + mult_error)
                else:
                    solved(filename, 'mult', error_message)
                    multpercent = percent_good_roots(mult_solutions, [p,q], ignore_out_of_range=False, tolerance=1e-9)
                    if multpercent == 1: #if 100% of the roots were right, report that
                        right_roots(filename, 'mult', mult_solutions, multpercent)
                    else:
                        wrong_roots(filename, 'mult', mult_solutions, multpercent)
                finally:
                    try:
                        multrand_solutions = solve([p,q], MSmatrix=0)
                    except Exception as e:
                        multrand_error = '\n' + str(e)
                        not_solved(filename, 'multrand', error_message + multrand_error)
                    else:
                        solved(filename, 'multrand', error_message)
                        multrandpercent = percent_good_roots(multrand_solutions, [p,q], ignore_out_of_range=False, tolerance=1e-9)
                        if multrandpercent == 1:
                            right_roots(filename, 'multrand', multrand_solutions, multrandpercent)
                        else:
                            wrong_roots(filename, 'multrand', multrand_solutions, multrandpercent)
                    finally:
                        try:
                            div_solutions = solve([p,q], MSmatrix=0)
                        except Exception as e:
                            div_error = '\n' + str(e)
                            not_solved(filename, 'div', error_message + div_error)
                        else:
                            solved(filename, 'div', error_message)
                            divpercent = percent_good_roots(div_solutions, [p,q], ignore_out_of_range=False, tolerance=1e-9)
                            if divpercent == 1:
                                right_roots(filename, 'div', div_solutions, divpercent)
                            else:
                                wrong_roots(filename, 'div', div_solutions, divpercent)

    #Report Number Solved/Right for each method
    print('\n{} Test Cases \t{} Easy         \t\t{} Medium             \t\t\t{} Hard             \t\t\t\t{} Impossible'.format(total_count, 128, 48, 15, 24))
    print('mult:         \t{} solved, {} right     \t{} solved, {} right    \t{} solved, {} right        \t\t{} solved, {} right'.format(len(mult_solved[0]), len(mult_right[0]), len(mult_solved[1]), len(mult_right[1]), len(mult_solved[2]), len(mult_right[2]), len(mult_solved[3]), len(mult_right[3])))
    print('multrand:     \t{} solved, {} right     \t{} solved, {} right    \t{} solved, {} right        \t\t{} solved, {} right'.format(len(multrand_solved[0]), len(multrand_right[0]), len(multrand_solved[1]), len(multrand_right[1]), len(multrand_solved[2]), len(multrand_right[2]), len(multrand_solved[3]), len(multrand_right[3])))
    print('div:          \t{} solved, {} right     \t{} solved, {} right    \t{} solved, {} right        \t\t{} solved, {} right'.format(len(div_solved[0]), len(div_right[0]), len(div_solved[1]), len(div_right[1]), len(div_solved[2]), len(div_right[2]), len(div_solved[3]), len(div_right[3])))

    print('\n\nMult Failed to Solve:\n')
    print('--Easy--\n')
    print(*mult_not_solved[0], sep='\n')
    print('--Medium--\n')
    print(*mult_not_solved[1], sep='\n')
    print('--Hard--\n')
    print(*mult_not_solved[2], sep='\n')
    print('--Impossible--\n')
    print(*mult_not_solved[3], sep='\n')

    print('\n\nmultrand Failed to Solve:\n')
    print('--Easy--\n')
    print(*multrand_not_solved[0], sep='\n')
    print('--Medium--\n')
    print(*multrand_not_solved[1], sep='\n')
    print('--Hard--\n')
    print(*multrand_not_solved[2], sep='\n')
    print('--Impossible--\n')
    print(*multrand_not_solved[3], sep='\n')

    print('\n\ndiv Failed to Solve:\n')
    print('--Easy--\n')
    print(*div_not_solved[0], sep='\n')
    print('--Medium--\n')
    print(*div_not_solved[1], sep='\n')
    print('--Hard--\n')
    print(*div_not_solved[2], sep='\n')
    print('--Impossible--\n')
    print(*div_not_solved[3], sep='\n')

    print('\n\nMult Found Incorrect Roots:\n')
    print('--Easy--\n')
    print(*mult_wrong[0], sep='\n')
    print('--Medium--\n')
    print(*mult_wrong[1], sep='\n')
    print('--Hard--\n')
    print(*mult_wrong[2], sep='\n')
    print('--Impossible--\n')
    print(*mult_wrong[3], sep='\n')

    print('\n\nmultrand Found Incorrect Roots:\n')
    print('--Easy--\n')
    print(*multrand_wrong[0], sep='\n')
    print('--Medium--\n')
    print(*multrand_wrong[1], sep='\n')
    print('--Hard--\n')
    print(*multrand_wrong[2], sep='\n')
    print('--Impossible--\n')
    print(*multrand_wrong[3], sep='\n')

    print('\n\ndiv Found Incorrect Roots:\n')
    print('--Easy--\n')
    print(*div_wrong[0], sep='\n')
    print('--Medium--\n')
    print(*div_wrong[1], sep='\n')
    print('--Hard--\n')
    print(*div_wrong[2], sep='\n')
    print('--Impossible--\n')
    print(*div_wrong[3], sep='\n')

    print(mult_not_solved == div_not_solved)
    print(mult_not_solved == multrand_not_solved)
