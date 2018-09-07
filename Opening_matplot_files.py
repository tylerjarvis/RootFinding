from numalgsolve.polyroots import solve
from numalgsolve.polynomial import MultiPower
from scipy.io import loadmat
from numalgsolve.ProjectiveSpace import common_root_at_inf, roots_at_inf
from glob import glob
from numpy import set_printoptions, isclose
import numpy as np
import traceback

def pad_with_zeros(matrix):
    '''
    Extends a nonsquare matrix into a square matrix with zeros in it.
    e.g. if A is a tall matrix, returns [A|0]

    Parameters
    ----------
    matrix (np.array): a nonsquare matrix

    returns
    -------
    square matrix with zeros in it (np.array)
    '''
    m,n = matrix.shape
    l = max(m,n)
    square_matrix = np.zeros((l, l))
    square_matrix[:m,:n] = matrix
    return square_matrix

num_good_tests = 0
good_tests = list()

set_printoptions(suppress=True, linewidth=500)

#Checks which tests cases are invalid because they have higehest degree coefficients which are zero
for filename in glob('./*/*.mat'):
	dct = loadmat(filename)
	p = MultiPower(dct['p'])
	q = MultiPower(dct['q'])
	try:
		assert np.allclose(np.fliplr(p.coeff), np.triu(np.fliplr(pad_with_zeros(p.coeff)))), "In file {}, p's coefficients are not upper left triangular. \n p.coef = \n{}\nq.coeff = \n{}".format(filename, p.coeff, q.coeff)
		assert np.allclose(np.fliplr(q.coeff), np.triu(np.fliplr(pad_with_zeros(q.coeff)))), "In file {}, q's coefficients are not upper left triangular. \n p.coef = \n{}\nq.coeff = \n{}".format(filename, p.coeff, q.coeff)
		assert not np.any(np.isclose(np.diag(np.fliplr(pad_with_zeros(p.coeff))), 0)), "In file {}, p has highest term coefficients close to zero. \n p.coef = \n{}\nq.coeff = \n{}".format(filename, p.coeff, q.coeff)
		assert not np.any(np.isclose(np.diag(np.fliplr(pad_with_zeros(q.coeff))), 0)), "In file {}, p has highest term coefficients close to zero. \n p.coef = \n{}\nq.coeff = \n{}".format(filename, p.coeff, q.coeff)
	except Exception as e:
		print('\nError: ',type(e).__name__, e, filename, traceback.format_exc())
	else: #check if test case is invalid because polys a common root at infinity
		good_tests.append(filename)
		"""try:
			if not common_root_at_inf([p,q]):
				good_tests.append(filename)
			else:
				print('\n',filename, 'has a common root at infinity')
				print('p=\n',p.coeff)
				print('p roots at inf', roots_at_inf(p))
				print('q=\n',q.coeff)
				print('q roots at inf', roots_at_inf(q))

		except Exception as e:
			print('\nError: ',type(e).__name__, e, filename, 'Failed finding roots at infinity')
			print('p=\n',p.coeff)
			print('q=\n',q.coeff)"""


print("Number of Good Tests =", len(good_tests))
print("Good Tests", good_tests)

total_count = 0
mult_right_count = 0
multR_right_count = 0
multrand_right_count = 0
div_right_count = 0

set_printoptions(suppress=True, linewidth=500)

#Tests against the valid test cases
for filename in good_tests:
	total_count += 1

	dct = loadmat(filename)
	p = MultiPower(dct['p'])
	q = MultiPower(dct['q'])

	print('\n~~~~~~~ Mx ~~~~~~~')
	try:
		multsolutions = solve([p,q], MSmatrix=1)
		for zero in multsolutions:
			assert isclose(0, p(zero)), "False Zero at {}".format(zero)
			assert isclose(0, q(zero)), "False Zero at {}".format(zero)
		mult_right_count += 1
	except Exception as e:
		print('Error: ',type(e).__name__, e, filename, traceback.format_exc())

	print('\n~~~~~~~ My ~~~~~~~')
	try:
		multRsolutions = solve([p,q], MSmatrix=2)
		for zero in multRsolutions:
			assert isclose(0, p(zero)), "False Zero at {}".format(zero)
			assert isclose(0, q(zero)), "False Zero at {}".format(zero)
		multR_right_count += 1
	except Exception as e:
		print('Error: ',type(e).__name__, e, filename, traceback.format_exc())

	print('\n~~~~~~~ Mf ~~~~~~~')
	try:
		multrandsolutions = solve([p,q], MSmatrix=0)
		for zero in multrandsolutions:
			assert isclose(0, p(zero)), "False Zero at {}".format(zero)
			assert isclose(0, q(zero)), "False Zero at {}".format(zero)
		multrand_right_count += 1
	except Exception as e:
		print('Error: ',type(e).__name__, e, filename, traceback.format_exc())

	print('\n~~~~~~~ div ~~~~~~~')
	try:
		solutions = solve([p,q], MSmatrix=-1)
		for zero in solutions:
			assert isclose(0, p(zero)), "False Zero at {}".format(zero)
			assert isclose(0, q(zero)), "False Zero at {}".format(zero)
		div_right_count += 1
	except Exception as e:
		print('Error: ',type(e).__name__, e, filename, traceback.format_exc())

print('mult: {} right out of {}, {:.2f}%'.format(mult_right_count, total_count, mult_right_count/total_count*100))
print('multR: {} right out of {}, {:.2f}%'.format(multR_right_count, total_count, multR_right_count/total_count*100))
print('multrand: {} right out of {}, {:.2f}%'.format(multrand_right_count, total_count, multrand_right_count/total_count*100))
print('div: {} right out of {}, {:.2f}%'.format(div_right_count, total_count, div_right_count/total_count*100))
