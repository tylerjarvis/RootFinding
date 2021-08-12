from yroots.polynomial import MultiPower
import pickle
import yroots as yr
import numpy as np
import time

a = -np.ones(3)
b = np.ones(3)
num_loops = 1
num_tests = 100
start_deg = 2
larger_deg = 20
deg_skip = 1

# timing_dict = np.load('YRoots_2D_poly_timings_no_sign_change.pkl', allow_pickle=True)
timing_dict = {i:float() for i in range(start_deg, larger_deg + 1)}
all_num_roots = []

for deg in range(start_deg, larger_deg + 1, deg_skip):
    coeffs = np.load("tests/random_tests/coeffs/dim3_deg{}_randn.npy".format(deg))
    deg_time = 0
    num_roots = []
    for test in range(num_tests):

        c1 = coeffs[test, 0, :, :, :]
        c2 = coeffs[test, 1, :, :, :]
        c3 = coeffs[test, 2, :, :, :]
        c1[0,0,0] = 0
        c2[0,0,0] = 0
        c3[0,0,0] = 0

        f = MultiPower(c1)
        g = MultiPower(c2)
        h = MultiPower(c3)

        test_time = 0
        for _ in range(num_loops):
            print("Degree {}, Test {}/{}".format(deg, test + 1, num_tests))
            start = time.time()
            roots = (yr.solve([f,g,h], a, b))
            end = time.time()

            test_time += end - start

        del c1, c2, c3, f, g, h

        deg_time += test_time/num_loops

    timing_dict[deg] = deg_time/num_tests


    with open('new_checks_dim3.pkl', 'wb') as ofile:
        pickle.dump(timing_dict, ofile)

    np.save("new_checks_dim3", all_num_roots, allow_pickle=True, fix_imports=True)

    del coeffs
    print("Degree {} takes on average {}s".format(deg, timing_dict[deg]))
