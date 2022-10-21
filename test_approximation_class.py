import numpy as np

import approximation_class
from yroots.utils import slice_top


#a test function
f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81
a,b = np.array([-1,-1]),np.array([1,1])

max_deg = {1: 100000, 2:1000, 3:9, 4:9, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2}

def get_err(M,M2):
    coeff2 = M2.copy()
    coeff2[slice_top(M.shape)] -= M.copy()
    return np.sum(np.abs(coeff2))

def test_rescale():
    """
    checks the rescale method
    """
    f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81
    a,b = np.array([-1,-1]),np.array([1,1])
    deg = 4
    f_approx = approximation_class.M_maker(f,a,b,deg)
    assert np.allclose(f_approx.M_rescaled * f_approx.inf_norm,f_approx.M) == True, "failed to scale by inf norm"

def test_deg_cap():
    """
    uses max_deg + 1,
    makes sure that for whatever dim we are using,
    the degree is reduced to the upper limit
    """
    f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81
    a,b = np.array([-1,-1]),np.array([1,1])
    deg = max_deg[len(a)] + 1 #edge case
    f_approx = approximation_class.M_maker(f,a,b,deg)
    assert f_approx.deg <= f_approx.max_deg[f_approx.dim]
    
def test_dict_length_doubling():
    """
    the number of double-ups is equal to the length of the dictionary
    """
    deg = 4
    f = lambda x,y: np.cos(10*x*y)
    f_approx = approximation_class.M_maker(f,a,b,deg)
    #print(len(f_approx.memo_dict))
    if f_approx.deg < max_deg[f_approx.dim]:
        assert len(f_approx.memo_dict) == np.log(f_approx.deg)/np.log(2)
    else:
        if f_approx.deg ==  2*list(f_approx.memo_dict.keys())[-2]:
            assert len(f_approx.memo_dict) == np.log(f_approx.deg)/np.log(2)
        else:
            assert len(f_approx.memo_dict) == 1 + np.log(f_approx.memo_dict[-2])/np.log(2)

def test_dict_length_no_double():
    deg = 4
    f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81
    f_approx = approximation_class.M_maker(f,a,b,deg)
    print(len(f_approx.memo_dict))
    if f_approx.deg < max_deg[f_approx.dim]:
        assert len(f_approx.memo_dict) == np.log(f_approx.deg)/np.log(2)
    else:
        if f_approx.deg ==  2*list(f_approx.memo_dict.keys())[-2]:
            assert len(f_approx.memo_dict) == np.log(f_approx.deg)/np.log(2)
        else:
            assert len(f_approx.memo_dict) == 1 + np.log(f_approx.memo_dict[-2])/np.log(2)

def test_cap_double_a():
    """
    test case: the error test for deg=max_deg fails but since it is the limit,
    we don't double up. Edge case A.
    """
    j = lambda x,y: np.exp(x-2*x**2-y**2)*np.sin(10*(x+y+x*y**2))
    deg = 500
    a,b = np.array([-1,-1]),np.array([1,1])
    j_approx=  approximation_class.M_maker(j,a,b,deg)
    assert j_approx.deg <= j_approx.max_deg[j_approx.dim]

def test_cap_double_b():
    """
    test case: the error test for deg=max_deg fails but since it is the limit,
    we don't double up. Edge case B.
    """
    j = lambda x,y: np.exp(x-2*x**2-y**2)*np.sin(10*(x+y+x*y**2))
    deg = 501
    a,b = np.array([-1,-1]),np.array([1,1])
    j_approx=  approximation_class.M_maker(j,a,b,deg)
    assert j_approx.deg <= j_approx.max_deg[j_approx.dim]

def test_other_attributes():
    """
    checks that all the attributes are what we say they are:
    length of a is the dim
    f is equal to f_approx.f attribute
    if f does not have the `evalute_grid` attrbiute:
        we want the memory location of memoized info to match current info
    number of rows of M minus one is equal to the degree
    "..." of M2 minus one is equal to double the degree
    """
    deg = 4
    f = lambda x,y: np.cos(10*x*y)
    f_approx = approximation_class.M_maker(f,a,b,deg)
    assert len(a) == f_approx.dim
    assert f == f_approx.f
    assert len(f_approx.M) - 1 == f_approx.deg
    assert len(f_approx.M2) - 1 == 2*f_approx.deg
    assert np.isclose(f_approx.err,get_err(f_approx.M,f_approx.M2)) == True
    vals = f_approx.chebyshev_block_copy(f_approx.values_block) #nonetype
    inf_normo = np.max(np.abs(vals))
    assert f_approx.inf_norm == inf_normo

def test_error_test():
    """
    Checks for a correct result of the error test

    If the approximation degree is within the limit, then the result of the test 
    should be a pass.

    If not within the limit AND we have doubled up (checked by evaluating 
    `len(f_approx.memo_dict)`), then we want a failure
    """
    a,b = np.array([-1,-1]),np.array([1,1])
    deg = 4

    f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81 #no doubling
    f_approx = approximation_class.M_maker(f,a,b,deg)
    g = lambda x,y: np.cos(10*x*y) #doubling
    g_approx = approximation_class.M_maker(g,a,b,deg)
    
    approxies = [f_approx,g_approx]

    for approx in approxies:
        if approx.deg < approx.max_deg[approx.dim]:
            test = approx.error_test(approx.err,approx.abs_approx_tol,approx.rel_approx_tol,approx.inf_norm) == True
            assert test == True
        elif len(f_approx.memo_dict) > 1:
            input_deg = approx.memo_dict.keys()[-2]
            M, infy = approx.interval_approximate_nd(f,a,b,input_deg,return_inf_norm=True)
            M2 = approx.interval_approximate_nd(f,a,b,2*input_deg)
            coeffs2 = M
            coeffs2[slice_top(M.shape)] -= M
            error = np.sum(np.abs(coeffs2))

            test = approx.error_test(error,approx.abs_approx_tol,approx.rel_approx_tol,infy)
            assert test == False