from yroots import eriks_code
from yroots.subdivision import solve
import numpy as np
from yroots.utils import transform, slice_top
from scipy.fftpack import fftn
from itertools import product
from time import time
from tests.chebfun2_suite import norm_pass_or_fail,residuals,residuals_pass_or_fail,verbose_pass_or_fail
from matplotlib import pyplot as plt

class M_maker:
    def __init__(self,f,a,b,deg,return_inf_norm=False,rel_approx_tol=1.e-15, abs_approx_tol=1.e-12):
        #print(deg)
        dim = len(a)
        if dim != len(b):
            raise ValueError("dimension mismatch")
        self.dim = dim
        self.f = f
        self.a = a
        self.b = b
        self.rel_approx_tol = rel_approx_tol
        self.abs_approx_tol = abs_approx_tol
        self.return_inf_norm = return_inf_norm
        self.deg = self.find_good_deg(f,deg,dim,a,b)

        print(self.deg)

        if self.return_inf_norm == True:
            self.M, self.inf_norm = self.interval_approximate_nd(self.f,self.a,self.b,self.deg,self.return_inf_norm)
            self.M2 = self.interval_approximate_nd(self.f,self.a,self.b,2*self.deg,self.return_inf_norm)[0]
            self.M2[slice_top(self.M.shape)] -= self.M
            self.err = np.sum(np.abs(self.M2))
        else:
            self.M = self.interval_approximate_nd(self.f,self.a,self.b,self.deg)
            self.M2 = self.interval_approximate_nd(self.f,self.a,self.b,2*self.deg)
            self.M2[slice_top(self.M.shape)] -= self.M
            self.err = np.sum(np.abs(self.M2))


    def error_test(self,error,abs_approx_tol,rel_approx_tol,inf_norm): 
        """
        Determines whether the approximation is within the error tolerance

        Parameters
        ----------
        error: float
        The absolute value of the difference of the sum of abs values of M and M2
        rel_approx_tol: float
        some input I might want to cover my bases about
        abs_approx_tol: float
        some input I might want to cover my bases about
        inf_norm: float
        the sup norm on the approximation

        Returns
        -------
        Bool: if the error test has been passed or not
        """
        print("welcome to approx class 2")
        return error > abs_approx_tol+rel_approx_tol*inf_norm

    def find_good_deg(self,f,deg,dim,a,b):
        """
        Finds the right degree with which to approximate on the interval.

        Parameters
        ----------
        f : function from R^n -> R
        The function to interpolate.
        deg : numpy array
        The degree of the interpolation in each dimension.
        dim: int
        Dimension
        a : numpy array
        The lower bound on the interval.
        b : numpy array
        The upper bound on the interval.

        Returns
        -------
        deg: the correct approximation degree
        """
        #print("finding a good degree")
        max_deg = {1: 100, 2:20, 3:9, 4:9, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2}

        coeff = self.interval_approximate_nd(f, a, b, deg)
        coeff2, inf_norm = self.interval_approximate_nd(f, a, b, deg*2, return_inf_norm=True)
        coeff2[slice_top(coeff.shape)] -= coeff
        self.err = np.sum(np.abs(coeff2))

        while deg < max_deg[dim]:
            print(self.err)
            if self.error_test(self.err,self.abs_approx_tol,self.rel_approx_tol,inf_norm):
                print("passed the test")
                return deg
            elif 2*deg > max_deg[dim]:
                print("maxxed out")
                return max_deg[dim] #MUST BE SELF-UPDATING
            else:
                print("failure and double")
                deg *= 2
                coeff = self.interval_approximate_nd(f, a, b, deg)
                coeff2, inf_norm = self.interval_approximate_nd(f, a, b, deg*2, return_inf_norm=True)
                coeff2[slice_top(coeff.shape)] -= coeff
                self.err = np.sum(np.abs(coeff2))

    def interval_approximate_nd(self,f, a, b, deg, return_inf_norm=False):
        """Finds the chebyshev approximation of an n-dimensional function on an
        interval.

        Parameters
        ----------
        f : function from R^n -> R
            The function to interpolate.
        a : numpy array
            The lower bound on the interval.
        b : numpy array
            The upper bound on the interval.
        deg : numpy array
            The degree of the interpolation in each dimension. #Question THIS IS A NUMPY ARRAY
        return_inf_norm : bool
            whether to return the inf norm of the function

        Returns
        -------
        coeffs : numpy array
            The coefficient of the chebyshev interpolating polynomial.
        inf_norm : float
            The inf_norm of the function
        """
        #print("doing an approximation")
        dim = len(self.a)
        if dim != len(self.b):
            raise ValueError("Interval dimensions must be the same!")

        if hasattr(self.f,"evaluate_grid"):
            cheb_values = np.cos(np.arange(deg+1)*np.pi/deg) #simply executes the lines within the function instead of the function call
            chepy_pts =  np.column_stack([cheb_values]*dim)
            cheb_pts = transform(chepy_pts,a,b)
            self.values_block = f.evaluate_grid(cheb_pts)
        else:
            cheb_vals = np.cos(np.arange(deg+1)*np.pi/deg)
            cheb_grid = np.meshgrid(*([cheb_vals]*dim),indexing='ij')
            flatten = lambda x: x.flatten()
            cheby_pts = np.column_stack(tuple(map(flatten, cheb_grid)))
            cheb_pts = transform(cheby_pts,a,b)
            self.values_block = f(*cheb_pts.T).reshape(*([deg+1]*dim))

        self.values = self.chebyshev_block_copy(self.values_block)

        if return_inf_norm:
            inf_norm = np.max(np.abs(self.values))

        x0_slicer, deg_slicer, slices, rescale = self.interval_approx_slicers(dim,deg)
        coeffs = fftn(self.values/rescale).real

        for x0sl, degsl in zip(x0_slicer, deg_slicer):
            # halve the coefficients in each slice
            coeffs[x0sl] /= 2
            coeffs[degsl] /= 2

        if return_inf_norm:
            return coeffs[tuple(slices)], inf_norm
        else:
            return coeffs[tuple(slices)]
    
    def chebyshev_block_copy(self,values_block):
        """This functions helps avoid double evaluation of functions at
        interpolation points. It takes in a tensor of function evaluation values
        and copies these values to a new tensor appropriately to prepare for
        chebyshev interpolation.

        Parameters
        ----------
        values_block : numpy array
        block of values from function evaluation
        Returns
        -------
        values_cheb : numpy array
        chebyshev interpolation values
        """
        #print("doing a block copy")
        #np.empty(tuple([2*deg])*dim, dtype=np.float64)
        dim = values_block.ndim
        deg = values_block.shape[0] - 1
        #values_cheb = values_arr(dim)
        values_cheb = np.empty(tuple([2*deg])*dim, dtype=np.float64) #self.values_cheb?
        block_slicers, cheb_slicers, slicer = self.block_copy_slicers(dim, deg)

        for cheb_idx, block_idx in zip(cheb_slicers, block_slicers):
            try:
                values_cheb[cheb_idx] = values_block[block_idx]
            except ValueError as e:
                if str(e)[:42] == 'could not broadcast input array from shape': 
                    #self.values_arr.memo[(dim, )] = np.empty(tuple([2*deg])*dim, dtype=np.float64) #I KNOW WHAT THIS DOES!
                    values_cheb = np.empty(tuple([2*deg])*dim, dtype=np.float64)
                    values_cheb[cheb_idx] = values_block[block_idx]
                else:
                    raise ValueError(e)
        return values_cheb[slicer]
    
    def block_copy_slicers(self,dim, deg):
        """Helper function for chebyshev_block_copy.
        Builds slice objects to index into the evaluation array to copy
        in preparation for the fft.

        Parameters
        ----------
        dim : int
            Dimension
        dim : int
            Degree of approximation

        Returns
        -------
        block_slicers : list of tuples of slice objects
            Slice objects used to index into the evaluations
        cheb_slicers : list of tuples of slice objects
            Slice objects used to index into the array we're copying evaluations to
        slicer : tuple of slice objets
            Used to index into the portion of that array we're using for the fft input
        """
        #print("getting slicers")
        block_slicers = []
        cheb_slicers = []
        full_arr_deg = 2*deg
        for block in product([False, True], repeat=dim):
            cheb_idx = [slice(0, deg+1)]*dim
            block_idx = [slice(0, full_arr_deg)]*dim
            for i, flip_dim in enumerate(block):
                if flip_dim:
                    cheb_idx[i] = slice(deg+1, full_arr_deg)
                    block_idx[i] = slice(deg-1, 0, -1)
            block_slicers.append(tuple(block_idx))
            cheb_slicers.append(tuple(cheb_idx))
        return block_slicers, cheb_slicers, tuple([slice(0, 2*deg)]*dim)

    def interval_approx_slicers(self,dim, deg):
        """Helper function for interval_approximate_nd. Builds slice objects to index
        into the output of the fft and divide some of the values by 2 and turn them into
        coefficients of the approximation.

        Parameters
        ----------
        dim : int
            The interpolation dimension.
        deg : int
            The interpolation degree. #SEE WE TAKE THIS AS A SCALAR

        Returns
        -------
        x0_slicer : list of tuples of slice objects
            Slice objects used to index into the the degree 1 monomials
        deg_slicer : list of tuples of slice objects
            Slice objects used to index into the the degree d monomials
        slices : tuple of slice objets
            Used to index into the portion of the array that are coefficients
        rescale : int
            amount to rescale the evaluations by in order to feed them into the fft
        """
        #print("getting helped")
        x0_slicer = [tuple([slice(None) if i != d else 0 for i in range(dim)])
                    for d in range(dim)]
        deg_slicer = [tuple([slice(None) if i != d else deg for i in range(dim)])
                    for d in range(dim)]
        slices = tuple([slice(0, deg+1)]*dim)
        return x0_slicer, deg_slicer, slices, deg**dim

def test_roots_1_1():
    # Test 1.1
    f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81
    g = lambda x,y: y-x**6
    f_deg,g_deg = 4,6
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    #yroots = solve([f,g],[-1,-1],[1,1], plot=False)
    f_approx = M_maker(f,a,b,f_deg) #use the class
    g_approx = M_maker(g,a,b,g_deg)
    Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
    yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
    t = time() - start
    print(yroots)
    actual_roots = np.load('Polished_results/polished_1.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_1.1.csv', delimiter=',')
    
    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 1.1, cheb_roots=chebfun_roots)
        
def test_roots_1_2():
    # Test 1.2
    f = lambda x,y: (y**2-x**3)*((y-0.7)**2-(x-0.3)**3)*((y+0.2)**2-(x+0.8)**3)*((y+0.2)**2-(x-0.8)**3)
    g = lambda x,y: ((y+.4)**3-(x-.4)**2)*((y+.3)**3-(x-.3)**2)*((y-.5)**3-(x+.6)**2)*((y+0.3)**3-(2*x-0.8)**3)
    f_deg, g_deg = 12,11
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    #yroots = solve([f,g],[-1,-1],[1,1], plot=False)
    f_approx = M_maker(f,a,b,f_deg) #use the class
    g_approx = M_maker(g,a,b,g_deg)
    Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
    yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
    print(yroots)
    t = time() - start
    # Get Polished results (Newton polishing misses roots)
    #yroots2 = solve([f,g],[-1,-1],[1,1], abs_approx_tol=[1e-8, 1e-12], rel_approx_tol=[1e-15, 1e-18],\ this test is wierd, what to do...
                #max_cond_num=[1e5, 1e2], good_zeros_factor=[100,100], min_good_zeros_tol=[1e-5, 1e-5],\
                #check_eval_error=[True,True], check_eval_freq=[1,2], plot=False, target_tol=[1e-13, 1e-13])
    actual_roots = np.load('Polished_results/polished_1.2.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_1.2.csv', delimiter=',')

    print(yroots)
    print(actual_roots)
    print(chebfun_roots)

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 1.1, cheb_roots=chebfun_roots)

def test_roots_1_3():
    # Test 1.3
    f = lambda x,y: y**2-x**3
    g = lambda x,y: (y+.1)**3-(x-.1)**2
    f_deg,g_deg = 3,3
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    #yroots = solve([f,g],[-1,-1],[1,1], plot=False)
    f_approx = M_maker(f,a,b,f_deg) #use the class
    g_approx = M_maker(g,a,b,g_deg)
    Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
    yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
    t = time() - start
    actual_roots = np.load('Polished_results/polished_1.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_1.3.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 1.1, cheb_roots=chebfun_roots)

def test_roots_1_4():
    # Test 1.4
    f = lambda x,y: x - y + .5
    g = lambda x,y: x + y
    f_deg,g_deg = 1,1
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    #yroots = solve([f,g],[-1,-1],[1,1], plot=False)
    f_approx = M_maker(f,a,b,f_deg) #use the class
    g_approx = M_maker(g,a,b,g_deg)
    Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
    yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
    t = time() - start
    # Single root has to be in matrix form because yroots
    # returns the roots in matrix form.
    a_roots = np.array([[-.25, .25]])
    chebfun_roots = np.array([np.loadtxt('Chebfun_results/test_roots_1.4.csv', delimiter=',')])

    return t, verbose_pass_or_fail([f,g], yroots, a_roots, 1.1, cheb_roots=chebfun_roots)

def test_roots_1_5():
    # Test 1.5
    f = lambda x,y: y + x/2 + 1/10
    g = lambda x,y: y - 2.1*x + 2
    f_deg,g_deg = 1,1
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    #yroots = solve([f,g],[-1,-1],[1,1], plot=False)
    f_approx = M_maker(f,a,b,f_deg) #use the class
    g_approx = M_maker(g,a,b,g_deg)
    Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
    yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
    t = time() - start
    # Single root has to be in matrix form because yroots
    # returns the roots in matrix form.
    a_roots = np.array([[0.730769230769231, -0.465384615384615]])

    chebfun_roots = np.array([np.loadtxt('Chebfun_results/test_roots_1.5.csv', delimiter=',')])

    return t, verbose_pass_or_fail([f,g], yroots, a_roots, 1.1, cheb_roots=chebfun_roots)

def test_roots_2_1():
    # Test 2.1
    f = lambda x,y: np.cos(10*x*y)
    g = lambda x,y: x + y**2
    f_deg,g_deg = 1,2
    start = time()
    yroots = solve([f,g],[-1,-1],[1,1], plot=False)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_2.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 2.1, cheb_roots=chebfun_roots)

def test_roots_2_2():
    # Test 2.2
    f = lambda x,y: x
    g = lambda x,y: (x-.9999)**2 + y**2-1
    f_deg,g_deg = 1,2
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    f_approx = M_maker(f,a,b,f_deg) #use the class
    g_approx = M_maker(g,a,b,g_deg)
    Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
    yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
    t = time() - start
    actual_roots = np.load('Polished_results/polished_2.2.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.2.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 2.2, cheb_roots=chebfun_roots)

def test_roots_2_3():
    # Test 2.3
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    f_deg,g_deg = 1,1
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    f_approx = M_maker(f,a,b,f_deg) #use the class
    g_approx = M_maker(g,a,b,g_deg)
    Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
    yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
    t = time() - start
    actual_roots = np.load('Polished_results/polished_2.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.3.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 2.3, cheb_roots=chebfun_roots)

def test_roots_2_4():
    # Test 2.4
    f = lambda x,y: np.exp(x-2*x**2-y**2)*np.sin(10*(x+y+x*y**2))
    g = lambda x,y: np.exp(-x+2*y**2+x*y**2)*np.sin(10*(x-y-2*x*y**2))
    f_deg,g_deg = 1,1
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    f_approx = M_maker(f,a,b,f_deg) #use the class
    g_approx = M_maker(g,a,b,g_deg)
    Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
    yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
    t = time() - start
    actual_roots = np.load('Polished_results/polished_2.4.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.4.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 2.4, cheb_roots=chebfun_roots)

def test_roots_2_5():
    # Test 2.5
    f = lambda x,y: 2*y*np.cos(y**2)*np.cos(2*x)-np.cos(y)
    g = lambda x,y: 2*np.sin(y**2)*np.sin(2*x)-np.sin(x)
    f_deg,g_deg = 1,1
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    f_approx = M_maker(f,a,b,f_deg) #use the class
    g_approx = M_maker(g,a,b,g_deg)
    Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
    yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
    t = time() - start
    actual_roots = np.load('Polished_results/polished_2.5.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.5.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 2.5, cheb_roots=chebfun_roots, tol=2.220446049250313e-12)

def test_roots_3_1():
    # Test 3.1
    f = lambda x,y: ((x-.3)**2+2*(y+0.3)**2-1)
    g = lambda x,y: ((x-.49)**2+(y+.5)**2-1)*((x+0.5)**2+(y+0.5)**2-1)*((x-1)**2+(y-0.5)**2-1)
    f_deg,g_deg = 2,6
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    f_approx = M_maker(f,a,b,f_deg) #use the class
    g_approx = M_maker(g,a,b,g_deg)
    Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
    yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
    t = time() - start
    actual_roots = np.load('Polished_results/polished_3.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_3.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 3.1, cheb_roots=chebfun_roots, tol=2.220446049250313e-11)

def test_roots_3_2():
    # Test 3.2
    f = lambda x,y: ((x-0.1)**2+2*(y-0.1)**2-1)*((x+0.3)**2+2*(y-0.2)**2-1)*((x-0.3)**2+2*(y+0.15)**2-1)*((x-0.13)**2+2*(y+0.15)**2-1)
    g = lambda x,y: (2*(x+0.1)**2+(y+0.1)**2-1)*(2*(x+0.1)**2+(y-0.1)**2-1)*(2*(x-0.3)**2+(y-0.15)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1)
    f_deg,g_deg = 8,8
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    f_approx = M_maker(f,a,b,f_deg) #use the class
    g_approx = M_maker(g,a,b,g_deg)
    Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
    yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
    t = time() - start
    actual_roots = np.load('Polished_results/polished_3.2.npy')

    yroots2 = solve([f,g],[-1,-1],[1,1], abs_approx_tol=[1e-8, 1e-15], rel_approx_tol=[1e-12, 1e-29],\
                max_cond_num=[1e5, 1e2], good_zeros_factor=[100,100], min_good_zeros_tol=[1e-5, 1e-5],\
                check_eval_error=[True,True], check_eval_freq=[1,1], plot=False)

    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_3.2.csv', delimiter=',')
    actual_roots = chebfun_roots

    return t, verbose_pass_or_fail([f,g], yroots, yroots2, 3.2, cheb_roots=chebfun_roots, tol=2.220446049250313e-11)

def test_roots_4_1():
    # Test 4.1
    # This system hs 4 true roots, but ms fails (finds 5).
    f = lambda x,y: np.sin(3*(x+y))
    g = lambda x,y: np.sin(3*(x-y))
    f_deg,g_deg = 20,20
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    f_approx = M_maker(f,a,b,f_deg) #use the class
    g_approx = M_maker(g,a,b,g_deg)
    Mf, Mg, err_f, err_g = f_approx.M, g_approx.M, f_approx.err, g_approx.err
    yroots = np.array(eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g])))
    t = time() - start
    actual_roots = np.load('Polished_results/polished_4.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_4.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 4.1, cheb_roots=chebfun_roots)

def test_roots_4_2():
    # Test 4.2 #PLEASE GET f_deg, g_deg right now
    f = lambda x,y: ((90000*y**10 + (-1440000)*y**9 + (360000*x**4 + 720000*x**3 + 504400*x**2 + 144400*x + 9971200)*(y**8) +
                ((-4680000)*x**4 + (-9360000)*x**3 + (-6412800)*x**2 + (-1732800)*x + (-39554400))*(y**7) + (540000*x**8 +
                2160000*x**7 + 3817600*x**6 + 3892800*x**5 + 27577600*x**4 + 51187200*x**3 + 34257600*x**2 + 8952800*x + 100084400)*(y**6) +
                ((-5400000)*x**8 + (-21600000)*x**7 + (-37598400)*x**6 + (-37195200)*x**5 + (-95198400)*x**4 +
                (-153604800)*x**3 + (-100484000)*x**2 + (-26280800)*x + (-169378400))*(y**5) + (360000*x**12 + 2160000*x**11 +
                6266400*x**10 + 11532000*x**9 + 34831200*x**8 + 93892800*x**7 + 148644800*x**6 + 141984000*x**5 + 206976800*x**4 +
                275671200*x**3 + 176534800*x**2 + 48374000*x + 194042000)*(y**4) + ((-2520000)*x**12 + (-15120000)*x**11 + (-42998400)*x**10 +
                (-76392000)*x**9 + (-128887200)*x**8 + (-223516800)*x**7 + (-300675200)*x**6 + (-274243200)*x**5 + (-284547200)*x**4 +
                (-303168000)*x**3 + (-190283200)*x**2 + (-57471200)*x + (-147677600))*(y**3) + (90000*x**16 + 720000*x**15 + 3097600*x**14 +
                9083200*x**13 + 23934400*x**12 + 58284800*x**11 + 117148800*x**10 + 182149600*x**9 + 241101600*x**8 + 295968000*x**7 +
                320782400*x**6 + 276224000*x**5 + 236601600*x**4 + 200510400*x**3 + 123359200*x**2 + 43175600*x + 70248800)*(y**2) +
                ((-360000)*x**16 + (-2880000)*x**15 + (-11812800)*x**14 + (-32289600)*x**13 + (-66043200)*x**12 + (-107534400)*x**11 +
                (-148807200)*x**10 + (-184672800)*x**9 + (-205771200)*x**8 + (-196425600)*x**7 + (-166587200)*x**6 + (-135043200)*x**5 +
                (-107568800)*x**4 + (-73394400)*x**3 + (-44061600)*x**2 + (-18772000)*x + (-17896000))*y + (144400*x**18 + 1299600*x**17 +
                5269600*x**16 + 12699200*x**15 + 21632000*x**14 + 32289600*x**13 + 48149600*x**12 + 63997600*x**11 + 67834400*x**10 +
                61884000*x**9 + 55708800*x**8 + 45478400*x**7 + 32775200*x**6 + 26766400*x**5 + 21309200*x**4 + 11185200*x**3 + 6242400*x**2 +
                3465600*x + 1708800)))
    g = lambda x,y: 1e-4*(y**7 + (-3)*y**6 + (2*x**2 + (-1)*x + 2)*y**5 + (x**3 + (-6)*x**2 + x + 2)*y**4 + (x**4 + (-2)*x**3 + 2*x**2 +
                x + (-3))*y**3 + (2*x**5 + (-3)*x**4 + x**3 + 10*x**2 + (-1)*x + 1)*y**2 + ((-1)*x**5 + 3*x**4 + 4*x**3 + (-12)*x**2)*y +
                (x**7 + (-3)*x**5 + (-1)*x**4 + (-4)*x**3 + 4*x**2))
    f_deg,g_deg = 8,8
    a,b = np.array([-1,-1]),np.array([1,1])
    start = time()
    yroots = solve([f,g],[-1, -1],[1,1], plot=False)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_4.2.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_4.2.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 4.2, cheb_roots=chebfun_roots)

def plot_timings(tests,timings):
    labels = [test.__name__[11:].replace('_','.') for test in tests]
    plt.figure(figsize=(8,5))
    plt.subplot(211)
    plt.bar(labels,timings)
    plt.xticks(rotation=45)
    plt.ylim(0,40)
    plt.subplot(212)
    plt.bar(labels,timings)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.ylim((10**-3,10**2))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    tests = np.array([test_roots_1_1,
                        test_roots_1_3,
                        test_roots_1_4,
                        test_roots_1_5,
                        test_roots_2_1,
                        test_roots_2_2,
                        test_roots_2_3,
                        test_roots_2_4,
                        test_roots_2_5,
                        test_roots_3_1,
                        test_roots_3_2,
                        test_roots_4_1,
                        test_roots_4_2])
    res_passes = np.zeros_like(tests,dtype=bool)
    norm_passes = np.zeros_like(tests,dtype=bool)
    times = np.zeros_like(tests)
    for i,test in enumerate(tests):
        t, passes = test()
        res_pass,norm_pass = passes
        res_passes[i] = res_pass
        norm_passes[i] = norm_pass
        times[i] = t
    print('\n\nSummary')
    print(f'Residual Test: Passed {np.sum(res_passes)} of {len(tests)}, {100*np.mean(res_passes)}%')
    where_failed_res = np.where(~res_passes)[0]
    failed_res_tests = tests[where_failed_res]
    print(f'Failed Residual Test on \n{[t.__name__ for t in failed_res_tests]}')
    print(f'Norm Test    : Passed {np.sum(norm_passes)} of {len(tests)}, {100*np.mean(norm_passes)}%')
    where_failed_norm = np.where(~norm_passes)[0]
    failed_norm_tests = tests[where_failed_norm]
    print(f'Failed Norm Test on \n{[t.__name__ for t in failed_norm_tests]}')
    plot_timings(tests,times)