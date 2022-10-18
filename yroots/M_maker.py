import numpy as np
from yroots.utils import transform, slice_top
from scipy.fftpack import fftn
from itertools import product

class M_maker:
    def __init__(self,f,a,b,guess_deg,rel_approx_tol=1.e-15, abs_approx_tol=1.e-12):
        """
        Used to find M, an array of Chebyshev coefficients.

        Attributes
        ----------
        dim: int
        the dimension of the space we approximate in
        f: vectorized, callable function
        the function from R^n --> R that we approximate
        a: ndarray
        the lower bounds on the region
        b: ndarray
        the upper bounds on the region
        rel_approx_tol: float
        relative approximation tolerance
        abs_approx_tol: float
        absolute approximation tolerance
        memo_dict: dictionary
        the evaluation of an approximating polynomial at each chebyshev point in the region
        keys are degree, values are arrays of evaluations
        deg: int
        the degree of the approximation
        values_block: ndarray
        the evaluation of the approximating polynomial at the chebyshev critical points in the region
        err: float
        the error on the approximation
        M: array
        the coefficient tensor
        M2: array
        the coefficient tensor of double degree
        M_rescaled: array 
        the coeffficient tensor divided by inf_norm
        inf_norm: float
        the max of the absolute values of the coefficients

        Parameters
        ----------
        f: vectorized, callable function
        the function from R^n --> R that we approximate
        a: ndarray
        the lower bounds on the region
        b: ndarray
        the upper bounds on the region
        guess_deg: int
        the user's guess on the degree of approximation
        rescale: bool
        whether to rescale by self.inf_norm or not
        rel_approx_tol: float
        relative approximation tolerance
        abs_approx_tol: float
        absolute approximation tolerance

        Methods
        -------
        error_test: determines whether the approximation is sufficiently accurate
        find_good_deg: uses error_test by doubling up to a good deg until error_test is passed
        interval_approximate_nd: calculates the chebyshev coefficients
        chebyshev_block_copy: preparatory step to Fast Fourier Transform, so as to save time complexity in getting the chebyshev coeffs
        block_copy_slicers: slicers to make the block copy of the values block
        interval_approx_slicers: slicers to make the whole approximation
        """
        self.max_deg = {1: 100000, 2:1000, 3:9, 4:9, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2} #need to experiment with this
        
        dim = len(a)
        if dim != len(b):
            raise ValueError("dimension mismatch")
        self.dim = dim
        self.f = f
        self.a = a
        self.b = b
        self.rel_approx_tol = rel_approx_tol
        self.abs_approx_tol = abs_approx_tol
        self.memo_dict = {}

        self.values_block = None
        self.M = None
        self.M2 = None
        self.err = None
        self.inf_norm = None

        self.find_good_approx(f,guess_deg,dim,a,b)        
        self.M_rescaled = self.M / self.inf_norm
        self.values_block = list(self.memo_dict.values())[-2]

    def get_err(self,M,M2):
        """
        Calculates the error of the approximation

        Parameters
        ----------
        M: array
        The coefficient tensor
        M2: array
        The coefficient tensor for a double degree approximation

        Returns
        -------
        (float) the error
        """
        coeff2 = M2.copy()
        coeff2[slice_top(M.shape)] -= M
        return np.sum(np.abs(coeff2))

    def error_test(self,error,abs_approx_tol,rel_approx_tol,inf_norm): 
        """
        Determines whether the approximation is within the error tolerance

        Parameters
        ----------
        error: float
        The absolute value of the difference of the sum of abs values of M and M2
        rel_approx_tol: float
        abs_approx_tol: float
        inf_norm: float
        the sup norm on the approximation

        Returns
        -------
        (bool) if the error test has been passed or not
        """
        return error < abs_approx_tol+rel_approx_tol*inf_norm

    def find_good_approx(self,f,deg,dim,a,b):
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
        self.M, self.inf_norm = self.interval_approximate_nd(f, a, b, deg, return_inf_norm=True)
        self.M2 = self.interval_approximate_nd(f,a,b,deg*2)
        self.err = self.get_err(self.M,self.M2)

        if deg >= self.max_deg[dim]:
            deg = self.max_deg[dim]

        while deg < self.max_deg[dim]:
            if self.error_test(self.err,self.abs_approx_tol,self.rel_approx_tol,self.inf_norm):
                break
            elif 2*deg > self.max_deg[dim]:
                deg = self.max_deg[dim]

                self.M, self.inf_norm = self.interval_approximate_nd(f, a, b, deg, return_inf_norm=True)
                self.M2 = self.interval_approximate_nd(f,a,b,deg*2)
                self.err = self.get_err(self.M,self.M2)
                
                break
            else:
                deg = 2*deg

                self.M, self.inf_norm = self.interval_approximate_nd(f, a, b, deg, return_inf_norm=True)
                self.M2 = self.interval_approximate_nd(f,a,b,deg*2)
                self.err = self.get_err(self.M,self.M2)

        self.deg = deg

    def interval_approximate_nd(self,f, a, b, deg, return_inf_norm=False, save_values_block=False):
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
            The degree of the interpolation in each dimension.
        return_inf_norm : bool
            whether to return the inf norm of the function
        save_values_block : bool
            whether to save the values block as an attribute

        Returns
        -------
        coeffs : numpy array
            The coefficient of the chebyshev interpolating polynomial.
        inf_norm : float
            The inf_norm of the function
        """
        half_deg = deg / 2

        if hasattr(f,"evaluate_grid"):
            cheb_values = np.cos(np.arange(deg+1)*np.pi/deg)
            chepy_pts =  np.column_stack([cheb_values]*self.dim)
            cheb_pts = transform(chepy_pts,a,b)
            values_block = f.evaluate_grid(cheb_pts) #version 0 no memoization here
            self.memo_dict[deg] = values_block
        else:
            cheb_vals = np.cos(np.arange(deg+1)*np.pi/deg)
            cheb_grid = np.meshgrid(*([cheb_vals]*self.dim),indexing='ij')
            flatten = lambda x: x.flatten()
            cheby_pts = np.column_stack(tuple(map(flatten, cheb_grid)))
            cheb_pts = transform(cheby_pts,a,b)
            
            if deg in self.memo_dict.keys():
                values_block = self.memo_dict[deg]

            elif half_deg in self.memo_dict.keys():
                half_deg_arr = self.memo_dict[half_deg].flatten()
                slices = tuple([slice(0, deg+1,2)]*self.dim)
                mask = np.ones([deg+1]*self.dim,dtype=bool)
                mask[slices] = False 
                unknowns_mask = mask.flatten() #this mask will say where the unknown stuff is in the array
                knowns_mask = ~unknowns_mask #this mask will say where the known stuff is
                values_arr = np.empty((deg+1)**self.dim)
                values_arr[knowns_mask] = half_deg_arr
                values_arr[unknowns_mask] = f(*cheb_pts[unknowns_mask].T)
                values_block = values_arr.reshape(*([deg+1]*self.dim))
            else:
                values_block = f(*cheb_pts.T).reshape(*([deg+1]*self.dim))
            
            if save_values_block == True:
                self.values_block = values_block
            
            self.memo_dict[deg] = values_block

        values = self.chebyshev_block_copy(values_block)

        if return_inf_norm:
            inf_norm = np.max(np.abs(values))

        x0_slicer, deg_slicer, slices, rescale = self.interval_approx_slicers(self.dim,deg)
        coeffs = fftn(values/rescale).real

        for x0sl, degsl in zip(x0_slicer, deg_slicer):
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
        dim = values_block.ndim
        deg = values_block.shape[0] - 1
        values_cheb = np.empty(tuple([2*deg])*dim, dtype=np.float64)
        block_slicers, cheb_slicers, slicer = self.block_copy_slicers(dim, deg)

        for cheb_idx, block_idx in zip(cheb_slicers, block_slicers):
            try:
                values_cheb[cheb_idx] = values_block[block_idx]
            except ValueError as e:
                if str(e)[:42] == 'could not broadcast input array from shape': 
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
        x0_slicer = [tuple([slice(None) if i != d else 0 for i in range(dim)])
                    for d in range(dim)]
        deg_slicer = [tuple([slice(None) if i != d else deg for i in range(dim)])
                    for d in range(dim)]
        slices = tuple([slice(0, deg+1)]*dim)
        return x0_slicer, deg_slicer, slices, deg**dim