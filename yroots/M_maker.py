import numpy as np
from yroots.utils import transform, slice_top
from scipy.fftpack import fftn
from itertools import product
import warnings
import time

class M_maker:
    def __init__(self,f,a,b,guess_deg,max_deg_edit=None,rel_approx_tol=1.e-15, abs_approx_tol=1.e-12):
        """
        Used to find M, an array of Chebyshev coefficients.

        Attributes
        ----------
        dim: int
            the dimension of the space we approximate in
        f: vectorized, callable function
            function from R^n --> R that we approximate
        a: ndarray
            lower bounds on the region
        b: ndarray
            upper bounds on the region
        rel_approx_tol: float
            relative approximation tolerance
        abs_approx_tol: float
            absolute approximation tolerance
        memo_dict: dictionary
            the evaluation of an approximating polynomial at each chebyshev point in the region
            keys: degree, values: array of evaluations
        deg: int
            degree of the approximation
        values_block: ndarray
            evaluation of the approximating polynomial at the chebyshev critical points in the region
        err: float
            error on the approximation
        M: array
            the coefficient tensor
        M2: array
            the coefficient tensor of double degree
        M_rescaled: array 
            the coeffficient tensor divided by inf_norm
        inf_norm: float
            max of the absolute values of the coefficients

        Parameters
        ----------
        f: vectorized, callable function
            function from R^n --> R that we approximate
        a: ndarray
            lower bounds on the region
        b: ndarray
            upper bounds on the region
        guess_deg: int
            user's guess on the degree of approximation
        rescale: bool
            whether to rescale by self.inf_norm or not
        rel_approx_tol: float
            relative approximation tolerance
        abs_approx_tol: float
            absolute approximation tolerance

        Methods
        -------
        error_test:
            determines whether the approximation is sufficiently accurate
        find_good_deg:
            uses error_test by doubling up to a good deg until error_test is passed
        interval_approximate_nd:
            calculates the chebyshev coefficients
        chebyshev_block_copy:
            preparatory step to Fast Fourier Transform, so as to save time complexity in getting the chebyshev coeffs
        block_copy_slicers:
            slicers to make the block copy of the values block
        interval_approx_slicers:
            slicers to make the whole approximation
        """

        self.max_n_coeffs = 500**2
        
        #self.max_deg = {1: 100000, 2:1000, 3:9, 4:9, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2} #need to experiment with this

        max_degs = [2**round(np.log2(self.max_n_coeffs**(1/k))) for k in range(1,11)]
        # Results in max_degs = [262144,512,64,16,16,8,8,4,4,4]
        max_degs[3] = 32
        dims = list(range(1,11))
        self.max_deg = dict(zip(dims,max_degs)) # dictionary to lookup proper max_deg for dims 1-10
    
        
        dim = len(a)
        if dim != len(b):
            raise ValueError("dimension mismatch")
        self.dim = dim # self.dim is now the length of the lists a, b of boundaries for the interval

        #Update self.max_deg if this instance was given a manually chosen max_deg
        if max_deg_edit is not None:
            if max_deg_edit > self.max_deg[self.dim]:
                warnings.warn("Terminating approximation at high degree--run time may be prolonged.")
            elif max_deg_edit < (1/10)*self.max_deg[self.dim]: #TODO: we can get creative about edge cases here, let's think about this soon
                warnings.warn("Terminating at low degree--approximation may be inaccruate")
            self.max_deg[self.dim] = max_deg_edit

        #Instantiate other attributes from given parameters.
        self.f = f
        self.a = a
        self.b = b
        self.rel_approx_tol = rel_approx_tol
        self.abs_approx_tol = abs_approx_tol
        self.memo_dict = {}
        self.fft_dict = {}

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
        Calculates the error of the approximation.
        Subtracts the n x n matrix for the approximation (M) from the top left corner
            of the 2n x 2n matrix for an approximation of double degree (M2) and then returns
            the sum of the absolute value of the resulting entries.

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
        coeff2[slice_top(M.shape)] -= M #slice_top imported from yroots.utils
        return np.sum(np.abs(coeff2))

    def error_test(self,error,abs_approx_tol,rel_approx_tol,inf_norm): 
        """
        Determines whether the approximation is within the error tolerance.
        Compares the error given with 1) the relative approximation tolerance and 2) a potential
            floating point error given by the product of the number of additional entries in M2,
            the inf_norm, and 10 * machine epsilon.
        Returns True if error is less than both 1 and 2; returns False otherwise.

        Parameters
        ----------
        error: float
            the absolute value of the difference of the sum of abs values of M and M2
        abs_approx_tol: float
            absolute approximation tolerance
        rel_approx_tol: float
            relaive approximation tolerance
        inf_norm: float
            the sup norm on the approximation

        Returns
        -------
        (bool) if the error test has been passed or not
        """
        mach_eps = np.finfo(float).eps
        num_entries_M2 = np.product(self.M2.shape) - np.product(self.M.shape) #number of additional entries from M2
        potential_float_err = 10*num_entries_M2*mach_eps*inf_norm #potential float error
        return error < max(rel_approx_tol,potential_float_err) #we decided abs_approx_tol is not needed

    def find_good_approx(self,f,deg,dim,a,b):
        """
        Finds the right degree with which to approximate on the interval.

        Parameters
        ----------
        f : function from R^n -> R
            the function to interpolate.
        deg : numpy array
            the degree of the interpolation in each dimension.
        dim: int
            dimension
        a : numpy array
            lower bound on the interval.
        b : numpy array
            upper bound on the interval.

        Returns
        -------
        deg: the correct approximation degree
        """

        # Get the approximation for degree deg (M) and degree 2*deg (M2), then use them to find the error. 
        print("Beginning approximation for function!")
        start = time.time()
        self.M, self.inf_norm = self.interval_approximate_nd(f, a, b, deg, return_inf_norm=True)
        self.M2 = self.interval_approximate_nd(f,a,b,deg*2)
        self.err = self.get_err(self.M,self.M2)

        #Set deg to self.max_deg if it exceeds the maximum degree allowed.
        if deg > self.max_deg[dim]:
            deg = self.max_deg[dim]

        while deg < self.max_deg[dim]: # Check for a better approximation if deg has not yet reached max.
            
            # If the approximation is already good enough by the error test, leave the loop.
            if self.error_test(self.err,self.abs_approx_tol,self.rel_approx_tol,self.inf_norm):
                break

            # If increasing deg by a factor of 2 would exceed the maximum degree,
                # set deg to the maximum allowed, get the error from the corresponding M
                #and M2, warn the user of the error, and break from the loop.
            elif 2*deg > self.max_deg[dim]:
                deg = self.max_deg[dim]
                self.M, self.inf_norm = self.interval_approximate_nd(f, a, b, deg, return_inf_norm=True)
                self.M2 = self.interval_approximate_nd(f,a,b,deg*2)
                self.err = self.get_err(self.M,self.M2)
                warnings.warn(f"Hit Max Degree in Approximation! Approximation error is {self.err}!")
                break

            else: # The approximation can be improved by a 2*deg approximation without exceeding max_deg.
                # Increase deg by a factor of 2, and get the error from the corresponding M and M2.
                deg = 2*deg
                self.M, self.inf_norm = self.interval_approximate_nd(f, a, b, deg, return_inf_norm=True)
                self.M2 = self.interval_approximate_nd(f,a,b,deg*2)
                self.err = self.get_err(self.M,self.M2)
        else: # The loop terminated because deg reached max_deg, so warn the user of the error.
            warnings.warn(f"Hit Max Degree in Approximation!  Approximation error is {self.err}!")
            
        #Update self.deg
        self.deg = deg
        print("Finished approximating function!")
        print(f"Total time taken:  {round(time.time()-start,2)} seconds.")
        print()

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

        if deg in self.fft_dict.keys():
            if return_inf_norm:
                return self.fft_dict[deg][0], self.fft_dict[deg][1]
            else:
                return self.fft_dict[deg][0]

        if hasattr(f,"evaluate_grid"):
            cheb_values = np.cos(np.arange(deg+1)*np.pi/deg) #extremizers of Chebyshev approximation
            chepy_pts =  np.column_stack([cheb_values]*self.dim) # dim-dimensional array of Chevy points
            #two dimensions
            #odds = chepy_pts[:,1::2]
            #xyz = np.column_stack((chepy_pts,odds))
            #p1 = f.evaluate_grid(xyz)
            #xyz2 = np.columns_stack((odds,odds))
            #p2 = f.evaluate_grid(xyz2)
            print(f"Recalculating at deg {deg}")
            cheb_pts = transform(chepy_pts,a,b) # transforms the points from [-1,1] interval to [a,b]
            values_block = f.evaluate_grid(cheb_pts) # gets function values; TODO: memoization?
            self.memo_dict[deg] = values_block # saves values at these extremizers for future reference
        
        else: # Create and transform to [a,b] a matrix whose rows are exactly coordinates of each extremizer
            cheb_vals = np.cos(np.arange(deg+1)*np.pi/deg) #calculate extremizers
            cheb_grid = np.meshgrid(*([cheb_vals]*self.dim),indexing='ij')
            flatten = lambda x: x.flatten()
            cheby_pts = np.column_stack(tuple(map(flatten, cheb_grid))) # contains every possible coordinate
            cheb_pts = transform(cheby_pts,a,b) # transforms the points from [-1,1] interval to [a,b]
            
            #Check to see if some/all of the function values at the extremizers have already been saved
            if deg in self.memo_dict.keys():
                values_block = self.memo_dict[deg]
            elif half_deg in self.memo_dict.keys():
                """Some of the values have already been calculated for deg/2, so
                    find these values, create masks to mark their indices, and fill them in.
                    Calculate the remaining values and fill them in as well."""
                half_deg_arr = self.memo_dict[half_deg].flatten() # list of all the values calculated for deg/2 
                slices = tuple([slice(0, deg+1,2)]*self.dim) # indexes of already calculated values
                mask = np.ones([deg+1]*self.dim,dtype=bool) 
                mask[slices] = False #True for all values that still need calculation, False otherwise
                unknowns_mask = mask.flatten() # unknown values = True
                knowns_mask = ~unknowns_mask # known values = True
                values_arr = np.empty((deg+1)**self.dim)
                values_arr[knowns_mask] = half_deg_arr # fill with carried over values from deg = n/2
                values_arr[unknowns_mask] = f(*cheb_pts[unknowns_mask].T) #calculate and fill remaining values
                values_block = values_arr.reshape(*([deg+1]*self.dim))
            else: # No previous calculations to use, so find the entire block from scratch.
                values_block = f(*cheb_pts.T).reshape(*([deg+1]*self.dim))
            
            # Save the calculated values of the function at the Chebyshev points for future reference.
            if save_values_block == True:
                self.values_block = values_block
            self.memo_dict[deg] = values_block

        values = self.chebyshev_block_copy(values_block)

        inf_norm = np.max(np.abs(values)) # largest absolute value of the function at any Chebyshev pt

        # Set up and perform the Fast Fourier Transform to find the Chebyshev approximation
        start = time.time()
        x0_slicer, deg_slicer, slices, rescale = self.interval_approx_slicers(self.dim,deg)
        coeffs = fftn(values/rescale).real
        for x0sl, degsl in zip(x0_slicer, deg_slicer): # Divide constant and last coefficients in each dimension by 2 as required
            coeffs[x0sl] /= 2
            coeffs[degsl] /= 2
        print(f"    deg {deg}: FFT took  {round(time.time()-start,2)} seconds.")
        self.fft_dict[deg] = [coeffs[tuple(slices)], inf_norm]

        # Return the found appoximation
        if return_inf_norm:
            return coeffs[tuple(slices)], inf_norm
        else:
            return coeffs[tuple(slices)]
    
    def chebyshev_block_copy(self,values_block):
        """Takes in a tensor of function evaluation values and copies these values to
            a new tensor appropriately to prepare for chebyshev interpolation.

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

        # Create an empty dim dimensional matrix with each dimension 2*deg (2*deg x 2*deg x ... x 2*deg)
        values_cheb = np.empty(tuple([2*deg])*dim, dtype=np.float64)
        block_slicers, cheb_slicers, slicer = self.block_copy_slicers(dim, deg) # get slicers

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
        deg : int
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
        into the output of the fft and divides some of the values by 2 and turn them into
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