import numpy as np

def rootInBox(root, a, b):
    """Checks to see if a root is in a box.

    Parameters
    ----------
    root : numpy array
        The root to check.
    a : numpy array
        The lower bound on the box.
    b : numpy array
        The upper bound on the box.
    Returns
    -------
    rootInBox : bool
        Whether the root is in the box
    """
    return np.all(root > a) and np.all(root < b)

class RootTracker:
    '''
    Class to track the roots that are found found using the subdivision solver.

    Attributes
    ----------
    roots: numpy array
        The roots of the system being solved
    possible_duplicates : list
        Roots that were outside their search interval so might be duplicates
    potential_roots : numpy array
        Places that may or may not have a root that we found.
    intervals : list
        The intervals that the roots were found in.
    polish_intervals : list
        The intervals to run polishing on.
    methods : list
        The methods used to find the roots.

    Methods
    -------
    __init__
        Initializes everything.
    add_roots
        Adds roots the were found to the list, along with their information.
    add_potential_roots
        Adds roots that were found by the solver but are questionable. We will
        want to double check that these roots aren't duplicated elsewhere or that
        they give a fairly good answer.
    get_polish_intervals
        Gets the intervals to run the next round of polishing on.
    '''
    def __init__(self):
        self.roots = np.array([])
        self.possible_duplicates = []
        self.potential_roots = np.array([])
        self.intervals = []
        self.methods = []
        #for tracking condition numbers and gradients
        self.conds = []
        self.grads = []

    def add_roots(self, zeros, a, b, method):
        ''' Store the roots that were found, along with the interval they were found in and the method used.

        Parameters
        ----------
        zeros : numpy array.
            The roots to store.
        a: numpy array
            The lower bounds of the interval the roots were found in.
        b: numpy array
            The upper bounds of the interval the roots were found in.
        method : string
            The method used to find the roots
        '''
        for zero in zeros:
            if rootInBox(zero, a, b):
                self.add_root(zero, a, b, method)
            else:
                found = False
                for a_,b_ in self.intervals:
                    if rootInBox(zero, a_, b_):
                        found = True
                        break
                if not found:
                    self.possible_duplicates.append([zero, a, b, method])

        temp = []
        for zero, a_, b_, method in self.possible_duplicates:
            if rootInBox(zero, a, b):
                pass
            else:
                temp.append([zero, a_, b_, method])
        self.possible_duplicates = temp


#         if not isinstance(a, np.ndarray):
#             dim = 1
#         else:
#             dim = len(a)
#         if len(self.roots) == 0:
#             if dim == 1:
#                 self.roots = np.zeros([0])
#             else:
#                 self.roots = np.zeros([0,dim])

#         if dim > 1:
#             self.roots = np.vstack([self.roots, zeros])
#         else:
#             self.roots = np.hstack([self.roots, zeros])
#         self.intervals += [(a,b)]*len(zeros)
#         self.methods += [method]*len(zeros)

    def add_root(self, zero, a, b, method):
        ''' Store the root that was found, along with the interval it was found in and the method used.

        Parameters
        ----------
        zero : numpy array.
            The root to store.
        a: numpy array
            The lower bounds of the interval the roots were found in.
        b: numpy array
            The upper bounds of the interval the roots were found in.
        method : string
            The method used to find the roots
        '''
        if not isinstance(a, np.ndarray):
            dim = 1
        else:
            dim = len(a)
        if len(self.roots) == 0:
            if dim == 1:
                self.roots = np.zeros([0])
            else:
                self.roots = np.zeros([0,dim])
        if dim > 1:
            self.roots = np.vstack([self.roots, zero])
        else:
            self.roots = np.hstack([self.roots, zero])
        self.intervals += [(a,b)]
        self.methods += [method]

    def add_potential_roots(self, potentials, a, b, method):
        ''' Store the potential roots that were found, along with the interval
        they were found in and the method used.

        Parameters
        ----------
        potentials : numpy array.
            The potential roots to store.
        a: numpy array
            The lower bounds of the interval the roots were found in.
        b: numpy array
            The upper bounds of the interval the roots were found in.
        method : string
            The method used to find the roots
        '''
        if not isinstance(a, np.ndarray):
            dim = 1
        else:
            dim = len(a)
        if len(self.potential_roots) == 0:
            if dim == 1:
                self.potential_roots = np.zeros([0])
            else:
                self.potential_roots = np.zeros([0,dim])

        if dim > 1:
            self.potential_roots = np.vstack([self.potential_roots, potentials])
        else:
            self.potential_roots = np.hstack([self.potential_roots, potentials])
        self.intervals += [(a,b)]*len(potentials)
        self.methods += [method]*len(potentials)

    def get_polish_intervals(self):
        ''' Find the intervals to run the polishing on.

        Deletes the rest of the info as subdivision will be rerun on these intervals.

        returns
        -------
        polish_intervals : list
            The intervals to rerun the search on.
        '''
        polish_intervals = np.unique(self.intervals,axis=0)
        self.intervals = []
        self.roots = []
        self.methods = []
        return polish_intervals

    def keep_possible_duplicates(self):
        ''' Adds the possible duplicate roots to the roots
        '''
        for zero, a, b, method in self.possible_duplicates:
            # Pass in None for the condition number since we don't have it
            self.add_root(zero, a, b, method)
        self.possible_duplicates = []
