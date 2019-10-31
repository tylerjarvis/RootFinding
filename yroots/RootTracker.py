import numpy as np

class RootTracker:
    '''
    Class to track the roots that are found found using the subdivision solver.

    Attributes
    ----------
    roots: numpy array
        The roots of the system being solved
    intervals : list
        The intervals that the roots were found in
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
    get_polish_intervals
        Gets the intervals to run the next round of polishing on.
    '''
    def __init__(self):
        self.roots = np.array([])
        self.intervals = []
        self.methods = []

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
            self.roots = np.vstack([self.roots, zeros])
        else:
            self.roots = np.hstack([self.roots, zeros])
        self.intervals += [(a,b)]*len(zeros)
        self.methods += [method]*len(zeros)

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
