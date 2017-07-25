import numpy as np
import heapq

class Term(object):
    '''
    Terms are just tuples of exponents with the grevlex ordering 
    '''
    def __init__(self,val):
        self.val = tuple(val)

    def __repr__(self):
        return str(self.val) + ' with grevlex order'

    def __lt__(self, other, order = 'grevlex'):
        '''
        Redfine less-than according to grevlex
        '''
        if order == 'grevlex': #Graded Reverse Lexographical Order
            if sum(self.val) < sum(other.val):
                return True
            elif sum(self.val) > sum(other.val):
                return False
            else:
                for i,j in zip(reversed(self.val),reversed(other.val)):
                    if i<j:
                        return False
                    if i > j:
                        return True
                return False
        elif order == 'lexographic': #Lexographical Order
            for i,j in zip(self.val,other.val):
                if i<j:
                    return True
                if i>j:
                    return False
            return False
        elif order == 'grlex': #Graded Lexographical Order
            if sum(self.val) < sum(other.val):
                return True
            elif sum(self.val) > sum(other.val):
                return False
            else:
                for i,j in zip(self.val,other.val):
                    if i<j:
                        return True
                    if i > j:
                        return False
                return False


    # Define the other relations in grevlex order   
        
    def __eq__(self, other):
        return self.val == other.val

    def __gt__(self, other):
        return not(self < other or self == other)
        
    def __ge__(self, other):
        return (self > other or self == other)

    def __le__(self,other):
        return (self < other or self == other)
    
    #Makes terms hashable so they can go in a set
    def __hash__(self):
        return hash(self.val)

class Term_w_InvertedOrder(Term):
    '''
    Called by MaxHeap object to reverse the ordering for a min heap
    Used exclusively with Terms
    '''
    def __init__(self,term):
        '''
        Takes in a Term.  val is the underlying tuple, term is the underlying term 
        '''
        self.val = term.val
        self.term = term

    # Invert the order 
    
    def __lt__(self,other): return self.term > other.term
    def __le__(self,other): return (self.term > other.term or self.term == other.term)
    def __ge__(self,other): return (self.term < other.term or self.term == other.term)
    def __gt__(self,other): return (self.term < other.term)
    
    def __repr__(self): 
        return str(list(self.val)) + ' with inverted grevlex order'


class MaxHeap(object):
    '''
    Implementation of a set max-priority queue--one that only adds 
    terms to the queue if they aren't there already
    
    Incoming and outgoing objects are all Terms (not Term_w_InvertedOrder)
    '''
    
    def __init__(self): 
        self.h = []         # empty heap
        self._set = set()   # empty set (of things already in the heap)

    def heappush(self, x): 
        if not x.val in self._set:       # check if already in the set
            x = Term_w_InvertedOrder(x)
            heapq.heappush(self.h,x)     # push with InvertedOrder
            self._set.add(x.val)         # but use the tuple in the set (it is easily hashable) 
        else:
            pass
            #print(x, 'is a duplicate')

    def heappop(self): 
        term = heapq.heappop(self.h).term   # only keep the original term--without the InvertedOrder
        self._set.discard(term.val)
        return term
    
    def __getitem__(self, i): 
        return self.h[i].term

    def __len__(self): 
        return len(self.h)

    def __repr__(self):
        return('A max heap of {} unique terms with the DegRevLex term order.'.format(len(self)))

class MinHeap(MaxHeap):
    '''
    Implementation of a set min-priorioty queue.
    
    '''

    def heappush(self,x): 
        ## Same as MaxHeap push, except that the term order is not inverted
        if not x in self._set:
            heapq.heappush(self.h, x)
            self._set.add(x)
        else:
            pass
        
    def heappop(self): 
        """ Same as MaxHeap pop except that the term itself IS the underlying term.
        """
        term = heapq.heappop(self.h)   
        self._set.discard(term.val)
        return term
    
    def __getitem__(self, i): 
        """ Same as MaxHeap getitem except that the term itself IS the underlying term.
        """
        return self.h[i]

    def __repr__(self):
        return('A min heap of {} unique terms with the DegRevLex term order.'.format(len(self)))

