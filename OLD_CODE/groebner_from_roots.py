'''

Catherine Kellar
Nov 24 17
Groebner basis generator given roots

'''
import numpy as np
import scipy as sp
from scipy.misc import comb

nchoosek = lambda n,k: int(comb(n,k))

def getMonBase(d,n):
    '''base = getMonBase(d,n)
    ---------------------
    Returns a set of monomials of total degree d in n variables. Each row of
    mon refers to a n-tuple of exponents of a monomial whereby each column
    corresponds to a variable.
    
    Inputs:
    d (int): maximum total degree of the monomials in the base
    
    n (int): number of exponents

    returns:
    base (matrix): matrix of lexicographic ordered monomials of degree d and n exponents
    
    example: d= 2, n = 3
    
    base =
     
          2     0     0
          1     1     0
          1     0     1
          0     2     0
          0     1     1
          0     0     2
    '''
    #MATLAB - ';' = supress output
    # x = [1 2] is just an array
    # x = [7;9] => [[7],[9]]
    # x' => x.T
    # x = zeros(6,3) => 6row x 3row of zeros
    if n == 1:
        base = [[d]]
    else:
        base = np.zeros(n)
        base[0] = d
        
        for i in range(d-1,-1,-1):        
            ones_col = np.reshape(i*np.ones(int(comb(d-i+n-2,n-2))),(-1,1))
            square = getMonBase(d-i, n-1)
            bottom_rows = np.hstack((ones_col, square))
            base = np.vstack((base, bottom_rows))
    return np.array(base).astype(int)

def getMon(d,n):
    '''#fullBase = getMon(d,n) or getMon(d,n,d0)
    ----------------------------------------------
    
    Returns a full canonical base of monomials of total degree d and in n
    variables. Each row of mon refers to a n-tuple of exponents of a monomial
    whereby each column corresponds to a variable.
    
    example: d= 2, n = 3
    
    fullBase =
    
         0     0     0
         1     0     0
         0     1     0
         0     0     1
         2     0     0
         1     1     0
         1     0     1
         0     2     0
         0     1     1
         0     0     2
    '''
    if ~isempty(varargin)
        d0 = varargin{1};
        if d0 > d
            fullBase = [];
            return
    else
        d0 = 0

    if d0 == 0
        fullBase = zeros(nchoosek(d+n,n),n);
        rowCounter = 2;
        for i = 1 : d,
            tempbase = getMonBase(i,n);
            fullBase(rowCounter:rowCounter+length(tempbase)-1,:)  = tempbase;
            rowCounter = rowCounter+length(tempbase);
    else
        rowCounter = 1;
        for i = d0 : d,
            tempbase = getMonBase(i,n);
            fullBase(rowCounter:rowCounter+length(tempbase)-1,:)  = tempbase;
            rowCounter = rowCounter+length(tempbase);

    return fullBase
'''
def diffBase(d,root,x):
    #y = diffBase(d,root,x)
    #----------------------
    #
    #Applies the (partial) differential operator on a polynomial base vector
    #of degree 'd' and evaluates it with 'root'. Graded xelicographic ordering
    #is implicitly assumed. Multiple differentiation is also supported, x
    #should then be a vector which indicates in which order there needs to be
    #differentiated.
    #
    #y     =   column vector, contains the evaluated polynomial base vector
    #
    #          d^n1x1 ... d^nnxn  |
    #          ------------------ | 
    #          dx1^n1 ... dxn^nn  |x = root
    #
    #d     =   scalar, degree of the multivariate polynomial base vector
    #
    #root  =   row vector, used to evaluate differentiated base with
    #
    #x     =   row vector, index of the variable to which the differentation needs
    #          to take place,
    #
    #               d        d        d
    #          1 = ---, 2 = ---, 3 = ---, etc...
    #              dx1      dx2      dx3  
    #              
    #          when a higher order differentiation is required then x is a row
    #          vector indicating the order in which the variables need to be
    #          differentiated, eg. x = [1 1 2] means first a 2nd order
    #          differentiation to x1, then 1st order differentiation to x2.
    #
    #EXAMPLE
    #-------
    #
    #second order derivative to y (= Dyy) of degree 6
    #evaluated in the point (2,3):
    #
    #diffBase(6,[2 3],[2 2])
    #
    #first 1st order derivative to y, then 1st order derivative to x (= Dyx),
    #degree 4 and evaluated in (-5,9):
    #
    #diffBase(4,[-5 9],[2 1])
    #
    #CALLS
    #-----
    #
    #getMon.m
    #
    #Kim Batselier, 2010-01-28

    n = size(root,2);
    monBase = getMon(d,n);
    l = length(monBase);
    #coef = zeros(l,1);
    Dn = size(x,2);

    if size(x,2) ~= Dn
        error('The ''x'' argument provided should be a vector indicating for each differentiation step to which variable needs to be differentiated')
    end

    #check function inputs
    if x > n
            error(['You cannot differentiate with respect to x' num2str(x) ', there are only ' num2str(n) ' variables.'])
    end

    coef = ones(l,1);

    #derivative of the monomial base
    DmonBase  = monBase;
    for i = 1 : Dn
        if exist('indices','var')
            clear indices
        end
        #first run of the coefficients
        indices = find(DmonBase(:,x(i)) ~= 0);
        coef(indices) = coef(indices).*DmonBase(indices,x(i));

        DmonBase = DmonBase-[zeros(l,x(i)-1) ones(l,1) zeros(l,n-x(i))];
        
        #negative exponents are manually put to zero
        DmonBase = DmonBase.*(~(DmonBase(:,x(i)) < 0)*ones(1,n));   
    end

    temp = zeros(l,n);
    for i = 1:length(indices)
        temp(indices(i),:) = root.^DmonBase(indices(i),:);
    end

    y = zeros(l,1);
    y = coef.*prod(temp,2)./getDenom(x);

        function denom = getDenom(x)
            xmax = max(x);
            exp = zeros(1,xmax);
            for i = 1 : xmax
                exp(i) = length(find(x== i));            
            end
            denom = prod(factorial(exp));
        end
    end

def makeRoot(d,root):
    #sol = makeRoot(d,root)
    #----------------------
    #
    #Evaluates a multivariate polynomial base vector.
    #
    #sol   =   column vector, contains the evaluated polynomial base vector
    #
    #d     =   scalar, degree of the multivariate polynomial base vector
    #
    #root  =   row vector, used to evaluate differentiated base with
    #
    #CALLS
    #-----
    #
    #getMon.m
    #
    #Kim Batselier, 2010-01

    n = size(root,2);

    monBase = getMon(d,n);

    l = length(monBase);

    for j = 1 : size(root,1)
        temp = zeros(l,n);
        for i = 1 : n
            temp(:,i) = (root(j,i)*ones(l,1)).^monBase(:,i);
        end
        
        sol(:,j) = prod(temp,2);
    end

    end

def getBasis(d, depth, root):
    """calculates the subspace basis for the kernel, used in calculating a groebner basis.
    Uses partial derivatives to achieve this.
    
    input: d(int): degree of monomial base
           depth(int): max dgree of differentiation
           root (array): the root at which we evaluate the derivatives
    return: D(matrix): matrix with each column being the partial evaluated at the root
            I(matrix): indices of differentiation
    """
    n = size(root,2) #2nd dimension of root - finding n for number of variables that polys should be in
    D(:,1) = makeRoot(d,root)

    I = getMon(depth,n);

    for i = 2 : size(I,1)
        temp = ones(1,I(i,1))
        for j = 2 : n
            temp = [temp j*ones(1,I(i,j))] 
        D(:,i) = diffBase(d,root,temp)

    I = I'


def groebner_from_roots(rootList):
    """generates a groebner basis from a list of roots in n dimensions
    
    input: list of roots in n dimensions - must all be unique
    
    return: list of polynomials that form a reduced groebner basis for said roots
    """
    a=[];
    b=1;
    polysys=[];

    m,n = rootList.size;

    mult=np.ones((1,m)); #there are no multiplicities allowed - m multiplicities total
    
    stop=0;
    d=0;

    while not stop
        d+=1;
        # construct kernel K
        # for now, make whole K everytime
        K=[];
        for i in range(1,m):      # for each root
            # determine order of differentiation    
            ddiff=0;
            while sp.misc.comb(ddiff+n,n) < m:
                ddiff+=1;
            ####################
            D=getKSB(d,ddiff,root(i,:));
            ####################
            K=[K D];
            
        indices = nchoosek(d-1+n,n)+1:nchoosek(d+n,n); #indices of all monomials of degree d
        
        # remove multiples of A from indices
        for i=1:length(indices) #for each new monomial
            for j=1:length(a)   #check whether it is multiple of a(j)
                if  sum((fite(indices(i),n)-fite(a(j),n)) >= 0) == n
                    #we found a multiple
                    indices(i) = 0;
                end
            end        
        end
        indices(indices==0)=[];
        if isempty(indices)
            stop=1;
        else
            #canonical decomposition
            for i=1:length(indices)
                [U,S,Z]=svd(full(K([b indices(i)],:)'));
                if size(S,2)==1
                    S=S(1,1);
                else
                    S=diag(S);
                end
                tol=m*S(1)*eps;
                rs=sum(S > tol);
                
                if (S(end) < tol) || (rs < length([b indices(i)]))
                    a=[a indices(i)];
                    temp=zeros(1,nchoosek(d+n,n));
                    temp([b indices(i)])=Z(:,end);
                    temp(abs(temp)<tol)=0;  #remove numerically zero coefficients
                    polysys=[polysys;vec2polysys(temp,n)];
                else
                    b=[b indices(i)];
                end
            end
        end
    end

    end'''
