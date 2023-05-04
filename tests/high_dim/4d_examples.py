import numpy as np
import yroots as yr
import scipy as sp
import matplotlib
from yroots.Combined_Solver import solve
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time
from mpmath import mp, mpmathify
mp.dps = 50

def residuals(f,g,h,f4,roots,t):
    Resid = list()
    Root = list()
    for i in range(len(roots)):
        for j in range(4):
            Root.append(roots[i,j])
        Resid.append(np.abs(f(Root[0],Root[1],Root[2],Root[3])))
        Resid.append(np.abs(g(Root[0],Root[1],Root[2],Root[3])))
        Resid.append(np.abs(h(Root[0],Root[1],Root[2],Root[3])))
        Resid.append(np.abs(f4(Root[0],Root[1],Root[2],Root[3])))
        Root = []

    hours = int(t // 3600)
    minutes = int((t%3600) // 60)
    seconds = int((t%3600)%60 // 1)
    msecs = int(np.round((t % 1) * 1000,0))
    if Resid != []:
        max_resid = np.amax(Resid)
    else:
        max_resid = None
    print("time elapsed: ",hours,"hours,", minutes,"minutes,",seconds, "seconds,",msecs, "milliseconds")
    print("Residuals: ", Resid, "\n")
    print("Max Residual: ", max_resid)
    return max_resid

def plot_resids(residuals):
    plt.scatter([i+1 for i in range(18)],residuals)
    plt.ylim(1e-20,1e-7)
    plt.xticks(range(1, 19, 2))
    plt.yscale('log')
    plt.axhline(y=2.22044604925031e-13,c='r')
    plt.xlabel('example #')
    plt.ylabel('max residual')
    plt.title('max Residuals for 3d examples (log scale)')
    plt.show()
    return

def newton_polish(funcs, derivs, roots):
    niter = 100
    tol = 1e-32
    new_roots = []

    for root in roots:
        i = 0
        x0, x1 = root, root
        while True:
            if i == niter:
                break
            A = np.array([derivs[j](mp.mpf(x0[0]), mp.mpf(x0[1]), mp.mpf(x0[2]), mp.mpf(x0[3])) for j in range(4)])
            B = np.array([mpmathify(funcs[j](mp.mpf(x0[0]), mp.mpf(x0[1]), mp.mpf(x0[2]), mp.mpf(x0[3]))) for j in range(4)])
            delta = np.array(mp.lu_solve(A, -B))
            norm = mp.norm(delta)
            x1 = delta + x0
            if norm < tol:
                break
            x0 = x1
            i += 1
        new_roots.append(x1)
    return np.array(new_roots)


def ex0(polish=False):
    f1 = lambda x1,x2,x3,x4 : x1
    f2 = lambda x1,x2,x3,x4 : x1 + x2
    f3 = lambda x1,x2,x3,x4 : x1 + x2 + x3
    f4 = lambda x1,x2,x3,x4 : x1 + x2 + x3 + x4

    start = time()
    roots = solve([f1,f2,f3,f4], -np.ones(4), np.ones(4), constant_check=False)
    t = time() - start
    print(roots)
    if polish:
        roots = newton_polish([f1,f2,f3,f4], dex0(), roots)
    print("======================= ex 0 =======================")
    return residuals(f1,f2,f3,f4,roots,t)

def dex0():
    df1 = lambda x1, x2, x3, x4: (1, 0, 0, 0)
    df2 = lambda x1, x2, x3, x4: (1, 1, 0, 0)
    df3 = lambda x1, x2, x3, x4: (1, 1, 1, 0)
    df4 = lambda x1, x2, x3, x4: (1, 1, 1, 1)
    return df1, df2, df3, df4

def ex1(polish=False):
    f1 = lambda x1,x2,x3,x4 : np.sin(x1*x3) + x1*np.log(x2+3) - x1**2
    f2 = lambda x1,x2,x3,x4 : np.cos(4*x1*x2) + np.exp(3*x2/(x1-2)) - 5
    f3 = lambda x1,x2,x3,x4 : np.cos(2*x2) - 3*x3 + 1/(x1-8)
    f4 = lambda x1,x2,x3,x4 : x1 + x2 - x3 - x4

    print("Timing start")
    start = time()
    roots = solve([f1,f2,f3,f4], -np.ones(4), np.ones(4))
    print("Timing finished")
    t = time() - start
    if polish:
        roots = polish([f1,f2,f3,f4], dex1(), roots)
        print(roots)
    print("======================= ex 1 =======================")
    return residuals(f1,f2,f3,f4,roots,t)

def dex1():
    df1 = lambda x1, x2, x3, x4 : (x3*np.cos(x1*x3) + np.log(x2 + 3) - 2*x1, x1/(x2 + 3), x1*np.cos(x1*x3), 0)
    df2 = lambda x1, x2, x3, x4 : (-4*x2*np.sin(4*x1*x2) - (3*x2/(x1 - 2)**2)*np.exp(3*x2/(x1 - 2)), 
                                   -4*x1*np.sin(4*x1*x2) + (3/(x1 - 2))*np.exp(3*x2/(x1 - 2)), 0, 0)
    df3 = lambda x1, x2, x3, x4 : (-1/(x1 - 8)**2, -2*np.sin(2*x2), -3, 0)
    df4 = lambda x1, x2, x3, x4 : (1, 1, -1, -1)
    return df1, df2, df3, df4

def ex2():
    f = lambda x,y,z,x4: np.cosh(4*x*y) + np.exp(z)- 5
    g = lambda x,y,z,x4: x - np.log(1/(y+3))
    h = lambda x,y,z,x4: x**2 -  z
    f4 = lambda x,y,z,x4: x + y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], -np.ones(4), np.ones(4))
    t = time() - start
    return residuals(f,g,h,f4,roots,t)

def dex2():
    df = lambda x, y, z, x4 : (4*y*np.sinh(4*x*y), 4*x*np.sinh(4*x*y), np.exp(z), 0)
    dg = lambda x, y, z, x4 : (1, 1/(y+3), 0, 0)
    dh = lambda x, y, z, x4 : (2*x, 0, -1, 0)
    df4 = lambda x, y, z, x4 : (1, 1, -1, -1)
    return df, dg, dh, df4

def ex3():
    f = lambda x,y,z,x4: y**2-x**3
    g = lambda x,y,z,x4: (y+.1)**3-(x-.1)**2
    h = lambda x,y,z,x4: x**2 + y**2 + z**2 - 1
    f4 = lambda x,y,z,x4: x + y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], -np.ones(4), np.ones(4))
    t = time() - start
    return residuals(f,g,h,f4,roots,t)

def dex3():
    df = lambda x, y, z, x4 : (-3*x**2, 2*y, 0, 0)
    dg = lambda x, y, z, x4 : (2*(x-.1), 3*(y+.1)**2, 0, 0)
    dh = lambda x, y, z, x4 : (2*x, 2*y, 2*z, 0)
    df4 = lambda x, y, z, x4 : (1, 1, 1, 1)
    return df, dg, dh, df4

def ex4():
    f = lambda x,y,z,x4: 2*z**11 + 3*z**9 - 5*z**8 + 5*z**3 - 4*z**2 - 1
    g = lambda x,y,z,x4: 2*y + 18*z**10 + 25*z**8 - 45*z**7 - 5*z**6 + 5*z**5 - 5*z**4 + 5*z**3 + 40*z**2 - 31*z - 6
    h = lambda x,y,z,x4: 2*x - 2*z**9 - 5*z**7 + 5*z**6 - 5*z**5 + 5*z**4 - 5*z**3 + 5*z**2 + 1
    f4 = lambda x,y,z,x4: x - y - z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], -np.ones(4), np.ones(4))
    t = time() - start
    return residuals(f,g,h,f4,roots,t)

def dex4():
    df = lambda x, y, z, x4 : (0, 0, 22*z**10 + 27*z**8 - 40*z**7 + 15*z**2 - 8*z, 0)
    dg = lambda x, y, z, x4 : (0, 2, 180*z**9 + 200*z**7 - 315*z**6 - 30*z**5 + 25*z**4 - 10*z**3 + 15*z**2 + 80*z - 31, 0)
    dh = lambda x, y, z, x4 : (2, 0, -18*z**8 - 35*z**6 + 30*z**5 - 25*z**4 + 20*z**3 - 15*z**2 + 10*z, 0)
    df4 = lambda x, y, z, x4 : (1, -1, -1, 1)

def ex5():
    f = lambda x,y,z,x4: np.sin(4*(x + z) * np.exp(y))
    g = lambda x,y,z,x4: np.cos(2*(z**3 + y + np.pi/7))
    h = lambda x,y,z,x4: 1/(x+5) - y
    f4 = lambda x,y,z,x4: x - y + z - x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], -np.ones(4), np.ones(4))
    t = time() - start
    return residuals(f,g,h,f4,roots,t)

def dex5():
    df = lambda x, y, z, x4 : (4*np.exp(y)*np.sin(4*(x + z)*np.exp(y)), 4*(x + z)*np.exp(y)*np.cos(4*(x + z)*np.exp(y)), 
                               4*np.exp(y)*np.sin(4*(x + z)*np.exp(y)), 0)
    dg = lambda x, y, z, x4 : (0, -2*np.sin(2*(z**3+y+np.pi/7)), -6*z**2*np.sin(2*(z**3 + y + np.pi/7)), 0)
    dh = lambda x, y, z, x4 : (-1/(x + 5)**2, -1, 0, 0)
    df4 = lambda x, y, z, x4 : (1, -1, 1, -1)

#returns known residual of Rosenbrock in 4d
def ex6():
    return 1.11071152275615E-10

def ex7():
    f = lambda x,y,z,x4: np.cos(10*x*y)
    g = lambda x,y,z,x4: x + y**2
    h = lambda x,y,z,x4: x + y - z
    f4 = lambda x,y,z,x4: x - y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], -np.ones(4), np.ones(4))
    t = time() - start
    return residuals(f,g,h,f4,roots,t)

def dex7():
    df = lambda x, y, z, x4 : (10*y*np.cos(10*x*y), 10*x*np.cos(10*x*y), 0, 0)
    dg = lambda x, y, z, x4 : (1, 2*y, 0, 0)
    dh = lambda x, y, z, x4 : (1, 1, -1, 0)
    df4 = lambda x, y, z, x4 : (1, -1, 1, 1)

def ex8():
    f = lambda x,y,z,x4: np.exp(2*x)-3
    g = lambda x,y,z,x4: -np.exp(x-2*y) + 11
    h = lambda x,y,z,x4: x + y + 3*z
    f4 = lambda x,y,z,x4: x + y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], -np.ones(4), np.ones(4))
    t = time() - start
    return residuals(f,g,h,f4,roots,t)

def dex8():
    df = lambda x, y, z, x4 : (2*np.exp(2*x), 0, 0, 0)
    dg = lambda x, y, z, x4 : (-np.exp(x - 2*y), 2*np.exp(x - 2*y), 0, 0)
    dh = lambda x, y, z, x4 : (1, 1, 3, 0)
    df4 = lambda x, y, z, x4 : (1, 1, 1, 1)

def ex9():
    f1 = lambda x,y,z,x4: 2*x / (x**2-4) - 2*x
    f2 = lambda x,y,z,x4: 2*y / (y**2+4) - 2*y
    f3 = lambda x,y,z,x4: 2*z / (z**2-4) - 2*z
    f4 = lambda x,y,z,x4: 2*x4 / (x4 **2 - 4) - 2*x4

    start = time()
    roots = solve([f1,f2,f3], -np.ones(4), np.ones(4))
    t = time() - start
    return residuals(f1,f2,f3,f4,roots,t)

def dex9():
    df1 = lambda x, y, z, x4 : (2*(x**2 + 4)/(x**2 - 4)**2 - 2, 0, 0, 0)
    df2 = lambda x, y, z, x4 : (0, 2*(y**2 - 4)/(y**2 + 4)**2 - 2, 0, 0)
    df3 = lambda x, y, z, x4 : (0, 0, 2*(z**2 + 4)/(z**2 - 4)**2 - 2, 0)
    df4 = lambda x, y, z, x4 : (0, 0, 0, x*(x4**2 + 4)/(x4**2 - 4)**2 - 2)

def ex10():
    f = lambda x,y,z,x4: 2*x**2 / (x**4-4) - 2*x**2 + .5
    g = lambda x,y,z,x4: 2*x**2*y / (y**2+4) - 2*y + 2*x*z
    h = lambda x,y,z,x4: 2*z / (z**2-4) - 2*z
    f4 = lambda x,y,z,x4:x + y + z + x4

    start = time()
    roots = solve([f, g, h, f4], np.array([-1,-1,-1,-1]), np.array([1,1,.8,1]))
    t = time() - start
    return residuals(f,g,h,f4,roots,t)

def dex10():
    df = lambda x, y, z, x4 : (-4*x*(x**4 + 4)/(x**4 - 4)**2, 0, 0, 0)
    dg = lambda x, y, z, x4 : (4*x*y/(y**2 + 4) + 2*z, -2*x**2*(y**2 - 4)/(y**2 + 4)**2 - 2, 2*x, 0)
    dh = lambda x, y, z, x4 : (0, 0, -2*(z**2 + 4)/(z**2 - 4)**2 - 2, 0)
    df4 = lambda x, y, z, x4 : (1, 1, 1, 1)
    return df, dg, dh, df4

def ex11():
    f = lambda x,y,z,x4: 144*((x*z)**4+y**4)-225*((x*z)**2+y**2) + 350*(x*z)**2*y**2+81
    g = lambda x,y,z,x4: y-(x*z)**6
    h = lambda x,y,z,x4: (x*z)+y-z
    f4 = lambda x,y,z,x4:-x - y + z - x4

    start = time()
    roots = solve([f,g,h,f4],np.array([-1,-1,-2,-1]),np.array([1,1,2,1]))
    t = time() - start
    return residuals(f,g,h,f4,roots,t)

def dex11():
    df = lambda x, y, z, x4 : (576*(x**3*z**4) - 450*x*z**2 + 700*x*z**2*y**2, 576*y**3 - 450*y + 700*x**2*z**2*y, 576*x**4*z**3 - 450*x**2*z + 700*x**2*z*y**2, 0)
    dg = lambda x, y, z, x4 : (-6*x**5*z**6, 1, -6*x**6*z**5, 0)
    dh = lambda x, y, z, x4 : (z, 1, x - 1, 0)
    df4 = lambda x, y, z, x4 : (-1, -1, 1, -1)
    return df, dg, dh, df4

def ex12():
    f = lambda x,y,z,x4: x**2+y**2-.49**2
    g = lambda x,y,z,x4: (x-.1)*(x*y - .2)
    h = lambda x,y,z,x4: x**2 + y**2 - z**2
    f4 = lambda x,y,z,x4: -x + y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], -np.ones(4), np.ones(4))
    t = time() - start
    return residuals(f,g,h,f4,roots,t)

def dex12():
    df = lambda x, y, z, x4 : (2*x*y**2, 2*x**2*y, 0, 0)
    dg = lambda x, y, z, x4 : (y*(x - .1) + (x*y-.2), x*(x - .1), 0, 0)
    dh = lambda x, y, z, x4 : (2*x, 2*y, 2*z, 0)
    df4 = lambda x, y, z, x4 : (-1, 1, 1, 1)
    return df, dg, dh, df4

def ex13():
    f = lambda x,y,z,x4: (np.exp(y-z)**2-x**3)*((y-0.7)**2-(x-0.3)**3)*((np.exp(y-z)+0.2)**2-(x+0.8)**3)*((y+0.2)**2-(x-0.8)**3)
    g = lambda x,y,z,x4: ((np.exp(y-z)+.4)**3-(x-.4)**2)*((np.exp(y-z)+.3)**3-(x-.3)**2)*((np.exp(y-z)-.5)**3-(x+.6)**2)*((np.exp(y-z)+0.3)**3-(2*x-0.8)**3)
    h = lambda x,y,z,x4: x + y + z
    f4 = lambda x,y,z,x4: x + y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    t = time() - start
    return residuals(f,g,h,f4,roots,t)

def ex14():
    f = lambda x,y,z,x4: ((x*z-.3)**2+2*(np.log(y+1.2)+0.3)**2-1)
    g = lambda x,y,z,x4: ((x-.49)**2+(y+.5)**2-1)*((x+0.5)**2+(y+0.5)**2-1)*((x-1)**2+(y-0.5)**2-1)
    h = lambda x,y,z,x4: x**4 + (np.log(y+1.4)-.3) - z
    f4 = lambda x,y,z,x4:x - y + z - x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    t = time() - start
    return residuals(f,g,h,f4,roots,t)

def dex14():
    df = lambda x, y, z, x4 : (2*z*(x*z - .3), 4*(np.log(y + 1.2) + .3)/(y + 1.2), 2*x*(x*z - .3), 0)
    dg = lambda x, y, z, x4 : ((2*(x + .5)*((x - .49)**2 + (y + .5)**2 - 1) + 2*(x - .49)*((x + .5)**2 + (y + .5)**2 - 1))*((x - 1)**2 + (y + .5)**2 - 1) + 2*(x - 1)*(((x - .49)**2 + (y + .5)**2 - 1)*((x + .5)**2 + (y + .5)**2 - 1)),
                               (2*(y + .5)*((x - .49)**2 + (y + .5)**2 - 1) + 2*(y + .5)*((x + .5)**2 + (y + .5)**2 - 1))*((x - 1)**2 + (y + .5)**2 - 1) + 2*(y + .5)*(((x - .49)**2 + (y + .5)**2 - 1)*((x + .5)**2 + (y + .5)**2 - 1)),
                               0, 0)
    dh = lambda x, y, z, x4 : (4*x**3, 1/(y + 1.4), -1, 0)
    df4 = lambda x, y, z, x4 : (1, -1, 1, -1)
    return df, dg, dh, df4

def ex15():
    f = lambda x,y,z,x4: np.exp(x-2*x**2-y**2-z**2)*np.sin(10*(x+y+z+x*y**2))
    g = lambda x,y,z,x4: np.exp(-x+2*y**2+x*y**2*z)*np.sin(10*(x-y-2*x*y**2))
    h = lambda x,y,z,x4: np.exp(x**2*y**2)*np.cos(x-y+z)
    f4 = lambda x,y,z,x4: x - y + z - x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    t = time() - start
    return residuals(f,g,h,f4,roots,t)

def dex15():
    df = lambda x, y, z, x4 : (10*(1 + y**2)*np.exp(x - 2*x**2 - y**2 - z**2)*np.cos(10*(x + y + z + x*y**2)) + (1 - 4*x)*np.exp(x-2*x**2-y**2-z**2)*np.sin(10*(x+y+z+x*y**2)),
                            10*(1 + 2*x*y)*np.exp(x - 2*x**2 - y**2 - z**2)*np.cos(10*(x + y + z + x*y**2)) - 2*y*np.exp(x-2*x**2-y**2-z**2)*np.sin(10*(x+y+z+x*y**2)),
                            10*np.exp(x - 2*x**2 - y**2 - z**2)*np.cos(10*(x + y + z + x*y**2)) - 2*z*np.exp(x-2*x**2-y**2-z**2)*np.sin(10*(x+y+z+x*y**2)), 0)
    dg = lambda x, y, z, x4 : (10*(1 - 2*y**2)*np.exp(-x+2*y**2+x*y**2*z)*np.cos(10*(x-y-2*x*y**2)) + (-2*x*y**2 + y**2*z)*np.exp(-x+2*y**2+x*y**2*z)*np.sin(10*(x-y-2*x*y**2)),
                            10*(-1-4*x*y)*p.exp(-x+2*y**2+x*y**2*z)*np.cos(10*(x-y-2*x*y**2)) + (-2*x**2*y + 2*x*y*z)*np.exp(-x+2*y**2+x*y**2*z)*np.sin(10*(x-y-2*x*y**2)),
                            x*y**2*np.exp(-x+2*y**2+x*y**2*z)*np.sin(10*(x-y-2*x*y**2)), 0)
    dh = lambda x, y, z, x4 : (-np.exp(x**2*y**2)*np.sin(x-y+z)+2*x*y**2*np.exp(x**2*y**2)*np.cos(x-y+z),
                            np.exp(x**2*y**2)*np.sin(x-y+z)+2*x**2*y*np.exp(x**2*y**2)*np.cos(x-y+z),
                            -np.exp(x**2*y**2)*np.sin(x-y+z), 0)
    df4 = lambda x, y, z, x4 : (1, -1, 1, -1)
    return df, dg, dh, df4

def ex16():
    f = lambda x,y,z,x4: ((x-0.1)**2+2*(y*z-0.1)**2-1)*((x*y+0.3)**2+2*(z-0.2)**2-1)
    g = lambda x,y,z,x4: (2*(x*z+0.1)**2+(y+0.1)**2-1)*(2*(z-0.3)**2+(y-0.15)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1)
    h = lambda x,y,z,x4: (2*(y+0.1)**2-(z+.15)**2-1)*(2*(x+0.3)**2+(z-.15)**2-1)
    f4 = lambda x,y,z,x4: x - y + z - x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    t = time() - start
    return residuals(f,g,h,f4,roots,t)

def dex16():
    df = lambda x, y, z, x4 : (2*y*(x*y+.3)*((x - .1)**2 + 2*(y*z-.1)**2-1) + 2*(x-.01)*((x*y+.3)**2+2*(z-.2)**2-1),
                                2*x*(x*y+.3)*((x - .1)**2 + 2*(y*z-.1)**2-1) + 2*z*(y*z-.1)*((x*y+.3)**2+2*(z-.2)**2-1),
                                2*(z-.2)*((x - .1)**2 + 2*(y*z-.1)**2-1) + 2*y*(y*z-.1)*((x*y+.3)**2+2*(z-.2)**2-1), 0)
    dg = lambda x, y, z, x4 : (2*(x-.21)*(2*(x*z+0.1)**2+(y+0.1)**2-1)*(2*(z-0.3)**2+(y-0.15)**2-1) + 4*z*(x*z+.1)*(2*(z-0.3)**2+(y-0.15)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1),
                                4*(y-.15)*(2*(x*z+0.1)**2+(y+0.1)**2-1)*(2*(z-0.3)**2+(y-0.15)**2-1) + ((x-0.21)**2+2*(y-0.15)**2-1)*(2*(y+.1)*(2*(z-0.3)**2+(y-0.15)**2-1)+2*(y-.15)*(2*(x*z+0.1)**2+(y+0.1)**2-1)),
                                4*(z-.3)*(2*(x*z+0.1)**2+(y+0.1)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1) + 4*x*(x*z + .1)*(2*(z-0.3)**2+(y-0.15)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1), 0)
    dh = lambda x, y, z, x4 : (4*(x + .3)*(2*(y+0.1)**2-(z+.15)**2-1),
                                4*(y+.1)*(2*(x+0.3)**2+(z-.15)**2-1),
                                2*(z-.15)*(2*(y+0.1)**2-(z+.15)**2-1) + 2*(z+.15)*(2*(x+0.3)**2+(z-.15)**2-1), 0)
    df4 = lambda x, y, z, x4 : (1, -1, 1, -1)
    return df, dg, dh, df4

def ex17():
    f = lambda x,y,z,x4: np.sin(3*(x+y+z))
    g = lambda x,y,z,x4: np.sin(3*(x+y-z))
    h = lambda x,y,z,x4: np.sin(3*(x-y-z))
    f4 = lambda x,y,z,x4: -x + y - z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    t = time() - start
    return residuals(f,g,h,f4,roots,t)

def dex17():
    df = lambda x, y, z, x4 : (3*np.cos(3*(x+y+z)), 3*np.cos(3*(x+y+z)), 3*np.cos(3*(x+y+z)), 0)
    dg = lambda x, y, z, x4 : (3*np.cos(3*(x+y-z)), 3*np.cos(3*(x+y-z)), -3*np.cos(3*(x+y-z)), 0)
    dh = lambda x, y, z, x4 : (3*np.cos(3*(x-y-z)), -3*np.cos(3*(x-y-z)), -3*np.cos(3*(x-y-z)), 0)
    df4 = lambda x, y, z, x4 : (-1, 1, -1, 1)
    return df, dg, dh, df4

def ex18():
    f = lambda x,y,z,x4: x - 2 + 3*sp.special.erf(z)
    g = lambda x,y,z,x4: np.sin(x*z)
    h = lambda x,y,z,x4: x*y + y**2 - 1
    f4 = lambda x,y,z,x4: x + y - z + x4

    a=[-1,-1,-1,-1]
    b=[1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4],a,b)
    t = time() - start
    return residuals(f,g,h,f4,roots,t)

def dex18():
    df = lambda x, y, z, x4 : (1, 0, 6/np.sqrt(np.pi)*np.exp(-z**2), 0)
    dg = lambda x, y, z, x4 : (z*np.cos(x*z), 0, x*np.cos(x*z), 0)
    dh = lambda x, y, z, x4 : (y, x + 2*y, 0, 0)
    df4 = lambda x, y, z, x4 : (1, 1, -1, 1)
    return df, dg, dh, df4

if __name__ == "__main__":
    # max_residuals = [ex1(),ex2(),ex3(),ex4(),ex5(),ex6(),ex7(),ex8(),ex9(),ex10(),ex11(),ex12(),ex13(),ex14(),ex15(),ex16(),ex17(),ex18()]
    # plot_resids(max_residuals)
    ex0()
