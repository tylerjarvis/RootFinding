from numalgsolve import polynomial
from numalgsolve import polyroots
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from math import sqrt

#bools for which tests to show

hyperbolas11 = True
hyperbolas_transformed  = False
hyperbolas_moved = False
hyperbolas_transformed_moved = False
circle_ellipse_origin = False
circle_ellipse_moved = False
circle_ellipse_transformed = False
circle_ellipse_transformed_moved = True

if hyperbolas11:
    print("Double Root Trials: \n p1 = x^2 + 2xy + y^2 - 3x - 5y + 4 \n p2 = - x^2 - 2xy - y^2 + 5x + 3y - 4")
    print("Two hyperbolas that intersect at (1,1). Double root there, probably roots at infinity??")

    p1_coef = np.array([[4, -5, 1],[-3,2,0],[1,0,0]]) #p1 = x^2 + 2xy + y^2 - 3x - 5y + 4
    p2_coef = np.array([[-4, 3, -1],[5,-2,0],[-1,0,0]]) #p2 = - x^2 - 2xy - y^2 + 5x + 3y - 4

    p1 = polynomial.MultiPower(p1_coef)
    p2 = polynomial.MultiPower(p2_coef)
    p1_switch_xy = polynomial.MultiPower(p1_coef.T)
    p2_switch_xy = polynomial.MultiPower(p2_coef.T)

    #In Cheb form

    c1_coef = np.array([[5,-5,.5],[-3,2,0],[.5,0,0]])
    c2_coef = np.array([[-5,3,-.5],[5,-2,0],[-.5,0,0]])

    c1 = polynomial.MultiCheb(c1_coef) # 1/2 T2x + 2 xy + 1/2 T2y - 5 y - 3 x + 5 == 0
    c2 = polynomial.MultiCheb(c2_coef) #- 1/2 T2x  -2 xy - 1/2 T2y + 3 y  + 5 x - 5 == 0
    c1_switch_xy = polynomial.MultiCheb(c1_coef.T)
    c2_switch_xy = polynomial.MultiCheb(c2_coef.T)

    #print("~ ~ ~ Power Form, M_f Matrix ~ ~ ~")
    #pmf = polyroots.solve([p1, p2])
    #print("Roots:\n",pmf)

    #print("~ ~ ~ Cheb Form, M_f Matrix ~ ~ ~")
    #cmf = polyroots.solve([c1, c2])
    #print("Roots:\n",cmf)

    #print("~ ~ ~ Power Form, M_x Matrix ~ ~ ~")
    #pmx = polyroots.solve([p1, p2], rand_poly = False)
    #print("Roots:\n",pmx)

    #print("~ ~ ~ Cheb Form, M_x Matrix ~ ~ ~")
    #cmx = polyroots.solve([c1, c2], rand_poly = False)
    #print("Roots:\n",cmx)

    #flip left/right because x and y are switched. Same for M_y and M_1/y matrices below
    #print("~ ~ ~ Power Form, M_y Matrix ~ ~ ~")
    #pmy = polyroots.solve([p1_switch_xy, p2_switch_xy], rand_poly = False)[:,::-1]
    #print("Roots:\n",pmy)

    #print("~ ~ ~ Cheb Form, M_y Matrix ~ ~ ~")
    #cmy = polyroots.solve([c1_switch_xy, c2_switch_xy], rand_poly = False)[:,::-1]
    #print("Roots:\n",cmy)

    print("~ ~ ~ Power Form, Division Matrix 1/y ~ ~ ~")
    pdy = polyroots.solve([p1_switch_xy, p2_switch_xy], method = "div")[:,::-1]
    print("Roots:\n",pdy)

    print("~ ~ ~ Cheb Form, Division Matrix 1/y ~ ~ ~")
    cdy = polyroots.solve([c1_switch_xy, c2_switch_xy], method = "div")[:,::-1]
    print("Roots:\n",cdy)

    print("~ ~ ~ Power Form, Division Matrix 1/x ~ ~ ~")
    pdx = polyroots.solve([p1, p2], method = "div")
    print("Roots:\n",pdx)

    print("~ ~ ~ Cheb Form, Division Matrix 1/x ~ ~ ~")
    cdx = polyroots.solve([c1, c2], method = "div")
    print("Roots:\n",cdx)

    print("\n\nCompare Roots:\n")
    #print("Power M_f\n", pmf)#along right line, inconsistent
    #print("Cheb M_f\n", cmf)#along right line, inconsistent
    #print("Power M_x\n", pmx)#along right line, weird
    #print("Cheb M_x\n", cmx)#along right line, weird
    #print("Power M_y\n", pmy)#along right line, weird
    #print("Cheb M_y\n", cmy)#along right line, weird
    print("Power M_1/y\n", pdy)#Good
    print("Cheb M_1/y\n", cdy)#Weird
    print("Power M_1/x\n", pdx)#Good
    print("Cheb M_1/x\n", cdx)#Weird

    #graph the polys
    delta = 0.01
    xrange = np.arange(-1, 2, delta)
    yrange = np.arange(-1, 2, delta)
    X, Y = np.meshgrid(xrange,yrange)

    ax1 = plt.subplot(121)
    ax1.set_title("No M_x or M_y Matrices")
    #polys
    ax1.contour(X, Y, X**2 + 2*X*Y + Y**2 - 3*X - 5*Y + 4, [0], colors="black", linestyles="dashed")
    ax1.contour(X, Y, -X**2 - 2*X*Y - Y**2 + 5*X + 3*Y - 4, [0], colors="black", linestyles="dashed")
    #roots
    #ax1.plot(pmf[:,0],pmf[:,1],".r",markersize=4)
    #ax1.plot(cmf[:,0],cmf[:,1], ".m",markersize=4)
    ax1.plot(pdy[:,0],pdy[:,1], "xb",markersize=6)
    ax1.plot(cdy[:,0],cdy[:,1], ".c",markersize=4)
    ax1.plot(pdx[:,0],pdx[:,1], "+", color = "gold",markersize=8)
    ax1.plot(cdx[:,0],cdx[:,1], ".",color ="orange",markersize=4)
    ax1.axis("equal")

    ax2 = plt.subplot(122)
    ax2.set_title("With M_x and M_y Matrices")
    #polys
    ax2.contour(X, Y, X**2 + 2*X*Y + Y**2 - 3*X - 5*Y + 4, [0], colors="black", linestyles="dashed")
    ax2.contour(X, Y, -X**2 - 2*X*Y - Y**2 + 5*X + 3*Y - 4, [0], colors="black", linestyles="dashed")
    #roots
    #ax2.plot(pmx[:,0],pmx[:,1],".", color="gray", label="Power M_x",markersize=4)
    ##ax2.plot(cmx[:,0],cmx[:,1], ".", color="silver", label="Cheb M_x",markersize=4)
    #ax2.plot(pmy[:,0],pmy[:,1],".", color="tan", label="Power M_y",markersize=4)
    ##ax2.plot(cmy[:,0],cmy[:,1], ".", color="peru", label="Cheb M_y",markersize=4)
    #ax2.plot(pmf[:,0],pmf[:,1],".r", label="Power M_f",markersize=4)
    #ax2.plot(cmf[:,0],cmf[:,1], ".m", label="Cheb M_f",markersize=4)
    ax2.plot(pdy[:,0],pdy[:,1], "xb", label="Power M_1/y",markersize=6)
    ax2.plot(cdy[:,0],cdy[:,1], ".c", label="Cheb M_1/y",markersize=4)
    ax2.plot(pdx[:,0],pdx[:,1], "+", color = "gold", label="Power M_1/x",markersize=8)
    ax2.plot(cdx[:,0],cdx[:,1], ".",color ="orange", label="Cheb M_1/x",markersize=4)
    ax2.axis("equal")

    plt.suptitle("Two hyperbolas that intersect at (1,1)")
    plt.figlegend(loc="lower left")
    plt.show()

if hyperbolas_transformed:
    print("Double Root Trials: \n p1 = 4.0*x^2 + (8.0*x - 8.2)*y + 4.0*y^2 - 7.6*x + 4 \n p2 = -4.0*x^2 + (-8.0*x + 7.8)*y - 4.0*y^2 + 8.4*x - 4")
    print("Same two hyperbolas. Transformed a little more, now intersect at (1/3,2/3). Double root there, probably roots at infinity??")

    p1_coef = np.array([[4, -8.2, 4],[-7.6, 8,0],[4,0,0]]) #p1 = x^2 + 2xy + y^2 - 3x - 5y + 4
    p2_coef = np.array([[-4, 7.8, -4],[8.4,-8,0],[-4,0,0]]) #p2 = - x^2 - 2xy - y^2 + 5x + 3y - 4

    p1 = polynomial.MultiPower(p1_coef)
    p2 = polynomial.MultiPower(p2_coef)
    p1_switch_xy = polynomial.MultiPower(p1_coef.T)
    p2_switch_xy = polynomial.MultiPower(p2_coef.T)

    #In Cheb form

    c1_coef = np.array([[8,-8.2,2],[-7.6,8,0],[2,0,0]])
    c2_coef = np.array([[-8,7.8,-2],[8.4,-8,0],[-2,0,0]])

    c1 = polynomial.MultiCheb(c1_coef) # 1/2 T2x + 2 xy + 1/2 T2y - 5 y - 3 x + 5 == 0
    c2 = polynomial.MultiCheb(c2_coef) #- 1/2 T2x  -2 xy - 1/2 T2y + 3 y  + 5 x - 5 == 0
    c1_switch_xy = polynomial.MultiCheb(c1_coef.T)
    c2_switch_xy = polynomial.MultiCheb(c2_coef.T)

    print("~ ~ ~ Power Form, M_f Matrix ~ ~ ~")
    #pmf = polyroots.solve([p1, p2])
    #print("Roots:\n",pmf)

    print("~ ~ ~ Cheb Form, M_f Matrix ~ ~ ~")
    #cmf = polyroots.solve([c1, c2])
    #print("Roots:\n",cmf)

    print("~ ~ ~ Power Form, M_x Matrix ~ ~ ~")
    #pmx = polyroots.solve([p1, p2], rand_poly = False)
    #print("Roots:\n",pmx)

    print("~ ~ ~ Cheb Form, M_x Matrix ~ ~ ~")
    #cmx = polyroots.solve([c1, c2], rand_poly = False)
    #print("Roots:\n",cmx)

    #flip left/right because x and y are switched. Same for M_y and M_1/y matrices below
    print("~ ~ ~ Power Form, M_y Matrix ~ ~ ~")
    #pmy = polyroots.solve([p1_switch_xy, p2_switch_xy], rand_poly = False)[:,::-1]
    #print("Roots:\n",pmy)

    print("~ ~ ~ Cheb Form, M_y Matrix ~ ~ ~")
    #cmy = polyroots.solve([c1_switch_xy, c2_switch_xy], rand_poly = False)[:,::-1]
    #print("Roots:\n",cmy)

    print("~ ~ ~ Power Form, Division Matrix 1/y ~ ~ ~")
    pdy = polyroots.solve([p1_switch_xy, p2_switch_xy], method = "div")[:,::-1]
    print("Roots:\n",pdy)

    print("~ ~ ~ Cheb Form, Division Matrix 1/y ~ ~ ~")
    cdy = polyroots.solve([c1_switch_xy, c2_switch_xy], method = "div")[:,::-1]
    print("Roots:\n",cdy)

    print("~ ~ ~ Power Form, Division Matrix 1/x ~ ~ ~")
    pdx = polyroots.solve([p1, p2], method = "div")
    print("Roots:\n",pdx)

    print("~ ~ ~ Cheb Form, Division Matrix 1/x ~ ~ ~")
    cdx = polyroots.solve([c1, c2], method = "div")
    print("Roots:\n",cdx)

    print("\n\nCompare Roots:\n")
    #print("Power M_f\n", pmf)#along right line
    #print("Cheb M_f\n", cmf)#along right line
    #print("Power M_x\n", pmx)#along right line
    #print("Cheb M_x\n", cmx)#along right line
    #print("Power M_y\n", pmy)#along right line
    #print("Cheb M_y\n", cmy)#along right line
    print("Power M_1/y\n", pdy)#Good
    print("Cheb M_1/y\n", cdy)#Weird
    print("Power M_1/x\n", pdx)#Good
    print("Cheb M_1/x\n", cdx)#Weird

    #graph the polys
    delta = 0.01
    xrange = np.arange(-1, 2, delta)
    yrange = np.arange(-1, 2, delta)
    X, Y = np.meshgrid(xrange,yrange)

    plt.clf()
    ax1 = plt.subplot(121)
    ax1.set_title("No M_x or M_y Matrices")
    #polys
    ax1.contour(X, Y, 4.0*X**2 + (8.0*X - 8.2)*Y + 4.0*Y**2 - 7.6*X + 4, [0], colors="black", linestyles="dashed")
    ax1.contour(X, Y, -4.0*X**2 + (-8.0*X + 7.8)*Y - 4.0*Y**2 + 8.4*X - 4, [0], colors="black", linestyles="dashed")
    #roots
    #ax1.plot(pmf[:,0],pmf[:,1],".r",markersize=4)
    #ax1.plot(cmf[:,0],cmf[:,1], ".m",markersize=4)
    ax1.plot(pdy[:,0],pdy[:,1], "xb",markersize=6)
    ax1.plot(cdy[:,0],cdy[:,1], ".c",markersize=4)
    ax1.plot(pdx[:,0],pdx[:,1], "+", color = "gold",markersize=8)
    ax1.plot(cdx[:,0],cdx[:,1], ".",color ="orange",markersize=4)
    ax1.axis("equal")

    ax2 = plt.subplot(122)
    ax2.set_title("With M_x and M_y Matrices")
    #polys
    ax2.contour(X, Y, 4.0*X**2 + (8.0*X - 8.2)*Y + 4.0*Y**2 - 7.6*X + 4, [0], colors="black", linestyles="dashed")
    ax2.contour(X, Y, -4.0*X**2 + (-8.0*X + 7.8)*Y - 4.0*Y**2 + 8.4*X - 4, [0], colors="black", linestyles="dashed")
    #roots
    #ax2.plot(pmx[:,0],pmx[:,1],".", color="gray", label="Power M_x",markersize=4)
    #ax2.plot(cmx[:,0],cmx[:,1], ".", color="silver", label="Cheb M_x",markersize=4)
    #ax2.plot(pmy[:,0],pmy[:,1],".", color="tan", label="Power M_y",markersize=4)
    #ax2.plot(cmy[:,0],cmy[:,1], ".", color="peru", label="Cheb M_y",markersize=4)
    #ax2.plot(pmf[:,0],pmf[:,1],".r", label="Power M_f",markersize=4)
    #ax2.plot(cmf[:,0],cmf[:,1], ".m", label="Cheb M_f",markersize=4)
    ax2.plot(pdy[:,0],pdy[:,1], "xb", label="Power M_1/y",markersize=6)
    ax2.plot(cdy[:,0],cdy[:,1], ".c", label="Cheb M_1/y",markersize=4)
    ax2.plot(pdx[:,0],pdx[:,1], "+", color = "gold", label="Power M_1/x",markersize=8)
    ax2.plot(cdx[:,0],cdx[:,1], ".",color ="orange", label="Cheb M_1/x",markersize=4)
    ax2.axis("equal")

    plt.suptitle("Transformed a little more, now intersect at (1/3,2/3)")
    plt.figlegend(loc="lower left")
    plt.show()

if hyperbolas_moved:
    print("Double Root Trials: \n p1 = x^2 + (2x - 2sqrt(2) - 2.8)*y + y^2 + x*(-2sqrt(2) - 0.8) + 2.8*sqrt(2) + 1.91 \n p2 = -x^2 + (-2x + 2sqrt(2) + 0.8)y - y^2 + x*(2sqrt(2) + 2.8) - 0.8sqrt(2) - 3.71")
    print("Hyperbolas, moved a little bit, now intersect at (.9, sqrt(2)). Double root there, probably roots at infinity??")

    p1_coef = np.array([[1.91 + 2.8*sqrt(2), -(2*sqrt(2) + 2.8), 1],[-(2*sqrt(2)+.8), 2, 0],[1,0,0]])
    p2_coef = np.array([[-3.71 - .8*sqrt(2), 2*sqrt(2) + .8, -1],[2*sqrt(2)+2.8, -2, 0],[-1,0,0]])

    p1 = polynomial.MultiPower(p1_coef)
    p2 = polynomial.MultiPower(p2_coef)
    p1_switch_xy = polynomial.MultiPower(p1_coef.T)
    p2_switch_xy = polynomial.MultiPower(p2_coef.T)

    #In Cheb form
    c1_coef = np.array([[2.91 + 2.8*sqrt(2), -(2*sqrt(2) + 2.8), .5],[-(2*sqrt(2)+.8), 2, 0],[.5,0,0]])
    c2_coef = np.array([[-4.71 - .8*sqrt(2), 2*sqrt(2) + .8, -.5],[2*sqrt(2)+2.8, -2, 0],[-.5,0,0]])

    c1 = polynomial.MultiCheb(c1_coef) # 1/2 T2x + 2 xy + 1/2 T2y - 5 y - 3 x + 5 == 0
    c2 = polynomial.MultiCheb(c2_coef) #- 1/2 T2x  -2 xy - 1/2 T2y + 3 y  + 5 x - 5 == 0
    c1_switch_xy = polynomial.MultiCheb(c1_coef.T)
    c2_switch_xy = polynomial.MultiCheb(c2_coef.T)

    #print("~ ~ ~ Power Form, M_f Matrix ~ ~ ~")
    #pmf = polyroots.solve([p1, p2])
    #print("Roots:\n",pmf)

    #print("~ ~ ~ Cheb Form, M_f Matrix ~ ~ ~")
    #cmf = polyroots.solve([c1, c2])
    #print("Roots:\n",cmf)

    #print("~ ~ ~ Power Form, M_x Matrix ~ ~ ~")
    #pmx = polyroots.solve([p1, p2], rand_poly = False)
    #print("Roots:\n",pmx)

    #print("~ ~ ~ Cheb Form, M_x Matrix ~ ~ ~")
    #cmx = polyroots.solve([c1, c2], rand_poly = False)
    #print("Roots:\n",cmx)

    #flip left/right because x and y are switched. Same for M_y and M_1/y matrices below
    #print("~ ~ ~ Power Form, M_y Matrix ~ ~ ~")
    #pmy = polyroots.solve([p1_switch_xy, p2_switch_xy], rand_poly = False)[:,::-1]
    #print("Roots:\n",pmy)

    #print("~ ~ ~ Cheb Form, M_y Matrix ~ ~ ~")
    #cmy = polyroots.solve([c1_switch_xy, c2_switch_xy], rand_poly = False)[:,::-1]
    #print("Roots:\n",cmy)

    print("~ ~ ~ Power Form, Division Matrix 1/y ~ ~ ~")
    pdy = polyroots.solve([p1_switch_xy, p2_switch_xy], method = "div")[:,::-1]
    print("Roots:\n",pdy)

    print("~ ~ ~ Cheb Form, Division Matrix 1/y ~ ~ ~")
    cdy = polyroots.solve([c1_switch_xy, c2_switch_xy], method = "div")[:,::-1]
    print("Roots:\n",cdy)

    print("~ ~ ~ Power Form, Division Matrix 1/x ~ ~ ~")
    pdx = polyroots.solve([p1, p2], method = "div")
    print("Roots:\n",pdx)

    print("~ ~ ~ Cheb Form, Division Matrix 1/x ~ ~ ~")
    cdx = polyroots.solve([c1, c2], method = "div")
    print("Roots:\n",cdx)

    print("\n\nCompare Roots:\n")
    #print("Power M_f\n", pmf)#along right line
    #print("Cheb M_f\n", cmf)#along right line
    #print("Power M_x\n", pmx)#along right line
    #print("Cheb M_x\n", cmx)#along right line
    #print("Power M_y\n", pmy)#along right line
    #print("Cheb M_y\n", cmy)#along right line
    print("Power M_1/y\n", pdy)#Good
    print("Cheb M_1/y\n", cdy)#Weird
    print("Power M_1/x\n", pdx)#Good
    print("Cheb M_1/x\n", cdx)#Weird

    #graph the polys
    delta = 0.01
    xrange = np.arange(-1, 2, delta)
    yrange = np.arange(-1, 2, delta)
    X, Y = np.meshgrid(xrange,yrange)

    plt.clf()
    ax1 = plt.subplot(121)
    ax1.set_title("No M_x or M_y Matrices")
    #polys
    ax1.contour(X, Y, X**2 + (2*X - 2*sqrt(2) - 2.8)*Y + Y**2 + X*(-2*sqrt(2) - 0.8) + 2.8*sqrt(2) + 1.91, [0], colors="black", linestyles="dashed")
    ax1.contour(X, Y, -X**2 + (-2*X + 2*sqrt(2) + 0.8)*Y - Y**2 + X*(2*sqrt(2) + 2.8) - 0.8*sqrt(2) - 3.71, [0], colors="black", linestyles="dashed")
    #roots
    #ax1.plot(pmf[:,0],pmf[:,1],".r",markersize=4)
    #ax1.plot(cmf[:,0],cmf[:,1], ".m",markersize=4)
    ax1.plot(pdy[:,0],pdy[:,1], "xb",markersize=6)
    ax1.plot(cdy[:,0],cdy[:,1], ".c",markersize=4)
    ax1.plot(pdx[:,0],pdx[:,1], "+", color = "gold",markersize=8)
    ax1.plot(cdx[:,0],cdx[:,1], ".",color ="orange",markersize=4)
    ax1.axis("equal")

    ax2 = plt.subplot(122)
    ax2.set_title("With M_x and M_y Matrices")
    #polys
    ax2.contour(X, Y, 4.0*X**2 + (8.0*X - 8.2)*Y + 4.0*Y**2 - 7.6*X + 4, [0], colors="black", linestyles="dashed")
    ax2.contour(X, Y, -4.0*X**2 + (-8.0*X + 7.8)*Y - 4.0*Y**2 + 8.4*X - 4, [0], colors="black", linestyles="dashed")
    #roots
    #ax2.plot(pmx[:,0],pmx[:,1],".", color="gray", label="Power M_x",markersize=4)
    #ax2.plot(cmx[:,0],cmx[:,1], ".", color="silver", label="Cheb M_x",markersize=4)
    #ax2.plot(pmy[:,0],pmy[:,1],".", color="tan", label="Power M_y",markersize=4)
    #ax2.plot(cmy[:,0],cmy[:,1], ".", color="peru", label="Cheb M_y",markersize=4)
    #ax2.plot(pmf[:,0],pmf[:,1],".r", label="Power M_f",markersize=4)
    #ax2.plot(cmf[:,0],cmf[:,1], ".m", label="Cheb M_f",markersize=4)
    ax2.plot(pdy[:,0],pdy[:,1], "xb", label="Power M_1/y",markersize=6)
    ax2.plot(cdy[:,0],cdy[:,1], ".c", label="Cheb M_1/y",markersize=4)
    ax2.plot(pdx[:,0],pdx[:,1], "+", color = "gold", label="Power M_1/x",markersize=8)
    ax2.plot(cdx[:,0],cdx[:,1], ".",color ="orange", label="Cheb M_1/x",markersize=4)
    ax2.axis("equal")

    plt.suptitle("Hyperbolas, moved a little bit, now intersect at (.9, sqrt(2))")
    plt.figlegend(loc="lower left")
    plt.show()

if hyperbolas_transformed_moved:
    print("Double Root Trials: \n p1 = 4x^2 + 8 xy - (4sqrt(2) + 3.8) y + 4 y^2 + (-4sqrt(2) - 3.2)x + 2.8*sqrt(2) + 1.91 \n p2 = -4 x^2 + -8 xy + (4sqrt(2) + 3.4) y - 4.0 y^2 + 4(sqrt(2) + 1) x - 0.8 sqrt(2) - 3.71")
    print("Hyperbolas, Transformed and moved a little bit, now intersect at (-3/2*sqrt(2) + 33/20, 2*sqrt(2) - 6/5). Double root there, probably roots at infinity??")

    p1_coef = np.array([[2.8*sqrt(2) + 1.91, - (4*sqrt(2) + 3.8),4],[-4*sqrt(2) - 3.2,8,0],[4,0,0]])
    p2_coef = np.array([[-0.8*sqrt(2) - 3.71, 4*sqrt(2) + 3.4,-4],[4*sqrt(2) + 4,-8,0],[-4,0,0]])

    p1 = polynomial.MultiPower(p1_coef)
    p2 = polynomial.MultiPower(p2_coef)
    p1_switch_xy = polynomial.MultiPower(p1_coef.T)
    p2_switch_xy = polynomial.MultiPower(p2_coef.T)

    #In Cheb form
    c1_coef = np.array([[2.8*sqrt(2) + 5.91,- 4.0*sqrt(2) - 3.8,2],[-4*sqrt(2) - 3.2,8,0],[2,0,0]])
    c2_coef = np.array([[- 0.8*sqrt(2) - 7.71,4.0*sqrt(2) + 3.4,-2],[4*sqrt(2) + 4,-8,0],[-2,0,0]])

    c1 = polynomial.MultiCheb(c1_coef)
    c2 = polynomial.MultiCheb(c2_coef)
    c1_switch_xy = polynomial.MultiCheb(c1_coef.T)
    c2_switch_xy = polynomial.MultiCheb(c2_coef.T)

    print("~ ~ ~ Power Form, M_f Matrix ~ ~ ~")
    pmf = polyroots.solve([p1, p2])
    print("Roots:\n",pmf)

    print("~ ~ ~ Cheb Form, M_f Matrix ~ ~ ~")
    cmf = polyroots.solve([c1, c2])
    print("Roots:\n",cmf)

    print("~ ~ ~ Power Form, M_x Matrix ~ ~ ~")
    pmx = polyroots.solve([p1, p2], rand_poly = False)
    print("Roots:\n",pmx)

    print("~ ~ ~ Cheb Form, M_x Matrix ~ ~ ~")
    cmx = polyroots.solve([c1, c2], rand_poly = False)
    print("Roots:\n",cmx)

    #flip left/right because x and y are switched. Same for M_y and M_1/y matrices below
    print("~ ~ ~ Power Form, M_y Matrix ~ ~ ~")
    pmy = polyroots.solve([p1_switch_xy, p2_switch_xy], rand_poly = False)[:,::-1]
    print("Roots:\n",pmy)

    print("~ ~ ~ Cheb Form, M_y Matrix ~ ~ ~")
    cmy = polyroots.solve([c1_switch_xy, c2_switch_xy], rand_poly = False)[:,::-1]
    print("Roots:\n",cmy)

    print("~ ~ ~ Power Form, Division Matrix 1/y ~ ~ ~")
    pdy = polyroots.solve([p1_switch_xy, p2_switch_xy], method = "div")[:,::-1]
    print("Roots:\n",pdy)

    print("~ ~ ~ Cheb Form, Division Matrix 1/y ~ ~ ~")
    cdy = polyroots.solve([c1_switch_xy, c2_switch_xy], method = "div")[:,::-1]
    print("Roots:\n",cdy)

    print("~ ~ ~ Power Form, Division Matrix 1/x ~ ~ ~")
    pdx = polyroots.solve([p1, p2], method = "div")
    print("Roots:\n",pdx)

    print("~ ~ ~ Cheb Form, Division Matrix 1/x ~ ~ ~")
    cdx = polyroots.solve([c1, c2], method = "div")
    print("Roots:\n",cdx)

    print("\n\nCompare Roots:\n")
    print("Actual Roots: ", [-3/2*sqrt(2) + 33/20, 2*sqrt(2) - 6/5])
    #print("Power M_f\n", pmf)#along right line
    #print("Cheb M_f\n", cmf)#along right line
    #print("Power M_x\n", pmx)#along right line
    #print("Cheb M_x\n", cmx)#along right line
    #print("Power M_y\n", pmy)#along right line
    #print("Cheb M_y\n", cmy)#along right line
    print("Power M_1/y\n", pdy)#Good
    print("Cheb M_1/y\n", cdy)#Weird
    print("Power M_1/x\n", pdx)#Good
    print("Cheb M_1/x\n", cdx)#Weird

    #graph the polys
    delta = 0.01
    xrange = np.arange(-1, 2, delta)
    yrange = np.arange(-1, 2, delta)
    X, Y = np.meshgrid(xrange,yrange)

    plt.clf()
    ax2 = plt.subplot(111)
    #polys
    ax2.contour(X, Y, 4*X**2 + 8*X*Y - (4*sqrt(2) + 3.8)*Y + 4*Y**2 + (-4*sqrt(2) - 3.2)*X + 2.8*sqrt(2) + 1.91, [0], colors="black", linestyles="dashed")
    ax2.contour(X, Y, -4*X**2 + -8*X*Y + (4*sqrt(2) + 3.4)*Y - 4.0*Y**2 + 4*(sqrt(2) + 1)*X - 0.8*sqrt(2) - 3.71, [0], colors="black", linestyles="dashed")
    #roots
    #ax2.plot(pmx[:,0],pmx[:,1],".", color="gray", label="Power M_x",markersize=4)
    #ax2.plot(cmx[:,0],cmx[:,1], ".", color="silver", label="Cheb M_x",markersize=4)
    #ax2.plot(pmy[:,0],pmy[:,1],".", color="tan", label="Power M_y",markersize=4)
    #ax2.plot(cmy[:,0],cmy[:,1], ".", color="peru", label="Cheb M_y",markersize=4)
    #ax2.plot(pmf[:,0],pmf[:,1],".r", label="Power M_f",markersize=4)
    #ax2.plot(cmf[:,0],cmf[:,1], ".m", label="Cheb M_f",markersize=4)
    ax2.plot(pdy[:,0],pdy[:,1], "xb", label="Power M_1/y",markersize=6)
    ax2.plot(cdy[:,0],cdy[:,1], ".c", label="Cheb M_1/y",markersize=4)
    ax2.plot(pdx[:,0],pdx[:,1], "+", color = "gold", label="Power M_1/x",markersize=8)
    ax2.plot(cdx[:,0],cdx[:,1], ".",color ="orange", label="Cheb M_1/x",markersize=4)
    ax2.axis("equal")

    plt.suptitle("Hyperbolas, Transformed and moved a little bit, now intersect at (-3/2*sqrt(2) + 33/20, 2*sqrt(2) - 6/5)")
    plt.figlegend(loc="lower left")
    plt.show()

if circle_ellipse_origin:
    print("Double Root Trials: \n p1 = x^2 + y^2 -1 \n p2 = 4x^2 + y^2 - 1 ")
    print("Circle and inscribed ellipse centered at the origin. Roots at (0,1)M2 and (0,-1)M2.")

    p1_coef = np.array([[-1,0,1],[0,0,0],[1,0,0]])
    p2_coef = np.array([[-1,0,1],[0,0,0],[4,0,0]])

    p1 = polynomial.MultiPower(p1_coef)
    p2 = polynomial.MultiPower(p2_coef)
    p1_switch_xy = polynomial.MultiPower(p1_coef.T)
    p2_switch_xy = polynomial.MultiPower(p2_coef.T)

    #In Cheb form

    c1_coef = np.array([[0,0,.5],[0,0,0],[.5,0,0]])
    c2_coef = np.array([[1.5, 0, .5],[0,0,0],[2,0,0]])

    c1 = polynomial.MultiCheb(c1_coef)
    c2 = polynomial.MultiCheb(c2_coef)
    c1_switch_xy = polynomial.MultiCheb(c1_coef.T)
    c2_switch_xy = polynomial.MultiCheb(c2_coef.T)

    print("~ ~ ~ Power Form, M_f Matrix ~ ~ ~")
    pmf = polyroots.solve([p1, p2])
    print("Roots:\n",pmf)

    print("~ ~ ~ Cheb Form, M_f Matrix ~ ~ ~")
    cmf = polyroots.solve([c1, c2])
    print("Roots:\n",cmf)

    print("~ ~ ~ Power Form, M_x Matrix ~ ~ ~")
    pmx = polyroots.solve([p1, p2], rand_poly = False)
    print("Roots:\n",pmx)

    print("~ ~ ~ Cheb Form, M_x Matrix ~ ~ ~")
    cmx = polyroots.solve([c1, c2], rand_poly = False)
    print("Roots:\n",cmx)

    #flip left/right because x and y are switched. Same for M_y and M_1/y matrices below
    print("~ ~ ~ Power Form, M_y Matrix ~ ~ ~")
    pmy = polyroots.solve([p1_switch_xy, p2_switch_xy], rand_poly = False)[:,::-1]
    print("Roots:\n",pmy)

    print("~ ~ ~ Cheb Form, M_y Matrix ~ ~ ~")
    cmy = polyroots.solve([c1_switch_xy, c2_switch_xy], rand_poly = False)[:,::-1]
    print("Roots:\n",cmy)

    print("~ ~ ~ Power Form, Division Matrix 1/y ~ ~ ~")
    #pdy = polyroots.solve([p1_switch_xy, p2_switch_xy], method = "div")[:,::-1]
    #print("Roots:\n",pdy)

    print("~ ~ ~ Cheb Form, Division Matrix 1/y ~ ~ ~")
    #cdy = polyroots.solve([c1_switch_xy, c2_switch_xy], method = "div")[:,::-1]
    #print("Roots:\n",cdy)

    print("~ ~ ~ Power Form, Division Matrix 1/x ~ ~ ~")
    #pdx = polyroots.solve([p1, p2], method = "div")
    #print("Roots:\n",pdx)

    print("~ ~ ~ Cheb Form, Division Matrix 1/x ~ ~ ~")
    #cdx = polyroots.solve([c1, c2], method = "div")
    #print("Roots:\n",cdx)

    print("\n\nCompare Roots:\n")
    print("Power M_f\n", pmf)#perfect
    print("Cheb M_f\n", cmf)#really good
    print("Power M_x\n", pmx)#really good
    print("Cheb M_x\n", cmx)#really good
    print("Power M_y\n", pmy)#perfect
    print("Cheb M_y\n", cmy)#perfect
    #print("Power M_1/y\n", pdy)
    #print("Cheb M_1/y\n", cdy)
    #print("Power M_1/x\n", pdx)
    #print("Cheb M_1/x\n", cdx)

    #graph the polys
    delta = 0.01
    xrange = np.arange(-4, 3, delta)
    yrange = np.arange(-3, 3, delta)
    X, Y = np.meshgrid(xrange,yrange)

    plt.clf()
    ax2 = plt.subplot(111)
    ax2 = plt.subplot(111)
    ax2.set_title("With M_x and M_y Matrices")
    #polys
    ax2
    ax2.contour(X, Y, X**2 + Y**2 - 1, [0], colors="black", linestyles="dashed")
    ax2.contour(X, Y, 4*X**2 + Y**2 - 1, [0], colors="black", linestyles="dashed")
    #roots
    ax2.plot(pmx[:,0],pmx[:,1],".", color="gray", label="Power M_x",markersize=4)
    ax2.plot(cmx[:,0],cmx[:,1], ".", color="silver", label="Cheb M_x",markersize=4)
    ax2.plot(pmy[:,0],pmy[:,1],".", color="tan", label="Power M_y",markersize=4)
    ax2.plot(cmy[:,0],cmy[:,1], ".", color="peru", label="Cheb M_y",markersize=4)
    ax2.plot(pmf[:,0],pmf[:,1],".r", label="Power M_f",markersize=4)
    ax2.plot(cmf[:,0],cmf[:,1], ".m", label="Cheb M_f",markersize=4)
    #ax2.plot(pdy[:,0],pdy[:,1], "xb", label="Power M_1/y",markersize=6)
    #ax2.plot(cdy[:,0],cdy[:,1], ".c", label="Cheb M_1/y",markersize=4)
    #ax2.plot(pdx[:,0],pdx[:,1], "+", color = "gold", label="Power M_1/x",markersize=8)
    #ax2.plot(cdx[:,0],cdx[:,1], ".",color ="orange", label="Cheb M_1/x",markersize=4)
    ax2.axis("equal")

    plt.suptitle("Circle and inscribed ellipse centered at the origin")
    plt.figlegend(loc="lower left")
    plt.show()

if circle_ellipse_moved:
    print("Double Root Trials: \n p1 = 1.0*x^2 + 1.0*y^2 + 2.4*x - 6.2*y + 10.05 \n p2 = 4.0*x^2 + 1.0*y^2 + 9.6*x - 6.2*y + 14.37")
    print("Circle and inscribed ellipse moved away from the origin. Center now at (-1.2, 3.1), Double roots at (-1.2, 4.1) and (-1.2,2.1).")

    p1_coef = np.array([[10.05,-6.2,1],[2.4,0,0],[1,0,0]])
    p2_coef = np.array([[14.37,-6.2,1],[9.6,0,0],[4,0,0]])

    p1 = polynomial.MultiPower(p1_coef)
    p2 = polynomial.MultiPower(p2_coef)
    p1_switch_xy = polynomial.MultiPower(p1_coef.T)
    p2_switch_xy = polynomial.MultiPower(p2_coef.T)
    #In Cheb form

    c1_coef = np.array([[11.05,-6.2,.5],[2.4,0,0],[.5,0,0]])
    c2_coef = np.array([[16.87,-6.2,.5],[9.6,0,0],[2,0,0]])

    c1 = polynomial.MultiCheb(c1_coef)
    c2 = polynomial.MultiCheb(c2_coef)
    c1_switch_xy = polynomial.MultiCheb(c1_coef.T)
    c2_switch_xy = polynomial.MultiCheb(c2_coef.T)

    print("~ ~ ~ Power Form, M_f Matrix ~ ~ ~")
    pmf = polyroots.solve([p1, p2])
    print("Roots:\n",pmf)

    print("~ ~ ~ Cheb Form, M_f Matrix ~ ~ ~")
    cmf = polyroots.solve([c1, c2])
    print("Roots:\n",cmf)

    print("~ ~ ~ Power Form, M_x Matrix ~ ~ ~")
    pmx = polyroots.solve([p1, p2], rand_poly = False)
    print("Roots:\n",pmx)

    print("~ ~ ~ Cheb Form, M_x Matrix ~ ~ ~")
    cmx = polyroots.solve([c1, c2], rand_poly = False)
    print("Roots:\n",cmx)

    #flip left/right because x and y are switched. Same for M_y and M_1/y matrices below
    print("~ ~ ~ Power Form, M_y Matrix ~ ~ ~")
    pmy = polyroots.solve([p1_switch_xy, p2_switch_xy], rand_poly = False)[:,::-1]
    print("Roots:\n",pmy)

    print("~ ~ ~ Cheb Form, M_y Matrix ~ ~ ~")
    cmy = polyroots.solve([c1_switch_xy, c2_switch_xy], rand_poly = False)[:,::-1]
    print("Roots:\n",cmy)

    print("~ ~ ~ Power Form, Division Matrix 1/y ~ ~ ~")
    pdy = polyroots.solve([p1_switch_xy, p2_switch_xy], method = "div")[:,::-1]
    print("Roots:\n",pdy)

    print("~ ~ ~ Cheb Form, Division Matrix 1/y ~ ~ ~")
    cdy = polyroots.solve([c1_switch_xy, c2_switch_xy], method = "div")[:,::-1]
    print("Roots:\n",cdy)

    print("~ ~ ~ Power Form, Division Matrix 1/x ~ ~ ~")
    pdx = polyroots.solve([p1, p2], method = "div")
    print("Roots:\n",pdx)

    print("~ ~ ~ Cheb Form, Division Matrix 1/x ~ ~ ~")
    cdx = polyroots.solve([c1, c2], method = "div")
    print("Roots:\n",cdx)

    print("\n\nCompare Roots:\n")
    print("Actual Roots: (-1.2, 4.1) and (-1.2,2.1)")
    print("Power M_f\n", pmf)#Spot. On.
    print("Cheb M_f\n", cmf)#Spot. On.
    print("Power M_x\n", pmx)#x-coord right, y-coord wrong
    print("Cheb M_x\n", cmx)#x-coord right, y-coord wrong, worse than Power M_x
    print("Power M_y\n", pmy)#perfect y-coord, x-coord off by 1.11, other by 1.08
    print("Cheb M_y\n", cmy)#perfect y-coords, x-coord off by 1.11
    print("Power M_1/y\n", pdy)#y-coord perfect, x-coord wrong. One x-coord pretty close
    print("Cheb M_1/y\n", cdy)#got x-coord almost right on one root
    print("Power M_1/x\n", pdx)#x-coord perfect, y-coord wrong
    print("Cheb M_1/x\n", cdx)#x-coord right, y-coord wrong

    #graph the polys
    delta = 0.01
    xrange = np.arange(-2.5, 0, delta)
    yrange = np.arange(-2, 4.5, delta)
    X, Y = np.meshgrid(xrange,yrange)

    plt.clf()
    ax2 = plt.subplot(111)
    ax2 = plt.subplot(111)
    ax2.set_title("With M_x and M_y Matrices")
    #polys
    ax2
    ax2.contour(X, Y, 1.0*X**2 + 1.0*Y**2 + 2.4*X - 6.2*Y + 10.05, [0], colors="black", linestyles="dashed")
    ax2.contour(X, Y, 4.0*X**2 + 1.0*Y**2 + 9.6*X - 6.2*Y + 14.37, [0], colors="black", linestyles="dashed")
    #roots
    ax2.plot(pmx[:,0],pmx[:,1],".", color="gray", label="Power M_x",markersize=4)
    ax2.plot(cmx[:,0],cmx[:,1], ".", color="silver", label="Cheb M_x",markersize=4)
    ax2.plot(pmy[:,0],pmy[:,1],".", color="tan", label="Power M_y",markersize=4)
    ax2.plot(cmy[:,0],cmy[:,1], ".", color="peru", label="Cheb M_y",markersize=4)
    ax2.plot(pmf[:,0],pmf[:,1],".r", label="Power M_f",markersize=4)
    ax2.plot(cmf[:,0],cmf[:,1], ".m", label="Cheb M_f",markersize=4)
    ax2.plot(pdy[:,0],pdy[:,1], "xb", label="Power M_1/y",markersize=6)
    ax2.plot(cdy[:,0],cdy[:,1], ".c", label="Cheb M_1/y",markersize=4)
    ax2.plot(pdx[:,0],pdx[:,1], "+", color = "gold", label="Power M_1/x",markersize=8)
    ax2.plot(cdx[:,0],cdx[:,1], ".",color ="orange", label="Cheb M_1/x",markersize=4)
    ax2.axis("equal")

    plt.suptitle("Circle and inscribed ellipse centered at the origin")
    plt.figlegend(loc="lower left")
    plt.show()

if circle_ellipse_transformed:
    print("Double Root Trials: \n p1 = 2.08*x^2 + 3.92*x*y + 2.02*y^2 - 1 \n p2 = 6.4*x^2 + 10.4*x*y + 4.45*y^2 - 1 ")
    print("Circle and inscribed ellipse centered at the origin, then transformed. Roots at (1.5,-2)M2 and (-1.5,2)M2.")

    p1_coef = np.array([[-1,0,2.02],[0,3.92,0],[2.08,0,0]])
    p2_coef = np.array([[-1,0,4.45],[0,10.4,0],[6.4,0,0]])

    p1 = polynomial.MultiPower(p1_coef)
    p2 = polynomial.MultiPower(p2_coef)
    p1_switch_xy = polynomial.MultiPower(p1_coef.T)
    p2_switch_xy = polynomial.MultiPower(p2_coef.T)

    #In Cheb form

    c1_coef = np.array([[1.05,0,1.01],[0,3.92,0],[1.04,0,0]])
    c2_coef = np.array([[4.425,0,2.25],[0,10.4,0],[3.2,0,0]])

    c1 = polynomial.MultiCheb(c1_coef) # 1/2 T2x + 2 xy + 1/2 T2y - 5 y - 3 x + 5 == 0
    c2 = polynomial.MultiCheb(c2_coef) #- 1/2 T2x  -2 xy - 1/2 T2y + 3 y  + 5 x - 5 == 0
    c1_switch_xy = polynomial.MultiCheb(c1_coef.T)
    c2_switch_xy = polynomial.MultiCheb(c2_coef.T)

    print("~ ~ ~ Power Form, M_f Matrix ~ ~ ~")
    pmf = polyroots.solve([p1, p2])
    print("Roots:\n",pmf)

    print("~ ~ ~ Cheb Form, M_f Matrix ~ ~ ~")
    cmf = polyroots.solve([c1, c2])
    print("Roots:\n",cmf)

    print("~ ~ ~ Power Form, M_x Matrix ~ ~ ~")
    pmx = polyroots.solve([p1, p2], rand_poly = False)
    print("Roots:\n",pmx)

    print("~ ~ ~ Cheb Form, M_x Matrix ~ ~ ~")
    cmx = polyroots.solve([c1, c2], rand_poly = False)
    print("Roots:\n",cmx)

    #flip left/right because x and y are switched. Same for M_y and M_1/y matrices below
    print("~ ~ ~ Power Form, M_y Matrix ~ ~ ~")
    pmy = polyroots.solve([p1_switch_xy, p2_switch_xy], rand_poly = False)[:,::-1]
    print("Roots:\n",pmy)

    print("~ ~ ~ Cheb Form, M_y Matrix ~ ~ ~")
    cmy = polyroots.solve([c1_switch_xy, c2_switch_xy], rand_poly = False)[:,::-1]
    print("Roots:\n",cmy)

    print("~ ~ ~ Power Form, Division Matrix 1/y ~ ~ ~")
    pdy = polyroots.solve([p1_switch_xy, p2_switch_xy], method = "div")[:,::-1]
    print("Roots:\n",pdy)

    print("~ ~ ~ Cheb Form, Division Matrix 1/y ~ ~ ~")
    cdy = polyroots.solve([c1_switch_xy, c2_switch_xy], method = "div")[:,::-1]
    print("Roots:\n",cdy)

    print("~ ~ ~ Power Form, Division Matrix 1/x ~ ~ ~")
    pdx = polyroots.solve([p1, p2], method = "div")
    print("Roots:\n",pdx)

    print("~ ~ ~ Cheb Form, Division Matrix 1/x ~ ~ ~")
    cdx = polyroots.solve([c1, c2], method = "div")
    print("Roots:\n",cdx)

    print("\n\nCompare Roots:\n")
    #Everything Power really good, everything cheb fine
    print("Power M_f\n", pmf)
    print("Cheb M_f\n", cmf)
    print("Power M_x\n", pmx)
    print("Cheb M_x\n", cmx)
    print("Power M_y\n", pmy)
    print("Cheb M_y\n", cmy)
    print("Power M_1/y\n", pdy)
    print("Cheb M_1/y\n", cdy)
    print("Power M_1/x\n", pdx)
    print("Cheb M_1/x\n", cdx)

    #graph the polys
    delta = 0.01
    xrange = np.arange(-4, 3, delta)
    yrange = np.arange(-3, 3, delta)
    X, Y = np.meshgrid(xrange,yrange)

    plt.clf()

    ax2 = plt.subplot(111)
    ax2.set_title("With M_x and M_y Matrices")
    #polys
    ax2.contour(X, Y, 2.08*X**2 + 3.92*X*Y + 2.02*Y**2 - 1, [0], colors="black", linestyles="dashed")
    ax2.contour(X, Y, 6.4*X**2 + 10.4*X*Y + 4.45*Y**2 - 1, [0], colors="black", linestyles="dashed")
    #roots
    ax2.plot(pmx[:,0],pmx[:,1],".", color="gray", label="Power M_x",markersize=4)
    ax2.plot(cmx[:,0],cmx[:,1], ".", color="silver", label="Cheb M_x",markersize=4)
    ax2.plot(pmy[:,0],pmy[:,1],".", color="tan", label="Power M_y",markersize=4)
    ax2.plot(cmy[:,0],cmy[:,1], ".", color="peru", label="Cheb M_y",markersize=4)
    ax2.plot(pmf[:,0],pmf[:,1],".r", label="Power M_f",markersize=4)
    ax2.plot(cmf[:,0],cmf[:,1], ".m", label="Cheb M_f",markersize=4)
    ax2.plot(pdy[:,0],pdy[:,1], "xb", label="Power M_1/y",markersize=6)
    ax2.plot(cdy[:,0],cdy[:,1], ".c", label="Cheb M_1/y",markersize=4)
    ax2.plot(pdx[:,0],pdx[:,1], "+", color = "gold", label="Power M_1/x",markersize=8)
    ax2.plot(cdx[:,0],cdx[:,1], ".",color ="orange", label="Cheb M_1/x",markersize=4)
    ax2.axis("equal")

    plt.suptitle("Circle and inscribed ellipse centered at the origin, then transformed")
    plt.figlegend(loc="lower left")
    plt.show()

if circle_ellipse_transformed_moved:
    print("Double Root Trials: \n p1 = 2.08*x^2 + (3.92*x - 7.82)*y + 2.02*y^2 - 7.16*x + 6.825 \n p2 = 6.4*x^2 + (10.4*x - 15.11)*y + 4.45*y^2 - 16.88*x + 13.2925 - 1")
    print("Circle and inscribed ellipse transformed and moved away from the origin.")

    p1_coef = np.array([[6.825, -7.82, 2.02],[-7.16, 3.92, 0], [2.08,0,0]])
    p2_coef = np.array([[12.2925, -15.11, 4.45],[-16.88, 10.4, 9],[6.4,0,0]])

    p1 = polynomial.MultiPower(p1_coef)
    p2 = polynomial.MultiPower(p2_coef)
    p1_switch_xy = polynomial.MultiPower(p1_coef.T)
    p2_switch_xy = polynomial.MultiPower(p2_coef.T)

    #In Cheb form

    c1_coef = np.array([[8.875,-7.82,1.01],[-7.16,3.92,0],[1.04,0,0]])
    c2_coef = np.array([[17.7175,-15.11,2.225],[-16.88,10.4,0],[3.2,0,0]])

    c1 = polynomial.MultiCheb(c1_coef)
    c2 = polynomial.MultiCheb(c2_coef)
    c1_switch_xy = polynomial.MultiCheb(c1_coef.T)
    c2_switch_xy = polynomial.MultiCheb(c2_coef.T)

    print("~ ~ ~ Power Form, M_f Matrix ~ ~ ~")
    pmf = polyroots.solve([p1, p2])
    print("Roots:\n",pmf)

    print("~ ~ ~ Cheb Form, M_f Matrix ~ ~ ~")
    cmf = polyroots.solve([c1, c2])
    print("Roots:\n",cmf)

    print("~ ~ ~ Power Form, M_x Matrix ~ ~ ~")
    pmx = polyroots.solve([p1, p2], rand_poly = False)
    print("Roots:\n",pmx)

    print("~ ~ ~ Cheb Form, M_x Matrix ~ ~ ~")
    cmx = polyroots.solve([c1, c2], rand_poly = False)
    print("Roots:\n",cmx)

    #flip left/right because x and y are switched. Same for M_y and M_1/y matrices below
    print("~ ~ ~ Power Form, M_y Matrix ~ ~ ~")
    pmy = polyroots.solve([p1_switch_xy, p2_switch_xy], rand_poly = False)[:,::-1]
    print("Roots:\n",pmy)

    print("~ ~ ~ Cheb Form, M_y Matrix ~ ~ ~")
    cmy = polyroots.solve([c1_switch_xy, c2_switch_xy], rand_poly = False)[:,::-1]
    print("Roots:\n",cmy)

    print("~ ~ ~ Power Form, Division Matrix 1/y ~ ~ ~")
    pdy = polyroots.solve([p1_switch_xy, p2_switch_xy], method = "div")[:,::-1]
    print("Roots:\n",pdy)

    print("~ ~ ~ Cheb Form, Division Matrix 1/y ~ ~ ~")
    cdy = polyroots.solve([c1_switch_xy, c2_switch_xy], method = "div")[:,::-1]
    print("Roots:\n",cdy)

    print("~ ~ ~ Power Form, Division Matrix 1/x ~ ~ ~")
    pdx = polyroots.solve([p1, p2], method = "div")
    print("Roots:\n",pdx)

    print("~ ~ ~ Cheb Form, Division Matrix 1/x ~ ~ ~")
    cdx = polyroots.solve([c1, c2], method = "div")
    print("Roots:\n",cdx)

    print("\n\nCompare Roots:\n")
    print("Real Roots: (-2.7, 5.1) and (.3,1.1)")
    #Everything power bad, gives the same 4 roots. Everything cheb basically perfect
    print("Power M_f\n", pmf)#Good
    print("Cheb M_f\n", cmf)#Good
    print("Power M_x\n", pmx)
    print("Cheb M_x\n", cmx)#Good
    print("Power M_y\n", pmy)
    print("Cheb M_y\n", cmy)#Good
    print("Power M_1/y\n", pdy)
    print("Cheb M_1/y\n", cdy)#Good
    print("Power M_1/x\n", pdx)
    print("Cheb M_1/x\n", cdx)#Good

    #graph the polys
    delta = 0.01
    xrange = np.arange(-4, 2, delta)
    yrange = np.arange(0, 6, delta)
    X, Y = np.meshgrid(xrange,yrange)

    ax2 = plt.subplot(111)
    ax2.set_title("With M_x and M_y Matrices")
    #polys
    ax2.contour(X, Y, 2.08*X**2 + (3.92*X - 7.82)*Y + 2.02*Y**2 - 7.16*X + 7.825 - 1, [0], colors="black", linestyles="dashed")
    ax2.contour(X, Y, 6.4*X**2 + (10.4*X - 15.11)*Y + 4.45*Y**2 - 16.88*X + 13.2925 - 1, [0], colors="black", linestyles="dashed")
    #roots
    ax2.plot((-2.7, .3),(5.1,1.1), '.', color = 'black', markersize = 10)
    ax2.plot(pmx[:,0],pmx[:,1],".", color="gray", label="Power M_x",markersize=4)
    ax2.plot(cmx[:,0],cmx[:,1], ".", color="silver", label="Cheb M_x",markersize=4)
    ax2.plot(pmy[:,0],pmy[:,1],".", color="tan", label="Power M_y",markersize=4)
    ax2.plot(cmy[:,0],cmy[:,1], ".", color="peru", label="Cheb M_y",markersize=4)
    ax2.plot(pmf[:,0],pmf[:,1],".r", label="Power M_f",markersize=4)
    ax2.plot(cmf[:,0],cmf[:,1], ".m", label="Cheb M_f",markersize=4)
    ax2.plot(pdy[:,0],pdy[:,1], "xb", label="Power M_1/y",markersize=6)
    ax2.plot(cdy[:,0],cdy[:,1], ".c", label="Cheb M_1/y",markersize=4)
    ax2.plot(pdx[:,0],pdx[:,1], "+", color = "gold", label="Power M_1/x",markersize=8)
    ax2.plot(cdx[:,0],cdx[:,1], ".",color ="orange", label="Cheb M_1/x",markersize=4)
    ax2.axis("equal")

    plt.suptitle("Circle and inscribed ellipse transformed and moved away from the origin")
    plt.figlegend(loc="lower left")
    plt.show()
