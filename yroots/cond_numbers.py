import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt


a_space = np.linspace(0,1,11)
b_space = np.linspace(-1,1,11)

for a in a_space:
    C_1_cond = []
    C_2_cond = []
    C_3_cond = []
    for b in b_space:

        C_1 = np.array([[1,b],
                        [0,a]])

        sing1 = linalg.svd(C_1)[1]
        cond1 = sing1[0] / sing1[-1]
        C_1_cond.append(cond1)

        C_2 = np.array([[1,b,2*b**2+a**2-1],
                        [0,a,4*a*b],
                        [0,0,a**2]])

        sing2 = linalg.svd(C_2)[1]
        cond2 = sing2[0] / sing2[-1]
        C_2_cond.append(cond2)

        C_3 = np.array([[1,b,2*b**2+a**2-1,6*(a**2)*b+4*b**3-3*b],
                        [0,a,4*a*b,3*a**3+12*a*b**2-3*a],
                        [0,0,a**2,6*(a**2)*b],
                        [0,0,0,a**3]])

        sing3 = linalg.svd(C_3)[1]
        cond3 = sing3[0] / sing3[-1]
        C_3_cond.append(cond3)

    plt.scatter(b_space,C_1_cond,label="C1 (2x2)")
    plt.scatter(b_space,C_2_cond,label="C2 (3x3)")
    plt.scatter(b_space,C_3_cond,label="C3 (4x4)")
    #plt.set_yscale("log")
    plt.title("a={}".format(a))
    plt.legend()
    plt.ylabel("condition number")
    plt.xlabel("beta")
    plt.show()

        
#svd
#ratios