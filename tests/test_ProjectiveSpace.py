from numalgsolve.ProjectiveSpace import *
from numalgsolve.polynomial import MultiCheb, MultiPower

def test_common_root_at_inf():

    f = MultiPower(np.array([[0,1]])) #f(x,y) = y
    g = MultiPower(np.array([[0,0,-1],[1,0,0]])) #f(x,y) = x - y^2
    #assert common_root_at_inf([f,g]) == True

    f = MultiPower(np.array([[0,1]])) #f(x,y) = y
    g = MultiPower(np.array([[0,0,-1],[1,0,0]]).T) #f(x,y) = x^2 - y
    #assert common_root_at_inf([f,g]) == False

    #from ./Easy/dpdx-dpdy_ex016.mat, which is a dupliate of ./Easy/dpdx-dpdy_ex007.mat and almost the same as ./Easy/dpdx-dpdy_ex008.mat
    f = MultiPower(np.array([[-0.21875,  0.,       1.3125,   0.,      -1.09375,  0.,       0.21875],
                             [ 0.,      -2.625,    0.,       4.375,    0.,      -1.3125,   0.],
                             [ 1.3125,   0. ,     -6.5625,   0. ,      3.28125,  0. ,      0.],
                             [ 0. ,      4.375,    0. ,     -4.375,    0. ,      0. ,      0.],
                             [-1.09375,  0. ,      3.28125,  0. ,      0. ,      0. ,      0. ],
                             [ 0. ,     -1.3125,   0. ,      0. ,      0. ,      0. ,      0.],
                             [ 0.21875,  0. ,      0. ,      0. ,      0. ,      0. ,      0.]]))
    g = MultiPower(np.array([[ 0.     ,  -0.65625,   0.     ,   1.09375,   0.      , -0.328125],
                             [ 0.65625,   0.      , -3.28125,   0.     ,   1.640625,  0.      ],
                             [ 0.     ,   3.28125 ,  0.     ,  -3.28125,   0.      ,  0.      ],
                             [-1.09375,   0.      ,  3.28125,   0.     ,   0.      ,  0.      ],
                             [ 0.     ,  -1.640625,  0.     ,   0.     ,   0.      ,  0.      ],
                             [ 0.328125,  0.      ,  0.     ,   0.     ,   0.      ,  0.      ]]))
    assert common_root_at_inf([f,g]) == True

    #from ./Easy/dpdx-dpdy_C1.mat
    f = MultiPower(np.array([[ 0.    , 1.  ,  -1.5  , -0.5  , -1.875,  0.   ,  0.875],
                             [ 0.    ,-3.  ,   1.5  ,  1.5  , -0.625,  0.   ,  0.   ],
                             [-0.125 , 2.5 ,   0.375, -1.5  ,  1.25 ,  0.   ,  0.   ],
                             [ 0.125 , 0.5 ,  -0.75 ,  0.5  ,  0.   ,  0.   ,  0.   ],
                             [ 0.125 ,-1.5 ,   0.375,  0.   ,  0.   ,  0.   ,  0.   ],
                             [-0.125 , 0.5 ,   0.   ,  0.   ,  0.   ,  0.   ,  0.   ]]))
    g = MultiPower(np.array([[ 0.,    0.25 , 9.  ],
                             [ 0.,    0.,    0.  ],
                             [-0.75, -1.,    0.  ]]))
    #assert common_root_at_inf([f,g]) == ??

    f = MultiPower(np.array([[0., 0., 0., 1.]]))
    g = MultiPower(np.array([[0., 0., 0., 0., 1.25]]))
    assert common_root_at_inf([f,g]) == True

    f = MultiPower(np.array([[0., 0., 0., 1.]]))
    g = MultiPower(np.array([[0., 0., 0., 0., 1.25]]))
    assert common_root_at_inf([f,g]) == True

def test_roots_at_inf():
    g = MultiPower(np.array([[0,0,1],[-1,0,0]]).T) #f(x,y) = x^2 - y
    assert roots_at_inf(g) == [(0,1)]

    f = MultiPower(np.array([[0,1]])) #f(x,y) = y
    g = MultiPower(np.array([[0,0,-1],[1,0,0]])) #f(x,y) = x - y^2
    assert roots_at_inf(f) == [(1,0)]
    assert roots_at_inf(g) == [(1,0)]


def test_pad_with_zeros():
    assert (pad_with_zeros(np.array([[0, 3],
                                    [1, 4],
                                    [2, 5]])) == np.array([[0, 3, 0],
                                                           [1, 4, 0],
                                                           [2, 5, 0]])).all()
    assert (pad_with_zeros(np.array([[0, 3, 6],
                                    [1, 4, 7]])) == np.array([[0, 3, 6],
                                                              [1, 4, 7],
                                                              [0, 0, 0]])).all()
    assert (pad_with_zeros(np.array([[0, 3.5],
                                    [1, 4],
                                    [2, 5],
                                    [5, 7.8]])) == np.array([[0, 3.5, 0, 0],
                                                             [1, 4, 0, 0],
                                                             [2, 5, 0, 0],
                                                             [5, 7.8, 0, 0]])).all()
    assert (pad_with_zeros(np.array([[0, 3, 6, 4],
                                    [-1, 4, 7, 4]])) == np.array([[0, 3, 6, 4],
                                                                  [-1, 4, 7, 4],
                                                                  [0, 0, 0, 0],
                                                                  [0, 0, 0, 0]])).all()
