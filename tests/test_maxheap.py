import numpy as np
import os, sys
from groebner.maxheap import MaxHeap, Term
import pytest



def test_push_pop():
    a0 = Term((0,0,1,0,0))
    a1 = Term((0,1,1,3,1))
    a2 = Term((0,1,1,3,0,0,0,1))
    a3 = Term((2,2,2,3,4,1,4,3))
    a4 = Term((0,1,1,2,2))
    maxh = MaxHeap()
    maxh.heappush(a1)
    maxh.heappush(a3)
    maxh.heappush(a0)
    maxh.heappush(a2)
    maxh.heappush(a4)
    assert maxh.heappop() == a3
    assert maxh.heappop() == a2

    maxh.heappush(a3)
    maxh.heappush(a3)

    assert maxh.heappop() == a3
    assert maxh.heappop() == a1
    assert maxh.heappop() == a4
    assert maxh.heappop() == a0
