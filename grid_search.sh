#!/bin/bash

python3 tol_gs.py svd 9 $> svd_deg_9.txt & 2> svd_deg_9.err &
python3 tol_gs.py svd 15 $> svd_deg_15.txt & 2> svd_deg_15.err &
python3 tol_gs.py svd 20 $> svd_deg_20.txt & 2> svd_deg_20.err &
python3 tol_gs.py svd 25 $> svd_deg_25.txt & 2> svd_deg_25.err &