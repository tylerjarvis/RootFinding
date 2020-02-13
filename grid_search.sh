#!/bin/bash

python3 tol_gs.py qrt 1 $> tol_gs_qrt_1.txt &
python3 tol_gs.py qrt 2 $> tol_gs_qrt_2.txt &
python3 tol_gs.py qrt 3 $> tol_gs_qrt_3.txt &

python3 tol_gs.py svd 1 $> tol_gs_svd_1.txt &
python3 tol_gs.py svd 2 $> tol_gs_svd_2.txt &
python3 tol_gs.py svd 3 $> tol_gs_svd_3.txt &

python3 tol_gs.py tvb 1 $> tol_gs_tvb_1.txt &
python3 tol_gs.py tvb 2 $> tol_gs_tvb_2.txt &
python3 tol_gs.py tvb 3 $> tol_gs_tvb_3.txt &