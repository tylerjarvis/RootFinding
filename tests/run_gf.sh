#!bin/sh

for coeff_num in {0..9}
do
  nohup python -u growth_factors.py 9 > gf9out$coeff_num.txt 2> gf9err$coeff_num.txt &
done
