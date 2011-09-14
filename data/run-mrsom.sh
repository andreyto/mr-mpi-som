#!/bin/bash

# Gen random matrix
#./gen_randmat.py randmat.txt 28 3 &&

# NORMAL
../build/src/txt2bin/txt2bin rgbs.txt rgbs.bin 3 30 &&
mpirun -np 4 ../build/src/mrsom2 -m train -i rgbs.bin -o rgbs -e 10 -n 30 -d 3 -b 4 &&
../build/src/mrsom2 -m test -c rgbs-codebook.txt -i rgbs.txt -o rgbs -d 3 -n 10 &&
./umat2fig.py rgbs-umat.txt rgbs-umat.png &&

# SPARSE
../build/src/txt2bin/txt2bin-sparse rgbs.txt rgbs 3 30 &&
mpirun -np 4 ../build/src/mrsom2 -m train -s 1 -i rgbs-sparse.bin -x rgbs-sparse.idx -t rgbs-sparse.num -o rgbs-sparse -e 10 -n 30 -d 3 -b 4 &&
../build/src/mrsom2 -m test -c rgbs-sparse-codebook.txt -i rgbs.txt -o rgbs -d 3 -n 10 &&
./umat2fig.py rgbs-sparse-umat.txt rgbs-sparse-umat.png 
