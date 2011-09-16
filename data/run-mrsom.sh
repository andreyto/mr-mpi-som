#!/bin/bash



# NORMAL
../build/src/txt2bin/txt2bin rgbs.txt rgbs.bin 3 30 &&
mpirun -np 4 ../build/src/mrsom -m train -i rgbs.bin -o rgbs -e 10 -n 30 -d 3 -b 4 &&
../build/src/mrsom -m test -c rgbs-codebook.txt -i rgbs.txt -o rgbs -d 3 -n 10 &&
./umat2fig.py rgbs-umat.txt rgbs-umat.png &&

# SPARSE
../build/src/txt2bin/txt2bin-sparse rgbs.txt rgbs 3 30 &&
mpirun -np 4 ../build/src/mrsom -m train -s 1 -i rgbs-sparse.bin -x rgbs-sparse.idx -t rgbs-sparse.num -o rgbs-sparse -e 20 -d 4 -n 30 -b 4 &&
../build/src/mrsom -m test -c rgbs-sparse-codebook.txt -i rgbs.txt -o rgbs -d 3 -n 10 &&
./umat2fig.py rgbs-sparse-umat.txt rgbs-sparse-umat.png &&

# Gen random matrix
./gen_randmat.py ./rand/randmat.txt 30 300 &&
../build/src/txt2bin/txt2bin-sparse ./rand/randmat.txt ./rand/randmat 30 300 &&
mpirun -np 4 ../build/src/mrsom -m train -s 1 -i ./rand/randmat-sparse.bin -x ./rand/randmat-sparse.idx -t ./rand/randmat-sparse.num -o ./rand/randmat-sparse -e 20 -d 30 -n 300 -b 8 &&
../build/src/mrsom -m test -c ./rand/randmat-sparse-codebook.txt -i ./rand/randmat.txt -o ./rand/randmat -d 30 -n 30 &&
./umat2fig.py ./rand/randmat-sparse-umat.txt ./rand/randmat-sparse-umat.png
