#!/bin/bash
../build/src/txt2bin rgbs.txt rgbs.bin 3 28 &&
mpirun -np 4 ../build/src/mrsom -m train -i rgbs.bin -o rgbs -e 10 -n 28 -d 3 -b 7 &&
../build/src/mrsom -m test -c rgbs-codebook.txt -i rgbs.txt -o rgbs -d 3 -n 10

#./gen_randmat.py randmat.txt 28 3 
#../build/src/txt2bin randmat.txt randmat.bin 3 28 &&
#mpirun -np 4 ../build/src/mrsom -m train -i rgbs.bin -e 10 -n 28 -d 3 -b 7 &&
