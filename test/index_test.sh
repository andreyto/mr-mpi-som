#!/bin/bash

cat index_test_infile.txt | ./getoffsets.py > index.dat 
echo 'Line #1'
./getline.py index_test_infile.txt index.dat 0
echo 'Line #499'
./getline.py index_test_infile.txt index.dat 499
echo 'Line #10000'
./getline.py index_test_infile.txt index.dat 9999


