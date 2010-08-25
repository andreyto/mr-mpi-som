#!/usr/bin/env python

f = open("index_test_infile.txt", "w")
for i in range(1,10001):
    f.write(str(i) + "\n")
f.close()
