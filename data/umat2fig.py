#!/usr/bin/env python
from pylab import *
from numpy import *
import sys
 
 
if __name__ == '__main__':
     
    if len(sys.argv) != 3:
        print "python umat2fig.py umat_infile fig_outfile"
        print "Usage: ./umat2fig.py rgbs-umat.txt rgbs-umat.png"
        sys.exit(1) 
        
    fileName = sys.argv[1]
    outFileName = sys.argv[2]
    R_matrix = loadtxt(fileName)
        
    imshow(R_matrix)
    savefig(outFileName)

