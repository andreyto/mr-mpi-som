from pylab import *
from numpy import *
import sys
 
fileName = sys.argv[1]
outFileName = sys.argv[2]
R_matrix = loadtxt(fileName)
    
imshow(R_matrix)
savefig(outFileName)

