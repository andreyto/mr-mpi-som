import sys, glob
import random
 
class Matrix(object):
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        # initialize matrix and fill with zeroes
        self.matrix = []
        for i in range(rows):
            ea_row = []
            for j in range(cols):
                ea_row.append(0)
            self.matrix.append(ea_row)
    
    def setitem(self, col, row, v):
        self.matrix[col-1][row-1] = v
    
    def getitem(self, col, row):
        return self.matrix[col-1][row-1]
        
    def getrow(self, col):
         return self.matrix[col-1]
    
    def __repr__(self):
        outStr = ""
        for i in range(self.rows):
            outStr += 'Row %s = %s\n' % (i+1, self.matrix[i])
        return outStr
        
if __name__ == "__main__":
    
    if len(sys.argv) != 4:   
        print "Usage: python gen_rand_matrix.py outFile cols rows"
        sys.exit(1)
        
    x = int(sys.argv[2]);
    y = int(sys.argv[3]);
    filename = sys.argv[1];    
    
    m0 = Matrix(y,x)    
    for i in range(m0.rows):
        for j in range(m0.cols):
            m0.setitem(i, j, format(random.random(), '0.2f'));    
    #print m0
    FILE = open(filename,"w")
    for i in range(m0.rows):
        for j in range(m0.cols):
            FILE.write(m0.getitem(i, j) + ",")
        FILE.write("\n")
    FILE.close()

