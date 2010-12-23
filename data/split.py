#!/usr/bin/python
 
import sys
 
    
argc = len(sys.argv)
print argc
if (argc != 5):
    print "python split.py input chunksize numfile outPrefix"
    exit(0)
    
input=open(sys.argv[1], 'r')
L=input.readlines()
 
chunkLineSize = int(sys.argv[2])
numFile = int(sys.argv[3])
outfilenamed = sys.argv[4]
line = 1;
linecount = 0;
outfilenamearray = []

for i in range(0, numFile):
    if line < 10:
        outfilename = outfilenamed + "000" + str(line) + ".txt"
    elif line < 100:
        outfilename = outfilenamed + "00" + str(line) + ".txt"
    elif line < 1000:
        outfilename = outfilenamed + "0" + str(line) + ".txt"
        
    outfile = open(outfilename, "w")
    for j in range(0, chunkLineSize):
        outfile.write(L[linecount])
        linecount += 1
    print "File =", outfilename
    outfilenamearray.append(outfilename)
    line += 1
    outfile.close()
    
masterfilename = "master-" + sys.argv[2] + "-" + sys.argv[3] + ".txt"
masterfile = open(masterfilename, "w")
for i in range(0, numFile):
    masterfile.write(outfilenamearray[i] + "\n")
masterfile.close()
    
 
