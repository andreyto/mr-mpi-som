#!/usr/bin/env python

import pypar
import Numeric
import numpy
import sys

from numpy.random import shuffle
from random import *
from math import *

import struct
#fmt = 'L' ## unsigned long
fmt = 'Q' ## unsigned long long
fmtSize = struct.calcsize(fmt)
        
def work(myid, numProcs, inFileName, nodes, width, height, \
             radiusDecaying, radDivVal, learningRateDecaying, deltaNodes, myData):

    ## NOTE: DO NOT NEED TO READ ALL FV. IMPROVE THIS!
    #f = open(inFileName, "r")
    #trainVector = []
    #count = 0
    #dt = numpy.dtype(float)
    #try:
        #for line in f:
            #values = line.rstrip().split(' ')
            #trainVector.append(numpy.array(values, dt))
    #finally:
        #f.close()
    
    #interval = len(trainVector)
    #myInterval = interval/numProcs
    #myLower = myid*myInterval
    
    #if myid == numProcs-1:
        #myUpper = interval+1
    #else:
        #myUpper = myLower + myInterval
    
    #myData = trainVector[myLower:myUpper]
    
    for j in range(len(myData)):
        best = best_match(nodes, width, myData[j])
        for loc in find_neighborhood(height, width, best, radiusDecaying):
            influence = exp( (-1.0 * (loc[2]**2)) / radDivVal)
            inf_lrd = influence * learningRateDecaying
            deltaNodes[loc[0],loc[1]] += inf_lrd * (myData[j] - nodes[loc[0],loc[1]])
        
    #print "### Rank =", myid, "processed the data from", myLower, "to", myUpper#, myData, deltaNodes
    return
  
  
def broadcast(x, root, myid, numProcs,):
    list = range(0, root) + range(root + 1, numProcs)

    # The process of rank 'root' sends 'x' to all other processes.
    if root == myid:
        # Determine the size of bsend's buffer.
        for i in list:
            pypar.push_for_alloc(x)

        # Allocate and attach bsend's buffer.
        pypar.alloc_and_attach()

        # Make all sends.
        for i in list:
            pypar.bsend(x, i)

        # Deallocate and detach bsend's buffer.
        pypar.detach_and_dealloc()

        x_ = x

    # All processes with rank distinct from 'root' start receiving.
    else:
        x_ = pypar.receive(root)
        
        
    return x_

def find_neighborhood(height, width, pt, dist):
    min_y = max(int(pt[0] - dist), 0)
    max_y = min(int(pt[0] + dist), height)
    min_x = max(int(pt[1] - dist), 0)
    max_x = min(int(pt[1] + dist), width)
    neighbors = []
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            dist = abs(y-pt[0]) + abs(x-pt[1])
            neighbors.append((y,x,dist))
    return neighbors

def best_match(nodes, width, target_FV):
    loc = numpy.argmin((((nodes - target_FV)**2).sum(axis=2))**0.5)
    r = 0
    while loc > width:
        loc -= width
        r += 1
    c = loc
    return (r, c)

def FV_distance(FV_1, FV_2):
    return (sum((FV_1 - FV_2)**2))**0.5
    
def eucl_dist(x,y):
    return numpy.sqrt(numpy.sum((x-y)**2))

def floatFormat(value):
    return "%.3f" % value
    
def save_2D_dist_map(fname, nodes, width, height):
    D = 2
    minDist = 1.5
    f = open(fname, "w")
    n = 0
    for r in range(height):
        for c in range(width):
            #print r, c, nodes[r,c]
            dist = 0.0
            nodesNum = 0
            #w1 = []
            for r2 in range(height):
                for c2 in range(width):
                    #w1 = nodes[r2, c2]
                    #w2 = []
                    for r3 in range(height):
                        for c3 in range(width):
                            if r2 == r3 and c2 == c3:
                                continue
                            #tmp = 0.0
                            #w2 = nodes[r3, c3]
                            #for x in range(D):
                                #tmp += pow(*(get_coords(pnode) + x) - *(get_coords(node) + x), 2.0f);
                            #tmp = sqrt(tmp);
                            co1 = numpy.array((r2, c2))
                            co2 = numpy.array((r3, c3))
                            tmp = eucl_dist(co1, co2) # coord dist
                            #print r2, c2, r3, c3, tmp
                            if tmp <= minDist:
                                nodesNum += 1
                                dist += eucl_dist(nodes[r2, c2], nodes[r3, c3]) # weight dist
                    
            dist /= nodesNum
            f.write(str(floatFormat(dist)))
        f.write("\n")
    f.close()
    return

def F(n, fn):
    '''
    Return the byte offset of line n from index file fn.
    '''
    f = open(fn)
    try:
        f.seek(n * fmtSize)
        data = f.read(fmtSize)
    finally:
        f.close()

    return struct.unpack(fmt, data)[0]


def getline(n, dataFile, indexFile):
    '''
    Return line n from data file using index file.
    '''
    n = F(n, indexFile)
    f = open(dataFile)
    try:
        f.seek(n)
        data = f.readline()
    finally:
        f.close()

    return data

def f(n): 
    #return struct.pack('L', n)
    return struct.pack('Q', n) # unsigned long long
    
print "Initialization..."
width = 50
height = 50
radius = (height+width)/3
learningRate = 0.05
iterations = 0
FVLen = 0
FVDim = 0

## INIT
if len(sys.argv) == 7:
    iterations = int(sys.argv[2])
    FVLen = int(sys.argv[3])
    FVDim = int(sys.argv[4])
    width = int(sys.argv[5])
    height = int(sys.argv[6])
elif len(sys.argv) == 5:
    iterations = int(sys.argv[2])
    FVLen = int(sys.argv[3])
    FVDim = int(sys.argv[4])
else:
    print "Usage: mpirun -np nProc python mrsom2.py dataFile numIter numFeature DimFeature [somX] [somY]"
    sys.exit(1)

## GET PARALLEL PARAMETERS
MPI_myid = pypar.rank()
MPI_numproc = pypar.size()
MPI_node = pypar.get_processor_name()
#print "I am proc %d of %d on node %s" %(MPI_myid, MPI_numproc, MPI_node)

## SOM NODES INIT AND BROADCASTING
if MPI_myid == 0:
    nodes = numpy.array([[ [random() for i in range(FVDim)] for x in range(width)] for y in range(height)])
else:
    nodes = numpy.array([[[0 for i in range(FVDim)] for x in range(width)] for y in range(height)])
nodes = broadcast(nodes, 0, MPI_myid, MPI_numproc)
pypar.barrier()
#print "Rank " + str(MPI_myid) + " received: " + str(nodes)

## COMMON INFORMATION
inFileName = sys.argv[1]
timeConstant = iterations / log(radius)
radiusDecaying = 0
radDivVal = 0
learningRateDecaying = 0
## init weights
deltaNodes = numpy.array([[[0 for i in range(FVDim)] for x in range(width)] for y in range(height)])

## NOTE: DO NOT NEED TO READ ALL FV. IMPROVE THIS!
## IMPROVE
## 1. Make index file for the inFile
## 2. Decide from-to index to divide the inFile
## 3. Read a chunk of the inFile 

## MAKE INDEX FILE
indexFileName = inFileName+".idx"
#if MPI_myid == 0:
    #import subprocess
    #cmd = " cat " + inFileName + " | ./getoffsets.py > " + indexFileName
    #proc = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,)
    #stdout_value = proc.communicate()[0]
    #s = stdout_value.split()
    #for i in range(len(s)):
        #print str(i) + " " + str(s[i])

#total = 0
#for n, line in enumerate(fileinput.input()):
    #sys.stdout.write(f(total))
    #total += len(line)
        
#n = int(lineno)
#n = range(0,3)
#for l in n:
    #print "### Rank " + str(MPI_myid) + " -> " + getline(l, inFileName, indexFileName)

    
## DECIDE A RANGE OF DATA       
interval = FVLen
myInterval = interval / MPI_numproc
myLower = MPI_myid * myInterval
if MPI_myid == MPI_numproc - 1:
    #myUpper = interval + 1
    myUpper = interval
else:
    myUpper = myLower + myInterval
     
## EACH WORKER READS ITS OWN PART OF DATA FROM THE INPUT FILE
trainVector = []
count = 0
dt = numpy.dtype(float)
n = range(myLower, myUpper)
#print len(n)
for l in n:
    line = getline(l, inFileName, indexFileName)
    #print line
    values = line.rstrip().split(' ')
    #print values
    trainVector.append(numpy.array(values, dt))
print "### Rank " + str(MPI_myid) + " has " + str(len(trainVector)) + \
      " rows of data (%d - %d)" % (myLower+1, myUpper)


#f = open(inFileName, "r")
#trainVector = []
#count = 0
#dt = numpy.dtype(float)
#try:
    #for line in f:
        #values = line.rstrip().split(' ')
        #trainVector.append(numpy.array(values, dt))
#finally:
    #f.close()
    
                    
for i in range(1, iterations+1):
    deltaNodes.fill(0)

    if MPI_myid == 0:
        radiusDecaying = radius*exp(-1.0*i/timeConstant)
        radDivVal = 2 * radiusDecaying * i
        learningRateDecaying = learningRate*exp(-1.0*i/timeConstant)
        #sys.stdout.write("\rTraining Iteration: " + str(i) + "/" + str(iterations))
        print "########################################### Training Iteration:", i
        print "### radiusDecaying: ", radiusDecaying
    
    ## BROADCAST UPDATED INFO
    radiusDecaying = broadcast(radiusDecaying, 0, MPI_myid, MPI_numproc)
    radDivVal = broadcast(radDivVal, 0, MPI_myid, MPI_numproc)
    learningRateDecaying = broadcast(learningRateDecaying, 0, MPI_myid, MPI_numproc)
    #print "### Rank " + str(MPI_myid) + " radiusDecaying: " + str(radiusDecaying)
    #print "Rank " + str(MPI_myid) + " " + inFileName
    pypar.barrier()
                
    ## TRAINING IN PAPRALLEL FOR GETTING NEW WIEGHT VECTORS FOR SOME NODES
    work(MPI_myid, MPI_numproc, inFileName, nodes, width, height, \
         radiusDecaying, radDivVal, learningRateDecaying, \
         deltaNodes, trainVector)
    shuffle(trainVector)
    #print "Proc %d finished working" % MPI_myid
    
    ## GATHERING SUB RESULTS
    if MPI_numproc > 1:
        if MPI_myid == 0:
            ## UPDATE NODES WEIGHTS WITH ROOT'S RESULT
            nodes += deltaNodes 
            #print "P%d updated its result" % (0)
            for id in range(1, MPI_numproc):
                #print "P%d receving the result from P%d" % (0, id)
                ## UPDATE NODE WEIGHTS WITH DELTA FROM EACH WORKERS
                nodes = nodes + pypar.receive(id)
        else:
            #print "P%d sending to P%d" % (MPI_myid, 0)
            pypar.send(deltaNodes, 0)
            #print "Proc %d after seding the result" % MPI_myid
    
    ## BROADCAST UPDATED NODE WIEGHTS
    nodes = broadcast(nodes, 0, MPI_myid, MPI_numproc)
    pypar.barrier()

## DEBUG (ONLY WORKS FOR FVDim=3 RGB DATA)
if MPI_myid == 0:
    print "Saving..."
    try:
        from PIL import Image
        print "Saving Image: sompy_test_colors.png..."
        img = Image.new("RGB", (width, height))
        for r in range(height):
            for c in range(width):
                #print int(color_som.nodes[r,c,0]), int(color_som.nodes[r,c,1]), int(color_som.nodes[r,c,2])
                img.putpixel((c,r), (int(nodes[r,c,0]), int(nodes[r,c,1]), int(nodes[r,c,2])))
        img = img.resize((width*10, height*10),Image.NEAREST)
        img.save("sompy_test_colors.png")
    except:
        print "Error saving the image, do you have PIL (Python Imaging Library) installed?"
    
    #print "Saving 2D map in 2D.map..."
    #fname = "2D.map"
    #save_2D_dist_map(fname, nodes, width, height)
    #print len(nodes), len(nodes[0]), len(nodes[0,0])
    #print nodes[0]
    
pypar.finalize() 
