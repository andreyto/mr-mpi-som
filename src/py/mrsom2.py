#!/usr/bin/env python

import pypar
#import Numeric
import numpy
import sys

from numpy.random import shuffle
from random import *
from math import *

import struct
#fmt = 'L' ## unsigned long
fmt = 'Q' ## unsigned long long
fmtSize = struct.calcsize(fmt)

import os
MRSOM_ROOT = os.environ.get("MRSOM_ROOT")
        
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
  
## v3
def broadcast(x, root, myid, numProcs):
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
    
def simple_broadcast_v4(x, root, myid, numprocs, buffer):
    """ Broadcast implementation using bsend() w/ explicit definition of
    receive buffers.

    Input parameters:
     - x: data to broadcast.
     - root: rank of the process that initiates the broadcast.
     - myid: rank of the process calling the function.
     - buffer: well-dimensioned user-defined buffer for receiving 'x'.

    Return value: the broadcasted data.
    """
    list = range(0, root) + range(root + 1, numprocs)

    # The process of rank 'root' sends 'x' to all the other processes.
    if root == myid:
        for i in list:

            # Determine automatically the size of bsend's buffer.
            pypar.push_for_alloc(x, use_buffer=True)

            # Allocate and attach bsend's buffer.   
            pypar.alloc_and_attach()                            

            # Send data to process of rank 'i'.
            pypar.bsend(x, i, use_buffer=True)              

            # Detach and deallocate bsend's buffer.
            pypar.detach_and_dealloc()                          

        buffer = x

    # All processes with rank distinct from 'root' start receiving.
    else:
        buffer = pypar.receive(root, buffer)
        
    return buffer

#def simple_broadcast_v5(x, root, myid, numprocs, buffer):
    #""" Same as simple_broadcast_v4 except that it uses Pypar's
    #bypass mode.

    #The use of bypass mode implies that the programmer has to define his 
    #own buffers on the receiving side (same as when 'use_buffer' is True) 
    #and that he is limited to send numpy arrays and nothing else !

    #Hence, this function works only with numpy arrays.

    #Input parameters:
     #- x: data to broadcast.
     #- root: rank of the process that initiates the broadcast.
     #- myid: rank of the process calling the function.
     #- buffer: well-dimensioned user-defined buffer for receiving 'x'.

    #Return value: the broadcasted data.
    #"""
    #list = range(0, root) + range(root + 1, numprocs)

    ## The process of rank 'root' sends 'x' to all the other processes.
    #if root == myid:
        #for i in list:

            ## Determine automatically the size of bsend's buffer.
            #pypar.push_for_alloc(x, bypass=True)                        

            ## Allocate and attach bsend's buffer.   
            #pypar.alloc_and_attach()                            

            ## Send data to process of rank 'i'.
            #pypar.bsend(x, i, bypass=True)              

            ## Detach and deallocate bsend's buffer.
            #pypar.detach_and_dealloc()                          

        #buffer = x

    ## All processes with rank distinct from 'root' start receiving.
    #else:
        #buffer = pypar.receive(root, buffer, bypass=True)
        
    #return buffer


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
    
#def save_umat(fname, nodes, width, height):
    #D = 2
    #minDist = 1.5
    #f = open(fname, "w")
    #n = 0
    #for r in range(height):
        #for c in range(width):
            #dist = 0.0
            #nodesNum = 0
            #for r2 in range(height):
                #for c2 in range(width):
                    #for r3 in range(height):
                        #for c3 in range(width):
                            #if r2 == r3 and c2 == c3:
                                #continue
                            #tmp = eucl_dist(numpy.array((r2, c2)), numpy.array((r3, c3))) # coord dist
                            #if tmp <= minDist:
                                #nodesNum += 1
                                #dist += eucl_dist(nodes[r2, c2], nodes[r3, c3]) # weight dist
                    
            #dist /= nodesNum
            #f.write(str(floatFormat(dist)))
        #f.write("\n")
    #f.close()
    #return

## Return the byte offset of line n from index file fn.
def F(n, fn):    
    f = open(fn)
    try:
        f.seek(n * fmtSize)
        data = f.read(fmtSize)
    finally:
        f.close()

    return struct.unpack(fmt, data)[0]

## Return line n from data file using index file.
def getline(n, dataFile, indexFile):
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
    
#if __name__ == "__main__":
 
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
 
## BROADCASTING W/O SPECIFYING EXPLICIT RECIEVING BUFFER
nodes = broadcast(nodes, 0, MPI_myid, MPI_numproc)
## BROADCASTING WITH SPECIFYING EXPLICIT RECIEVING BUFFER
#recvBuffer = numpy.array([[[0 for i in range(FVDim)] for x in range(width)] for y in range(height)])
#recvBuffer = buffered_broadcast_bypass(nodes, 0, MPI_myid, MPI_numproc, recvBuffer)
#recvBuffer = simple_broadcast_v4(nodes, 0, MPI_myid, MPI_numproc, recvBuffer)
#print "v4 - rank " + str(MPI_myid) + " received: " + str(recvBuffer)
#nodes = recvBuffer
 
#if MPI_myid == 0:
    ##snd_tab = numpy.array([[[1, 2, 3, 4], [5, 6, 7, 8]],[[11, 12, 13, 14], [15, 16, 17, 18]]])
    ##snd_tab = numpy.array( [[[random() for i in range(4)] for x in range(2)] for y in range(2)] )
    #snd_tab = numpy.array( [random() for i in range(10)] )
#else:
    ##snd_tab = numpy.array([[[0, 0, 0, 0], [0, 0, 0, 0]],[[0, 0, 0, 0], [0, 0, 0, 0]]])
    ##snd_tab = numpy.array( [[[0 for i in range(4)] for x in range(2)] for y in range(2)] )
    #snd_tab = numpy.array( [0 for i in range(10)] )
    
#rcv_tab = numpy.array( [[[0 for i in range(4)] for x in range(2)] for y in range(2)] )
#rcv_tab = numpy.array( [0 for i in range(10)] )
#rcv_tab = simple_broadcast_v4(snd_tab, 0, MPI_myid, MPI_numproc, rcv_tab) 
#rcv_tab = broadcast(snd_tab, 0, MPI_myid, MPI_numproc) 
#print "v4 - rank " + str(MPI_myid) + " received: " + str(rcv_tab)

pypar.barrier()
#print "Rank " + str(MPI_myid) + " received: " + str(nodes)

## COMMON INFORMATION
inFileName = sys.argv[1]
timeConstant = iterations / log(radius)
radiusDecaying = 0
radDivVal = 0
learningRateDecaying = 0
## INIT WEIGHTS
deltaNodes = numpy.array([[[0 for i in range(FVDim)] for x in range(width)] for y in range(height)])

## NOTE: DO NOT NEED TO READ ALL FV. IMPROVE THIS!
## IMPROVE
# 1. Make index file for the inFile
## 2. DECIDE FROM-TO INDEX TO DIVIDE THE INFILE
## 3. READ A CHUNK OF THE INFILE 

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
print "### Rank %d reads %d rows (%d - %d) from %s." \
      % (MPI_myid, len(trainVector), myLower+1, myUpper, inFileName)
      
pypar.barrier()

## FILE READ (original)
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
    
    ## BROADCASTING W/O SPECIFYING EXPLICIT RECIEVING BUFFER
    nodes = broadcast(nodes, 0, MPI_myid, MPI_numproc)
    #print nodes
    ## BROADCASTING WITH SPECIFYING EXPLICIT RECIEVING BUFFER
    #recvBuffer.fill(0)
    #recvBuffer = buffered_broadcast(nodes, 0, MPI_myid, MPI_numproc, recvBuffer)
    #recvBuffer = simple_broadcast_v4(nodes, 0, MPI_myid, MPI_numproc, recvBuffer)
    #print recvBuffer
    #nodes = recvBuffer
    
    pypar.barrier()


if MPI_myid == 0:
    print "Saving..."
    
    ## SAVE WEIGHTS IN A FILE
    mapFileName = inFileName+".map"
    mapFile = open(mapFileName, "w")
    mapFile.write("%d %s %d %d\n" % (FVDim, "rect", width, height))
    for r in range(height):
        for c in range(width):
            #mapFile.write(nodes[r,c].tostring().)
            print >>mapFile, str(nodes[r,c]).replace('[',' ').replace(']', ' ').strip()
    mapFile.close()
    
    ## USING UMAT COMMAND IN SOM_PAK 3.1 TO SAVE CODEMAP IN EPS
    import subprocess
    cmd = MRSOM_ROOT + "/tools/umat -cin " + mapFileName + " > " + mapFileName + ".eps"
    proc = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,)
    #stdout_value = proc.communicate()[0]
    #s = stdout_value.split()
    #for i in range(len(s)):
        #print str(i) + " " + str(s[i])
    
    ## DEBUG (ONLY WORKS FOR FVDim=3 RGB DATA)
    try:
        from PIL import Image
        print "Saving (debug)..."
        debugImageFile = inFileName + ".png"
        img = Image.new("RGB", (width, height))
        for r in range(height):
            for c in range(width):
                #print int(color_som.nodes[r,c,0]), int(color_som.nodes[r,c,1]), int(color_som.nodes[r,c,2])
                img.putpixel((c,r), (int(nodes[r,c,0]), int(nodes[r,c,1]), int(nodes[r,c,2])))
        img = img.resize((width*10, height*10),Image.NEAREST)
        img.save(debugImageFile)
    except:
        print "Error saving the image, do you have PIL (Python Imaging Library) installed?"
    
    #print "Saving U-Mat..."
    #fname = inFileName+".map"
    #save_umat(fname, nodes, width, height)
    #print len(nodes), len(nodes[0]), len(nodes[0,0])
    #print nodes[0]
    print "Done!"
    
pypar.finalize() 

## EOF


