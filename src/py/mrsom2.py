#from random import *
#from math import *
#import sys
##import scipy
#import time
 
    
def work(myid, numprocs, data, inFileName):
    
    import Numeric
    
    # Identify local slice and process it
    #
    interval = len(data)
    myinterval = interval/numprocs    
    mylower = myid*myinterval
    
    if myid == numprocs-1:
        myupper = interval+1
    else:  
        myupper = mylower + myinterval
    
    mydata = data[mylower:myupper]
    
    # Computation (average)
    #
    myavg = float(Numeric.sum(mydata))/len(mydata)
    print "P%d: %s Local avg=%.4f" % (myid, str(mydata), myavg)
    
    f = open(inFileName, "r")
    train_vector = []
    count = 0
    dt = numpy.dtype(float) 
    try:
        for line in f:
            values = line.rstrip().split(' ')
            train_vector.append(numpy.array(values, dt))
    finally:
        f.close()
        
        
    return myavg*len(mydata)
  
  
def broadcast(x, root, myid, numprocs,):
    list = range(0, root) + range(root + 1, numprocs)

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
  
###################################################
# Main program - communication takes place here
#
import pypar, Numeric
import numpy
import sys
from random import *
from math import *

# Get data. Here it is just generated but it could be read 
# from file or given as an input parameter.
# 
if len(sys.argv) != 4:
    print "Usage: dataFile numIter dim"
    sys.exit(1)
print "Initialization..."
width = 50
height = 50
FV_size = int(sys.argv[3])
lrate = 0.05
iterations = int(sys.argv[2])
radius = (height+width)/3
learning_rate = 0.05
lower = 100
upper = 121
data = Numeric.array(range(lower,upper))

#
# Get parallel parameters
#
MPI_myid = pypar.rank()
MPI_numproc = pypar.size()
MPI_node = pypar.get_processor_name()
print "I am proc %d of %d on node %s" %(MPI_myid, MPI_numproc, MPI_node)


if MPI_myid == 0:
    nodes = numpy.array([[ [random() for i in range(FV_size)] for x in range(width)] for y in range(height)])
else:
    nodes = numpy.array([[[0 for i in range(FV_size)] for x in range(width)] for y in range(height)])
nodes = broadcast(nodes, 0, MPI_myid, MPI_numproc)     
pypar.barrier()
print "Rank " + str(MPI_myid) + " received: " + str(nodes)


inFileName = sys.argv[1]   

time_constant = iterations/log(radius)        
radius_decaying = 0
rad_div_val = 0
learning_rate_decaying = 0
delta_nodes = numpy.array([[[0 for i in range(FV_size)] for x in range(width)] for y in range(height)])
            
for i in range(1, iterations+1):
    #
    # Do work in parallel
    #
    x = work(MPI_myid, MPI_numproc, data, inFileName)   #Do work on all processors
    print "Proc %d finished working" % MPI_myid

    #  
    # Communication
    #
    if MPI_numproc > 1:
        #
        # Processor 0 gathers all results and merge them
        #
        if MPI_myid == 0:
            for id in range(1, MPI_numproc):
                print "P%d receving from P%d" % (0, id)
                x = x + pypar.receive(id)  #Add up (would be more complex in general)
      
        #
        # All other processors send their results back to processor 0
        #  
        else:
            print "P%d sending to P%d" % (MPI_myid, 0)  
            pypar.send(x, 0)

    print "Proc %d after communication" % MPI_myid    
    #
    # Compute overall average and report
    #    
  
    if MPI_myid == 0:  
        avg = x/len(data)     
        print "Global average is %.4f" % avg      
    
    nodes = broadcast(nodes, 0, MPI_myid, MPI_numproc)     
    pypar.barrier()

pypar.finalize()    


