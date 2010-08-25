#from __future__ import division

from random import *
from math import *
import sys
#import scipy
import time
try:
    import numpy
except:
    raise Exception, 'Module numpy must be present to run pypar'
    
try:
    import pypar
except:
    raise 'Could not find module pypar'
print 'Modules numpy, pypar imported OK'


def master():
    print '[MASTER]: I am processor %d of %d on node %s' % (MPI_myid, MPI_numproc, MPI_node)
    
    return
    
    
def slave():
    print '[SLAVE %d]: I am processor %d of %d on node %s' % (MPI_myid, MPI_myid, MPI_numproc, MPI_node)
    
    return
    
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

def simple_broadcast_v3(x, root, myid, numprocs):
    """ Alternative implementation to 'simple_broadcast_v1()'.

    Make all bsend calls using a single buffer and a
    single call to 'mpi_alloc_and_attach()' and 'mpi_detach_and_dealloc()'.

    This implementation illustrates how to use 'push_for_alloc()' in order
    to allocate a single buffer, big enough, to handle all sends within
    one call to 'mpi_alloc_and_attach()' and 'mpi_detach_and_dealloc()'.

    Input parameters:
     - x: data to broadcast.
     - root: rank of the process that initiates the broadcast.
     - myid: rank of the process calling the function.

    Return value: the broadcasted data.
    """

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
                    
#class SOM:

    #def __init__(self, height=10, width=10, FV_size=10, learning_rate=0.005):
        #self.height = height
        #self.width = width
        #self.FV_size = FV_size
        #self.radius = (height+width)/3
        #self.learning_rate = learning_rate
        #self.nodes = numpy.array([[ [random() for i in range(FV_size)] for x in range(width)] for y in range(height)])

    ## train_vector: [ FV0, FV1, FV2, ...] -> [ [...], [...], [...], ...]
    ## train vector may be a list, will be converted to a list of scipy arrays
    #def train(self, iterations=1000, train_vector=[[]]):
        #for t in range(len(train_vector)):
            #train_vector[t] = numpy.array(train_vector[t])
        #time_constant = iterations/log(self.radius)
        #delta_nodes = numpy.array([[[0 for i in range(self.FV_size)] for x in range(self.width)] for y in range(self.height)])
        
        #for i in range(1, iterations+1):
            #delta_nodes.fill(0)
            #radius_decaying= self.radius*exp(-1.0*i/time_constant)
            #rad_div_val = 2 * radius_decaying * i
            #learning_rate_decaying = self.learning_rate*exp(-1.0*i/time_constant)
            #sys.stdout.write("\rTraining Iteration: " + str(i) + "/" + str(iterations))
            
            #for j in range(len(train_vector)):
                #best = self.best_match(train_vector[j])
                #for loc in self.find_neighborhood(best, radius_decaying):
                    #influence = exp( (-1.0 * (loc[2]**2)) / rad_div_val)
                    #inf_lrd = influence*learning_rate_decaying
                    #delta_nodes[loc[0],loc[1]] += inf_lrd*(train_vector[j]-self.nodes[loc[0],loc[1]])
            
            ### BATCH
            #self.nodes += delta_nodes
        #sys.stdout.write("\n")
    
    ## Returns a list of points which live within 'dist' of 'pt'
    ## Uses the Chessboard distance
    ## pt is (row, column)
    #def find_neighborhood(self, pt, dist):
        #min_y = max(int(pt[0] - dist), 0)
        #max_y = min(int(pt[0] + dist), self.height)
        #min_x = max(int(pt[1] - dist), 0)
        #max_x = min(int(pt[1] + dist), self.width)
        #neighbors = []
        #for y in range(min_y, max_y):
            #for x in range(min_x, max_x):
                #dist = abs(y-pt[0]) + abs(x-pt[1])
                #neighbors.append((y,x,dist))
        #return neighbors
    
    ## Returns location of best match, uses Euclidean distance
    ## target_FV is a scipy array
    #def best_match(self, target_FV):
        #loc = numpy.argmin((((self.nodes - target_FV)**2).sum(axis=2))**0.5)
        #r = 0
        #while loc > self.width:
            #loc -= self.width
            #r += 1
        #c = loc
        #return (r, c)

    ## returns the Euclidean distance between two Feature Vectors
    ## FV_1, FV_2 are scipy arrays
    #def FV_distance(self, FV_1, FV_2):
        #return (sum((FV_1 - FV_2)**2))**0.5

if __name__ == "__main__":
    
    datetime = time.ctime(time.time())
    MPI_myid = pypar.rank()
    MPI_numproc = pypar.size()
    MPI_node = pypar.get_processor_name()
    print "I am proc %d of %d on node %s" %(MPI_myid, MPI_numproc, MPI_node)
    
    if len(sys.argv) != 4:
        print "Usage: dataFile numIter dim"
        pypar.finalize()
        sys.exit(1)
    
    print "Initialization..."
    #color_som = SOM(width, height, FV_size, lrate)
    width = 50
    height = 50
    FV_size = int(sys.argv[3])
    lrate = 0.05
    iterations = int(sys.argv[2])
    radius = (height+width)/3
    learning_rate = 0.05
    
    ####################################################################
    if MPI_myid == 0:
        nodes = numpy.array([[ [random() for i in range(FV_size)] for x in range(width)] for y in range(height)])
    else:
        nodes = numpy.array([[[0 for i in range(FV_size)] for x in range(width)] for y in range(height)])
    
    #rcv_tab = numpy.array([[[0 for i in range(FV_size)] for x in range(width)] for y in range(height)])
    #nodes = simple_broadcast_v4(snd_tab, 0, MPI_myid, MPI_numproc, rcv_tab) 
    #rcv_tab = simple_broadcast_v4(nodes, 0, MPI_myid, MPI_numproc, rcv_tab) 
    #print "v4 - rank " + str(MPI_myid) + " received: " + str(rcv_tab)
    
    nodes = simple_broadcast_v3(nodes, 0, MPI_myid, MPI_numproc) 
    #print "v3 - rank " + str(MPI_myid) + " received: " + str(nodes)
       
    pypar.barrier()
    
    ####################################################################    
    inFileName = sys.argv[1]   
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
    #print data       
    
    ####################################################################    
    #for t in range(len(train_vector)):
        #train_vector[t] = numpy.array(train_vector[t])
    time_constant = iterations/log(radius)        
    delta_nodes = numpy.array([[[0 for i in range(FV_size)] for x in range(width)] for y in range(height)])
    radius_decaying = 0
    rad_div_val = 0
    learning_rate_decaying = 0
    
    print "Training..."
    for i in range(1, iterations+1):
        delta_nodes.fill(0)
        
        if MPI_myid == 0:
            radius_decaying = radius*exp(-1.0*i/time_constant)
            rad_div_val = 2 * radius_decaying * i
            learning_rate_decaying = learning_rate*exp(-1.0*i/time_constant)
            sys.stdout.write("\rTraining Iteration: " + str(i) + "/" + str(iterations))
        
        radius_decaying = simple_broadcast_v3(radius_decaying, 0, MPI_myid, MPI_numproc) 
        print "v3 - rank " + str(MPI_myid) + " received: " + str(radius_decaying)
        radius_decaying = simple_broadcast_v3(radius_decaying, 0, MPI_myid, MPI_numproc) 
        rad_div_val = simple_broadcast_v3(rad_div_val, 0, MPI_myid, MPI_numproc) 
        learning_rate_decaying = simple_broadcast_v3(learning_rate_decaying, 0, MPI_myid, MPI_numproc)         
        pypar.barrier()

        for j in range(len(train_vector)):
            best = best_match(nodes, width, train_vector[j])
            for loc in find_neighborhood(height, width, best, radius_decaying):
                influence = exp( (-1.0 * (loc[2]**2)) / rad_div_val)
                inf_lrd = influence*learning_rate_decaying
                delta_nodes[loc[0],loc[1]] += inf_lrd*(train_vector[j]-nodes[loc[0],loc[1]])
                
        nodes += delta_nodes ## BATCH UPDATE
    sys.stdout.write("\n")    
    
    #data = [ [0, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255], [255, 0, 0], [255, 0, 255], [255, 255, 0], [255, 255, 255]]
    #print "Training..."
    #color_som.train(numIter, data)
    #print color_som.nodes*(255.)
    #color_som.nodes = color_som.nodes*(255.)
    
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
    

        
        
    pypar.finalize()
    print 'MPI environment finalized.'
        
