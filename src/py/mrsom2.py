import pypar, Numeric
import numpy
import sys
from random import *
from math import *
    
def work(myid, numprocs, data, inFileName, nodes, width, height, \
             radius_decaying, rad_div_val, learning_rate_decaying, delta_nodes):

    ## NOTE: DO NOT NEED TO READ ALL FV. IMPROVE THIS!
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
    
    interval = len(train_vector)
    myinterval = interval/numprocs    
    mylower = myid*myinterval
    
    if myid == numprocs-1:
        myupper = interval+1
    else:  
        myupper = mylower + myinterval
    
    mydata = train_vector[mylower:myupper]
    
    for j in range(len(mydata)):
        best = best_match(nodes, width, mydata[j])
        for loc in find_neighborhood(height, width, best, radius_decaying):
            influence = exp( (-1.0 * (loc[2]**2)) / rad_div_val)
            inf_lrd = influence * learning_rate_decaying
            delta_nodes[loc[0],loc[1]] += inf_lrd * (mydata[j]-nodes[loc[0],loc[1]])
        
    print "### Rank =", myid, "processed the data from", mylower, "to", myupper#, mydata, delta_nodes
    return  
  
  
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
    

## INIT 
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

## GET PARALLEL PARAMETERS
MPI_myid = pypar.rank()
MPI_numproc = pypar.size()
MPI_node = pypar.get_processor_name()
print "I am proc %d of %d on node %s" %(MPI_myid, MPI_numproc, MPI_node)

## SOM NODES INIT AND BROADCASTING
if MPI_myid == 0:
    nodes = numpy.array([[ [random() for i in range(FV_size)] for x in range(width)] for y in range(height)])
else:
    nodes = numpy.array([[[0 for i in range(FV_size)] for x in range(width)] for y in range(height)])
nodes = broadcast(nodes, 0, MPI_myid, MPI_numproc)     
pypar.barrier()
#print "Rank " + str(MPI_myid) + " received: " + str(nodes)

## COMMON INFORMATION
inFileName = sys.argv[1]   
time_constant = iterations/log(radius)        
radius_decaying = 0
rad_div_val = 0
learning_rate_decaying = 0
delta_nodes = numpy.array([[[0 for i in range(FV_size)] for x in range(width)] for y in range(height)])
            
for i in range(1, iterations+1):
    delta_nodes.fill(0)

    if MPI_myid == 0:
        radius_decaying = radius*exp(-1.0*i/time_constant)
        rad_div_val = 2 * radius_decaying * i
        learning_rate_decaying = learning_rate*exp(-1.0*i/time_constant)
        #sys.stdout.write("\rTraining Iteration: " + str(i) + "/" + str(iterations))
        print "################################# Training Iteration:", i
        print "### Radius_decaying: ", radius_decaying
    
    ## BROADCAST UPDATED INFO    
    radius_decaying = broadcast(radius_decaying, 0, MPI_myid, MPI_numproc) 
    rad_div_val = broadcast(rad_div_val, 0, MPI_myid, MPI_numproc) 
    learning_rate_decaying = broadcast(learning_rate_decaying, 0, MPI_myid, MPI_numproc)         
    #print "### Rank " + str(MPI_myid) + " radius_decaying: " + str(radius_decaying)
    #print "Rank " + str(MPI_myid) + " " + inFileName
    pypar.barrier()
            
    ## TRAINING IN PAPRALLEL FOR GETTING NEW WIEGHT VECTORS FOR SOME NODES
    work(MPI_myid, MPI_numproc, data, inFileName, nodes, width, height, \
             radius_decaying, rad_div_val, learning_rate_decaying, delta_nodes)   
    print "Proc %d finished working" % MPI_myid
    
    ## GATHERING SUB RESULTS    
    if MPI_numproc > 1:
        if MPI_myid == 0:
            nodes += delta_nodes ## ADD ROOT'S RESULT
            print "P%d updated its result" % (0)
            for id in range(1, MPI_numproc):
                print "P%d receving the result from P%d" % (0, id)
                #x = x + pypar.receive(id)  #Add up (would be more complex in general)
                nodes = nodes + pypar.receive(id)  
        else:
            #print "P%d sending  to P%d" % (MPI_myid, 0)  
            #pypar.send(x, 0)
            pypar.send(delta_nodes, 0)
            print "Proc %d after seding the result" % MPI_myid    
    #
    # Compute overall average and report
    #    
  
    #if MPI_myid == 0:  
        #avg = x/len(data)     
        #print "Global average is %.4f" % avg      
    
    nodes = broadcast(nodes, 0, MPI_myid, MPI_numproc)     
    pypar.barrier()
    
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


