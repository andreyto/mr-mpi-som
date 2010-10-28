#!/usr/bin/env python

from __future__ import division

## Kyle Dickerson
## kyle.dickerson@gmail.com
## Jan 15, 2008
##
## Self-organizing map using scipy
## This code is licensed and released under the GNU GPL

## This code uses a square grid rather than hexagonal grid, as scipy allows for fast square grid computation.
## I designed sompy for speed, so attempting to read the code may not be very intuitive.
## If you're trying to learn how SOMs work, I would suggest starting with Paras Chopras SOMPython code:
##  http://www.paraschopra.com/sourcecode/SOM/index.php
## It has a more intuitive structure for those unfamiliar with scipy, however it is much slower.

## If you do use this code for something, please let me know, I'd like to know if has been useful to anyone.

from random import *
from math import *
import sys
#import scipy
import numpy
 

class SOM:

    def __init__(self, height=10, width=10, FV_size=10, learning_rate=0.005):
        self.height = height
        self.width = width
        self.FV_size = FV_size
        self.radius = (height+width)/3
        self.learning_rate = learning_rate
        self.nodes = numpy.array([[ [random() for i in range(FV_size)] for x in range(width)] for y in range(height)])

    # train_vector: [ FV0, FV1, FV2, ...] -> [ [...], [...], [...], ...]
    # train vector may be a list, will be converted to a list of scipy arrays
    def train(self, iterations=1000, train_vector=[[]]):
        for t in range(len(train_vector)):
            train_vector[t] = numpy.array(train_vector[t])
        time_constant = iterations/log(self.radius)
        delta_nodes = numpy.array([[[0 for i in range(self.FV_size)] for x in range(self.width)] for y in range(self.height)])
        
        for i in range(1, iterations+1):
            delta_nodes.fill(0)
            radius_decaying= self.radius*exp(-1.0*i/time_constant)
            rad_div_val = 2 * radius_decaying * i
            learning_rate_decaying = self.learning_rate*exp(-1.0*i/time_constant)
            sys.stdout.write("\rTraining Iteration: " + str(i) + "/" + str(iterations))
            
            for j in range(len(train_vector)):
                best = self.best_match(train_vector[j])
                for loc in self.find_neighborhood(best, radius_decaying):
                    influence = exp( (-1.0 * (loc[2]**2)) / rad_div_val)
                    inf_lrd = influence*learning_rate_decaying
                    delta_nodes[loc[0],loc[1]] += inf_lrd*(train_vector[j]-self.nodes[loc[0],loc[1]])
            
            ## BATCH
            self.nodes += delta_nodes
        sys.stdout.write("\n")
    
    # Returns a list of points which live within 'dist' of 'pt'
    # Uses the Chessboard distance
    # pt is (row, column)
    def find_neighborhood(self, pt, dist):
        min_y = max(int(pt[0] - dist), 0)
        max_y = min(int(pt[0] + dist), self.height)
        min_x = max(int(pt[1] - dist), 0)
        max_x = min(int(pt[1] + dist), self.width)
        neighbors = []
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                dist = abs(y-pt[0]) + abs(x-pt[1])
                neighbors.append((y,x,dist))
        return neighbors
    
    # Returns location of best match, uses Euclidean distance
    # target_FV is a scipy array
    def best_match(self, target_FV):
        loc = numpy.argmin((((self.nodes - target_FV)**2).sum(axis=2))**0.5)
        r = 0
        while loc > self.width:
            loc -= self.width
            r += 1
        c = loc
        return (r, c)

    # returns the Euclidean distance between two Feature Vectors
    # FV_1, FV_2 are scipy arrays
    def FV_distance(self, FV_1, FV_2):
        return (sum((FV_1 - FV_2)**2))**0.5

if __name__ == "__main__":
    
    if len(sys.argv) != 4:
        print "Usage: dataFile numIter dim"
        sys.exit(1)
        
    inFileName = sys.argv[1]
    f = open(inFileName, "r")
    data = []
    count = 0
    dt = numpy.dtype(float) 
    try:
        for line in f:
            values = line.rstrip().split(' ')
            data.append(numpy.array(values, dt))
    finally:
        f.close()
    print data
   
    width = 50
    height = 50
    FVDim = int(sys.argv[3])
    lrate = 0.05
    numIter = int(sys.argv[2])
    
    print "Initialization..."
    color_som = SOM(width, height, FVDim, lrate)
    
    #data = [ [0, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255], [255, 0, 0], [255, 0, 255], [255, 255, 0], [255, 255, 255]]
    print "Training..."
    color_som.train(numIter, data)
    #print color_som.nodes*(255.)
    #color_som.nodes = color_som.nodes*(255.)
    
    try:
        from PIL import Image
        print "Saving Image: sompy_test_colors.png..."
        img = Image.new("RGB", (width, height))
        for r in range(height):
            for c in range(width):
                #print int(color_som.nodes[r,c,0]), int(color_som.nodes[r,c,1]), int(color_som.nodes[r,c,2])
                img.putpixel((c,r), (int(color_som.nodes[r,c,0]), int(color_som.nodes[r,c,1]), int(color_som.nodes[r,c,2])))
        img = img.resize((width*10, height*10),Image.NEAREST)
        img.save("sompy_test_colors.png")
    except:
        print "Error saving the image, do you have PIL (Python Imaging Library) installed?"
        
        
        
        
        
