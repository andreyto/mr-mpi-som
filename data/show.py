import sys, glob
from scipy import *
from pylab import *

#plotfile = 'plot.png'

#if len(sys.argv) > 2:
    #plotfile = sys.argv[2]
#if len(sys.argv) > 1:
    #datafile = sys.argv[1]
#else:
    #print "No data file name given. Please enter"
    #datafile = raw_input("-> ")
#if len(sys.argv) <= 2:
    #print "No output file specified using default (plot.png)"
#if len(glob.glob(datafile))==0:
    #print "Data file %s not found. Exiting" % datafile
    #sys.exit()

#data=io.array_import.read_array(datafile)

#gplt.plot(data[:,0],data[:,1],'title "Weight vs. time" with points')
#gplt.xtitle('Time [h]')
#gplt.ytitle('Hydrogen release [wt. %]')
#gplt.grid("off")
#gplt.output(plotfile,'png medium transparent picsize 600 400')  

from numpy import *
from scipy.io import read_array

#R_matrix = read_array('rgb_map',columns=((1,-1)))
R_matrix = loadtxt('result.map')
    
    
    
#f=open('rgb_map.online','r')

#numbers = []
#for eachLine in f:
   #if <here test if the line is from the numbers block>:
      #line = [int(x) for x in eachLine.split(',')]
      #numbers.append(line)
      
      

#cdict = {'red': ((0.0, 0.0, 0.0),
                 #(0.5, 1.0, 0.7),
                 #(1.0, 1.0, 1.0)),
         #'green': ((0.0, 0.0, 0.0),
                   #(0.5, 1.0, 0.0),
                   #(1.0, 1.0, 1.0)),
         #'blue': ((0.0, 0.0, 0.0),
                  #(0.5, 1.0, 0.0),
                  #(1.0, 0.5, 1.0))}
#my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',R_matrix,256)
#pcolor(rand(10,10),cmap=my_cmap)
#colorbar()
imshow(R_matrix)
savefig('test')

#R_matrix2 = loadtxt('rgb_map2')
#imshow(R_matrix2)
#savefig('test2')
