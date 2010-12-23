from pylab import *
#import matplotlib.pyplot as plt
#plt.plot([1,2,3,4], [1,4,9,16], 'ro')

#x = [1,2,3,4,5,6]
x = [32, 64, 128, 256, 512, 1024]
#y2 = [123.71, 62.28, 33.9, 19.4, 13.14, 11.18]
#y1 = [129.31, 61.37, 32.9, 17.24, 11.23, 6.44]
y3 = [0, 67.24, 32.45, 0, 0, 5.51]
#est = [200.1, 100.698, 50.67, 25.645, 13.136, 6.89]
est = [183, 92.272, 46.47, 23.57, 12.127, 6.4]
#y2 = [587, 318, 192, 154, 155, 156]
#y3 = [1700, 828, 423, 229, 168, 159]
#y4 = [0,0,0,0,285,270]
#y5 = [4856,1648,725,386,214,160]

y1 = [122.45, 61.57, 30.9, 15.55, 8.1, 4.17]
y2 = [123.63, 61.8, 30.85, 15.57, 7.87, 4.02]

figure(1)
ax=subplot(111)

#ax.plot(x1700, y, 'ro-', linewidth=1.5)
#ax.set_xscale('log', basex=2, label_minor=True)
#ax.set_yscale('log', basex=10, label_minor=True)


#ax.loglog(x, est, 'md-', linewidth=1.5, basex=2, basey=10, label='est')
#ax.loglog(x, y5, 'rd-', linewidth=1.5, basex=2, basey=10, label='v5')
ax.loglog(x, y1, 'gs-.', linewidth=1.5, basex=2, basey=10, label='v2')
ax.loglog(x, y2, 'ro-', linewidth=1.5, basex=2, basey=10, label='v1')
#ax.loglog(x, y3, 'ko-.', linewidth=1.5, basex=2, basey=10, label='v3')


#ax.set_ylim(1e2, 1e4)
ax.set_xlim(16, 2048)

#ax.set_xticks(x, minor=True)
#ax.set_yticks(y, minor=True)
#ax.set_xticklabels(('','$2^5$\n(32)','6\n(64)','7\n(128)','8\n(256)','9\n(512)','10\n(1024)',''))
#ax.set_xticklabels(('', '$2^5$','$2^6$','$2^7$','$2^8$','$2^9$','$2^{10}$', ''))
ax.set_xticklabels(('', '32','64','128','256','512','1024', ''))
#ax.yaxis.grid(True, linestyle='-.', which='major')
#ax.xaxis.grid(True, linestyle='-.', which='major')
ax.yaxis.grid(True, linestyle='-.', which='minor')
ax.xaxis.grid(True, linestyle='-.', which='minor')
#ax.yaxis.grid(True, linestyle='-', which='major')
#ax.xaxis.grid(True, linestyle='-', which='major')

ax.set_xlabel('Number of cores ($log_2$)', fontsize=17)
ax.set_ylabel('Running time in minutes ($log_{10}$)', fontsize=17)
    
fontsize=16
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
fontsize=16
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)

#ax.xaxis.set_major_locator(AutoLocator())
#ax.yaxis.set_major_locator(AutoLocator())
#x_major = ax.xaxis.get_majorticklocs()
#dx_minor =  (x_major[-1]-x_major[0])/(len(x_major)-1) /2.
#ax.xaxis.set_minor_locator(MultipleLocator(dx_minor)) 
#y_major = ax.yaxis.get_majorticklocs()
#dy_minor =  (y_major[-1]-y_major[0])/(len(y_major)-1) /5.
#ax.yaxis.set_minor_locator(MultipleLocator(dy_minor)) 

#ax.set_title('Scalability result with 400bp x 1000 queries x 12 files', fontsize=17)

for i in range(0,len(x)):
    ax.annotate(y1[i], xy=(x[i], y1[i]),  xycoords='data',
                xytext=(-2, -20), textcoords='offset points', fontsize=10, color='r')
                ##,arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2")
                ##)
                
for i in range(0,len(x)):
    ax.annotate(y2[i], xy=(x[i], y2[i]),  xycoords='data',
                xytext=(2, 5), textcoords='offset points', fontsize=10, color='g')
                #,arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2")
                #)
                
#for i in range(0,len(x)):
    #ax.annotate(y5[i], xy=(x[i], y5[i]),  xycoords='data',
                #xytext=(2, 2), textcoords='offset points', fontsize=10, color='r')
                ##,arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2")
                ##)


ax.legend( ('1,024 work items', 
            '2,048 work items'
            #'4,096 work items'
            #'1000x12 query files, sort in workers', 
            #'1000x12 query files, sort in master'            
            ) )
show()
