from pylab import *

## Constants
NEPOCHS         = 10
SOMX            = 50
SOMY            = 50
SOMD            = 2
#NFEATURE        = 81920
NFEATURE        = 256
NFEATUREPERWORK = 40
NMAP            = NFEATURE / NFEATUREPERWORK
#NRANK           = 1024
NRANK           = 4
NDIM            = 256
TOTALT          = 0

## Unit time
T_alloc1byte    = 0.01
T_dealloc1byte  = T_alloc1byte
T_bcast1byte    = 0.1
T_mpisend1byte  = 0.02
T_randgen       = 0.01
T_mmapfile      = 1
T_multiply      = 0.001
T_divide        = T_multiply
T_add           = 0.0001
T_subtract      = T_add

# alloc codebook, numer, denom
TOTALT += T_alloc1byte * 4 * SOMX * SOMY * NDIM
TOTALT += T_alloc1byte * 4 * SOMX * SOMY * NDIM
TOTALT += T_alloc1byte * 4 * SOMX * SOMY

# init codebook with rand num
TOTALT += T_randgen * SOMX * SOMY * NDIM

# mmap input file
TOTALT += T_mmapfile
    
    
for iter in range(0, NEPOCHS):
    print "Iter = ", iter
    TOTALT += 4 * T_multiply + 1 * T_divide
    TOTALT += T_add
    
    # broadcast R
    TOTALT += T_bcast1byte
    # broadcast codebook
    TOTALT += T_bcast1byte * SOMX * SOMY * NDIM

    TOTALT += T_divide
    
    ## map #####################################################################
    
    for nmap in range(0, NMAP):
        
        TOTALT += T_alloc1byte * 4 * 4

        for nfeatureperwork in range(0, NFEATUREPERWORK):
            
            ## get_bmu_coord  
            for somy in range(0, SOMY):
                for somx in range(0, SOMX):
                    ## get_distance
                    for ndim in range(0, NDIM):
                        TOTALT += T_multiply * 7 + T_add * 6 + T_subtract * 2
                        TOTALT += T_multiply * 2
            
            ## compute weight
            for somy in range(0, SOMY):
                for somx in range(0, SOMX):
                    ## get_distance
                    for ndim in range(0, SOMD):
                        TOTALT += T_multiply * 1 + T_add * 1 + T_subtract * 2
                        TOTALT += T_multiply * 2
                    
                    # neighborhood funct
                    TOTALT += 4 * T_multiply + 1 * T_divide
                    
                    # compute numer and denom
                    for ndim in range(0, NDIM):
                        TOTALT += T_multiply * 5 + T_add * 4
                    TOTALT += T_add
                    
        TOTALT += T_dealloc1byte * 4 * 4
                   
        ## Add to KV
        TOTALT += T_alloc1byte * (NDIM + 1)
        for somy in range(0, SOMY):
            for somx in range(0, SOMX):         
                TOTALT += T_multiply * 1 + T_add * 1
        TOTALT += T_dealloc1byte * (NDIM + 1)
        
    ## map #####################################################################
                
    ## collate: aggregate & convert


    ## reduce


    ## gather


    ## reduce


print "Total time = ", TOTALT
