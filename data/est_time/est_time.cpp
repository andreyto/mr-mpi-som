#include <stdio.h>
#include <iostream>
//using namespace std;

/// Variables
const int NRANK             = 1024;
const int NFEATURE          = 81920;
const int NDIM              = 256;
const int NFEATUREPERWORK   = 40;
const int PAGESZ            = 128;

/// Constants
const int NEPOCHS           = 10;
const int SOMX              = 50;
const int SOMY              = 50;
const int SOMD              = 2;
const int NMAP              = NFEATURE / NFEATUREPERWORK;
const int NKEY              = SOMX * SOMY; 
const int NEXCHANGE         = (SOMX*SOMY)-((SOMX*SOMY)/NMAP); 
//const int NEXCHANGE         = (SOMX*SOMY)-((SOMX*SOMY)/NMAP) / NRANK;  // consider parallel op.
const int NVALUES           = NRANK;
const int NCALL             = NMAP / NRANK;     // consider parallel op.

double TOTALT               = 0;                // total est. running time

/// Unit time
const double C              = 0.00000001;
const double T_alloc1byte   = C * 6;
const double T_dealloc1byte = C * 5;
const double T_randgen      = C * 10;
const double T_mmapfile     = C * 10;
const double T_multiply     = C * 3;
const double T_divide       = T_multiply;
const double T_add          = C * 1;
const double T_subtract     = T_add;
const double T_exp          = C * 6;
const double T_hashkey      = T_multiply * 10;

/// communication unit time
const double T_bcast1byte           = C * 10;
const double T_mpisend1byte         = C * 8;
const double T_mpiallreduce1byte    = C * 10;

void map()
{          
    /// alloc coords
    TOTALT += T_alloc1byte * 4 * 4;
    
    for (int nfeatureperwork = 0; nfeatureperwork < NFEATUREPERWORK; nfeatureperwork++) {
        
        /// get_bmu_coord  
        for (int somy = 0; somy < SOMY; somy++) {
            for (int somx = 0; somx < SOMX; somx++) {
                /// get_distance
                for (int ndim = 0; ndim <NDIM; ndim++) {
                    TOTALT += T_multiply * 7 + T_add * 7 + T_subtract * 2;
                    TOTALT += T_multiply * 2;
                }
            }
        }
       
        /// compute weight
        for (int somy = 0; somy < SOMY; somy++) {
            for (int somx = 0; somx < SOMX; somx++) {
                /// get_distance
                for (int ndim = 0; ndim <NDIM; ndim++) {
                    TOTALT += T_multiply * 1 + T_add * 1 + T_subtract * 2;
                    TOTALT += T_multiply * 2;
                }
                
                /// neighborhood funct
                TOTALT += T_multiply * 4 + 1 * T_divide * 1 + T_exp;
                
                /// compute numer and denom
                for (int ndim = 0; ndim <NDIM; ndim++) {
                    TOTALT += T_multiply * 5 + T_add * 4;
                }
                TOTALT += T_add * 1;
            }
        }
    }               
    TOTALT += T_dealloc1byte * 4 * 4;
               
    /// Add to KV
    TOTALT += T_alloc1byte * (NDIM + 1);
    for (int somy = 0; somy < SOMY; somy++) {
        for (int somx = 0; somx < SOMX; somx++) {
            TOTALT += T_multiply * 1 + T_add * 1;
        }
    }
    TOTALT += T_dealloc1byte * (NDIM + 1);
}

void collate()
{
    /// collate: aggregate + convert #######################################
        
    /// aggregate ----------------------------------------------------------
    TOTALT += T_alloc1byte * PAGESZ * 2;
    TOTALT += T_alloc1byte * PAGESZ * 1;
    TOTALT += T_alloc1byte * PAGESZ * 1;
    TOTALT += T_alloc1byte * PAGESZ * 1;
    TOTALT += T_mpiallreduce1byte * 1;
    
    //for (int ipage = 0; ipage < MAXPAGE; ipage++) {
        
        /// request page
        TOTALT += T_mpisend1byte * PAGESZ;
        
        for (int nkey = 0; nkey < NKEY; nkey++) {
            TOTALT += T_multiply * 4 + T_add * 4 + T_subtract * 1;
            TOTALT += T_hashkey * NKEY;
        }
        
        /// exchange KV among procs
        for (int nexchange = 0; nexchange < NEXCHANGE; nexchange++) {
            //TOTALT += T_mpisend1byte * 1 * NRANK; 
            TOTALT += T_mpisend1byte * 1; /// consider parallel op
            TOTALT += T_mpiallreduce1byte * 1;
        }
        
    //}
    
    TOTALT += T_dealloc1byte * PAGESZ * 2;
    TOTALT += T_dealloc1byte * PAGESZ * 1;
    TOTALT += T_dealloc1byte * PAGESZ * 1;
    TOTALT += T_dealloc1byte * PAGESZ * 1;
    
    /// kv->complete()
    for (int nexchange = 0; nexchange < NEXCHANGE; nexchange++) {
        TOTALT += T_add * 4;
        TOTALT += T_mpiallreduce1byte * 1;
    }
    TOTALT += T_mpiallreduce1byte * 1;       
    ///
    
    
    /// convert ------------------------------------------------------------
    /// kmv->convert(kv);
    TOTALT += T_alloc1byte * SOMX * SOMY; // KMV:partitions
    TOTALT += T_alloc1byte * SOMX * SOMY; // KMV:sets
    TOTALT += T_alloc1byte * 2;
    
    TOTALT += T_multiply * 4 + T_divide * 3 + T_add * 3; // include max and min
    
    TOTALT += T_dealloc1byte * 2;
    TOTALT += T_dealloc1byte * SOMX * SOMY; // KMV:partitions
    TOTALT += T_dealloc1byte * SOMX * SOMY; // KMV:sets
    
    /// kmv->complete();
    for (int nexchange = 0; nexchange < NEXCHANGE; nexchange++) {
        TOTALT += T_add * 6;
        TOTALT += T_mpiallreduce1byte * 1;
    }
    TOTALT += T_mpiallreduce1byte * 1;   
    ///
    
    TOTALT += T_mpiallreduce1byte * 1;
}

void reduce1()
{
    /// reduce #############################################################
    //for (int nmap = 0; nmap < NMAP; nmap++) {
        //for (int ipage = 0; ipage < MAXPAGE; ipage++) {
        
            /// request page
            TOTALT += T_mpisend1byte * PAGESZ;     
                       
            for (int nkey = 0; nkey < NKEY; nkey++) {
                TOTALT += T_multiply * 4 + T_add * 4 + T_subtract * 1;
                TOTALT += T_hashkey * NKEY;
            }
            
            /// mr_sum
            TOTALT += T_alloc1byte * (NDIM + 1);
            for (int ndim = 0; ndim <NDIM; ndim++) {
                for (int n = 0; n < NVALUES; n++) { 
                    /// numer += *((FLOAT_T*)multivalue + i + n*(NDIMEN+1));
                    TOTALT += T_multiply * 1 + T_add * 4;
                }
            }
            for (int n = 0; n < NVALUES; n++) { 
                TOTALT += T_multiply * 1 + T_add * 4;
            }            
            
            TOTALT += T_dealloc1byte * (NDIM + 1);
            
        //}
    //}
}

void gather()
{
    /// gather #############################################################
    TOTALT += T_mpisend1byte * SOMX * SOMY * NDIM * 4;
    
    /// kv->complete()
    for (int nexchange = 0; nexchange < NEXCHANGE; nexchange++) {
        TOTALT += T_add * 4;
        TOTALT += T_mpiallreduce1byte * 1;
    }
    TOTALT += T_mpiallreduce1byte * 1;   
}

void reduce2()
{
    /// reduce #############################################################
    /// mr_update_weight
    for (int ndim = 0; ndim <NDIM; ndim++) {
        TOTALT += T_divide * 1 + T_add * 1;
    }
}




int main(int argc, char **argv)
{
    /// alloc codebook, numer, denom
    TOTALT += T_alloc1byte * 4 * SOMX * SOMY * NDIM;
    TOTALT += T_alloc1byte * 4 * SOMX * SOMY * NDIM;
    TOTALT += T_alloc1byte * 4 * SOMX * SOMY;

    /// init codebook with rand num
    TOTALT += T_randgen * SOMX * SOMY * NDIM;

    /// mmap input file
    TOTALT += T_mmapfile;
        
    for (int iter = 0; iter < NEPOCHS; iter++) {
        
        printf("Epoch = %d\n", iter);
        TOTALT += 4 * T_multiply + 1 * T_divide;
        TOTALT += T_add;
        
        /// broadcast R
        TOTALT += T_bcast1byte;
        
        /// broadcast codebook
        TOTALT += T_bcast1byte * SOMX * SOMY * NDIM;
        TOTALT += T_divide;
        
        for (int ncall = 0; ncall < NCALL; ncall++) {
            map();  /// mr_train_batch     
        }          
        collate();   
        reduce1();  /// mr_sum             
        gather();
        reduce2();  /// mr_update_weight       
         
    } /// epochs

    std::cout << "Total time (min) = " << TOTALT/60 << std::endl;
    
    return 1;
}

// eof


