////////////////////////////////////////////////////////////////////////////////
//
//  Parallelizing SOM on MR-MPI
//
//  Author: Seung-Jin Sul
//          (ssul@jcvi.org)
//
//  Revisions
//  v.0.0.9
//      6.1.2010        Start implementing serial SOM.
//      6.2.2010        MPI connected. CMake done!
//      6.3.2010        MR-MPU conencted as static library.
//      6.14.2010       serial online/batch SOM done!
//      6.15.2010       Batch SOM test done! Start mapreducing.
//      6.17.2010       Found the bottleneck in batch mode == get_bmu
//                      improved mr_train_batch is done! ==> train_batch_serial
//      6.23.2010       bcast and scatter are done! Now all tasks have
//                      the initial weight vectors and the feature
//                      vectors are evenly distributed.
//
//  v.1.0.0
//      6.24.2010       Initial version of MRSOM's done!! Promoted to v.1.0.0.
//
//  v.1.1.0
//      6.30.2010       Reimplement without classes.
//
//  v.2.0.0
//      07.01.2010      Incorporated DMatrix struct.
//                      Change the MR part with gather().
//  v.2.0.1
//                      Change command line arg. proceesing
//                      Add random feature vector generation
//
//  v.2.0.2
//      07.07.2010      Update CMakeLists.txt to have find_program for MPI
//                      and add .gitignore
//      07.08.2010      Add other distance metrics than euclidean.
//                      Add other normalization func
//
//  v.3.0.0
//      07.13.2010      DEBUG: Check R > 1.0 in MR_compute_weight!!!!!!!
//                      This solves the problem of resulting abnormal map
//                      when epochs > x.
//                      DEBUG: check if (temp_demon != 0) in MR_accumul_weight.
//                      Add input vector shuffling.
//
//  v.4.0.0
//      11.03.2010      Init running version done!
//
//      11.04.2010      Initial result came out. Found out-of-disk operations. 
//                      Try float compaction -> mrsom4.1.0.cpp
//
//      11.05.2010      Hydrid compaction v4.1.1. => v5
//      
//  v.5.0.0 => v.6.0.0
//      11.08.2010      To do 1. do not use split files for input vector. Use 
//                      memory-mapped file using Boost lib. Distribute indexes
//                      to workers. 2. Increase memsize from 64MB to 512MB. Make
//                      memsize configurable by command line argument. 3. Set
//                      out-of-core operation working folder to /tmp. 4. float 
//                      could be short. User typedef and double. 5. Consider
//                      minpage and maxpage settings.
//
//      11.09.2010      Boost memory mapped file is added.
//
//  v7
//      11.10.2010      key -> integer     
//      11.11.2010      key -> uint64_t (8bytes) for keyalign   
//
//  v8
//      11.15.2010      Update following AT's comments
//      11.16.2010      v8 done.
//
////////////////////////////////////////////////////////////////////////////////

/*
 *     This program is free software; you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation; either version 2 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with this program; if not, write to the Free Software
 *     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *     MA 02110-1301, USA.
 */

/*
 *                            PUBLIC DOMAIN NOTICE
 *               National Center for Biotechnology Information
 *
 *  This software/database is a "United States Government Work" under the
 *  terms of the United States Copyright Act.  It was written as part of
 *  the author's official duties as a United States Government employee and
 *  thus cannot be copyrighted.  This software/database is freely available
 *  to the public for use. The National Library of Medicine and the U.S.
 *  Government have not placed any restriction on its use or reproduction.
 *
 *  Although all reasonable efforts have been taken to ensure the accuracy
 *  and reliability of the software and data, the NLM and the U.S.
 *  Government do not and cannot warrant the performance or results that
 *  may be obtained by using this software or data. The NLM and the U.S.
 *  Government disclaim all warranties, express or implied, including
 *  warranties of performance, merchantability or fitness for any particular
 *  purpose.
 *
 *  Please cite the author in any work or product based on this material.
 */

/// MPI and MapReduce-MPI
#include "mpi.h"
///AT use gcc -Idirectory  and then #include "mrmpi/..."
#include "mrmpi/mapreduce.h"
#include "mrmpi/keyvalue.h"

#include <math.h>
#include <assert.h>
 
/// For save
#include <fstream>

/// For timing
#include <sys/time.h>

/// For Boost memory mapped file
#include <iterator>
#include <boost/iostreams/code_converter.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/filesystem/operations.hpp>          /// for real file size
boost::iostreams::mapped_file_source MMAPFILE;      /// Read-only Boost mmap file

#define FLOAT_T float
//#define FLOAT_T double
#define SZFLOAT sizeof(FLOAT_T)
#define MAXSTR 255
//#define KEY_T unsigned int

/// For Boost Multidimensional Array
#include <boost/multi_array.hpp>
typedef boost::multi_array<FLOAT_T, 3> ARRAY_3D_T;    /// 3D arrat
typedef boost::multi_array<FLOAT_T, 2> ARRAY_2D_T;    /// 2D arrat

/// For shared_ptr
//#include <boost/smart_ptr.hpp>

/// For CODEBOOK
ARRAY_3D_T CODEBOOK;

using namespace MAPREDUCE_NS;
using namespace std;

enum DISTTYPE       { EUCL, SOSD, TXCB, ANGL, MHLN };

/// GLOBALS
size_t NDIMEN = 0;            /// Num of dimensionality
size_t SOM_X = 50;            /// Width of SOM MAP 
size_t SOM_Y = 50;            /// Height of SOM MAP
size_t SOM_D = 2;             /// Dimension of SOM MAP, 2=2D
unsigned int NEPOCHS;         /// Iterations (=epochs)
unsigned int DISTOPT = EUCL;  /// Distance metric: 0=EUCL, 1=SOSD, 2=TXCB, 3=ANGL, 4=MHLN
uint64_t NVECS = 0;           /// Total num of feature vectors 
uint64_t NVECSPERRANK = 0;    /// Num of feature vectors per task
unsigned int SZPAGE = 64;     /// Page size (MB), default = 64MB
FLOAT_T *FDATA = NULL;        /// Feature data
unsigned int MYID;
FLOAT_T R = 0.0;

/// MR-MPI fuctions and related functions
void mr_train_batch(int itask, KeyValue *kv, void *ptr);
void mr_sum(char *key, int keybytes, char *multivalue, int nvalues, int *valuebytes, KeyValue *kv, void *ptr);
void mr_update_weight(uint64_t itask, char *key, int keybytes, char *value, int valuebytes, KeyValue *kv, void *ptr);

void get_bmu_coord(int* p, int itask, uint64_t n);
float get_distance(size_t som_y, size_t som_x, int itask, size_t row, unsigned int distance_metric);
 
/// Save U-matrix
int save_umat(char *fname);

/// To make result file name with date
string get_timedate(void);
float get_distance(const float *vec1, const float *vec2, unsigned int distance_metric);
float *get_wvec(size_t somy, size_t somx);

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{    
    if (argc == 7) {         
        /// syntax: mrsom FILE NEPOCHS NVECS NDIMEN NVECSPERRANK SZPAGE
        NEPOCHS = atoi(argv[2]);
        NVECS = atoi(argv[3]);
        NDIMEN = atoi(argv[4]);
        NVECSPERRANK = atoi(argv[5]);
        SZPAGE = atoi(argv[6]);
    }
    else if (argc == 9) {  
        /// syntax: mrsom FILE NEPOCHS NVECS NDIMEN NVECSPERRANK SZPAGE SOMX SOMY
        NEPOCHS = atoi(argv[2]);
        NVECS = atoi(argv[3]);
        NDIMEN = atoi(argv[4]);
        NVECSPERRANK = atoi(argv[5]);
        SZPAGE = atoi(argv[6]);
        SOM_X = atoi(argv[7]);
        SOM_Y = atoi(argv[8]);
    }
    else {
        printf("    mrsom FILE NEPOCHS NVECS NDIMEN NVECSPERRANK SZPAGE [X Y]\n\n");
        printf("    FILE        = master file.\n");
        printf("    NEPOCHS     = number of iterations.\n");
        printf("    NVECS       = number of feature vectors.\n");
        printf("    NDIMEN      = number of dimensionality of feature vector.\n");
        printf("    NVECSPERRANK = num vectors per task.\n");
        printf("    SZPAGE      = page size (MB).\n");
        printf("    [X Y]       = optional, SOM map size. Default = [50 50]\n");
        exit(0);
    }
    
    /// 
    /// Codebook
    ///
    CODEBOOK.resize(boost::extents[SOM_Y][SOM_X][NDIMEN]);

    ///
    /// Fill initial random weights
    ///
    srand((unsigned int)time(0));
    for (size_t som_y = 0; som_y < SOM_Y; som_y++) {        
        for (size_t som_x = 0; som_x < SOM_X; som_x++) {
            for (size_t d = 0; d < NDIMEN; d++) {
                int w = 0xFFF & rand();
                w -= 0x800;
                CODEBOOK[som_y][som_x][d] = (FLOAT_T)w / 4096.0f;
            }
        }
    }

    ///
    /// Creat memory-mapped file for feature vectors
    ///
    const std::string path(argv[1]);
    ///AT use boost class directly
    unsigned long int real_file_size_ = boost::filesystem::file_size(path);
    MMAPFILE.open(path, real_file_size_, 0);
    
    if (!MMAPFILE.is_open()) {
        cerr << "### ERROR: failed to create mmap file\n";
        exit(1);
    }
    
    ///
    /// MPI init
    ///
    MPI_Init(&argc, &argv);

    char MPI_procName[MAXSTR];
    int MPI_myId, MPI_nProcs, MPI_length;
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_myId);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_nProcs);
    MPI_Get_processor_name(MPI_procName, &MPI_length);
    fprintf(stdout, "### INFO: [Rank %d] %s \n", MPI_myId, MPI_procName);
    MPI_Barrier(MPI_COMM_WORLD); ///////////////////////////////////////
    
    /// 
    /// MR-MPI
    ///
    MapReduce *mr = new MapReduce(MPI_COMM_WORLD);
    /*
    * mapstyle = 0 (chunk) or 1 (stride) or 2 (master/slave)
    * all2all = 0 (irregular communication) or 1 (use MPI_Alltoallv)
    * verbosity = 0 (none) or 1 (summary) or 2 (histogrammed)
    * timer = 0 (none) or 1 (summary) or 2 (histogrammed)
    * memsize = N = number of Mbytes per page of memory
    * minpage = N = # of pages to pre-allocate per processor
    * maxpage = N = max # of pages allocatable per processor, 0 = no limit
    * keyalign = N = byte-alignment of keys
    * valuealign = N = byte-alignment of values
    * fpath = string 
    */
    mr->verbosity = 1;
    mr->timer = 0;
    mr->mapstyle = 2;       /// master/slave mode
    mr->memsize = SZPAGE;   /// page size
    //mr->minpage = 1;
    //mr->maxpage = 7;
    mr->keyalign = 8;       /// default: key type = uint_64t = 8 bytes
    //mr->valuealign = 8;     /// sizeof(float)*2 = 8 bytes
    //mr->fpath = "/tmp";     /// place to save out-of-core file(s)
    MPI_Barrier(MPI_COMM_WORLD);
    
    ///
    /// Parameters for SOM
    ///
    float N = (float)NEPOCHS;       /// iterations
    float nrule, nrule0 = 0.9f;     /// learning rate factor
    float R0;
    R0 = SOM_X / 2.0f;              /// init radius for updating neighbors
    R = R0;
    unsigned int x = 0;                      /// 0...N-1
    MYID = MPI_myId;
    
    ///AT gf.fdata = reinterpret_cast<float*>(MMAPFILE.data()); 
    /// or (float*)(MMAPFILE.data())
    //gf.fdata = reinterpret_cast<float*>((char*)MMAPFILE.data());
    FDATA = reinterpret_cast<float*>((char*)MMAPFILE.data());
        
    ///
    /// Training
    ///
    while (NEPOCHS && R > 1.0) {
        if (MPI_myId == 0) {
            R = R0 * exp(-10.0f * (x * x) / (N * N));
            x++;
            printf("### BATCH-  epoch: %d   R: %.2f \n", (NEPOCHS - 1), R);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        ///
        /// MPI_FLOAT / MPI_DOUBLE
        ///
        MPI_Bcast(&R, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast((void *)CODEBOOK.data(), SOM_Y*SOM_X*NDIMEN, MPI_FLOAT, 0, MPI_COMM_WORLD);          
        
        /// 
        /// 1. map: Each worker loads its feature vector and add 
        /// <(somy, somx, d), (numer, denom)> into KV
        /// 2. collate: collect same key
        /// 3. reduce: compute sum of numer and denom
        /// 4. map: compute new weight and upadte CODEBOOK
        ///
            
        /// v2-2. using Boost mmaped file
        uint64_t nmap = NVECS / NVECSPERRANK;
        ///AT is gf passed locally (not sent through MPI)?
        uint64_t nRes = mr->map(nmap, &mr_train_batch, NULL);
        //mr->kv->print(1, 2, 8, NDIMEN);   
        //mr->print(-1, 1, 2, 3);
        //cout << endl;
        mr->kv_stats(2);
        
        mr->collate(NULL);
        ////cout << "### collate DONE ###\n";
        //mr->print(-1, 1, 2, 3);          
        //cout << endl;
        ////mr->kmv_stats(2);
        
        nRes = mr->reduce(&mr_sum, NULL);
        ////cout << "### reduce, mr_sum DONE ###\n";
        //mr->kv->print(1, 2, 8, NDIMEN);  
        //cout << endl;
        ////mr->kv_stats(2);        
        
        mr->gather(1);
        ////cout << "### gather DONE ###\n";
        //mr->kv->print(-1, 2, 8, NDIMEN);    
        //cout << endl;
        mr->kv_stats(2);        
        
        nRes = mr->map(mr, &mr_update_weight, NULL);
        //cout << "### map, mr_update_weight DONE ###\n";
        //mr->print(-1, 1, 5, 5);            
        
        MPI_Barrier(MPI_COMM_WORLD); 

        NEPOCHS--;
    }  
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    ///
    /// Save SOM map and u-mat
    ///
    if (MPI_myId == 0) {
        printf("### Saving SOM map and U-Matrix...\n");
        string outFileName = "result-umat-" + get_timedate() + ".txt";
        cout << "    U-mat file = " << outFileName << endl; 
        int ret = save_umat((char*)outFileName.c_str());
        if (ret < 0) printf("    Failed (1) !\n");
        else {
            printf("    Converting SOM map to U-map...\n");
            printf("    Done (1) !\n");
        }
        
        ///
        /// Save SOM map for umat tool
        /// Usage: /tools/umat -cin result.map > result.eps
        ///
        string outFileName2 = "result-map-" + get_timedate() + ".txt";  
        cout << "    MAP file = " << outFileName2 << endl;       
        ofstream mapFile(outFileName2.c_str());
        printf("    Saving SOM map...\n");
        char temp[80];
        if (mapFile.is_open()) {
            mapFile << NDIMEN << " rect " << SOM_X << " " << SOM_Y << endl;
            for (size_t som_y = 0; som_y < SOM_Y; som_y++) { 
                for (size_t som_x = 0; som_x < SOM_X; som_x++) { 
                    for (size_t d = 0; d < NDIMEN; d++) {
                        sprintf(temp, "%0.10f", CODEBOOK[som_y][som_x][d]);
                        mapFile << temp << " ";
                    }
                    mapFile << endl;
                }
            }
            mapFile.close();
            printf("    Done (2) !\n");
        }
        else printf("    Failed (2) !\n");
        
        string cmd = "python ./show2.py " + outFileName;
        system((char*)cmd.c_str());
        cmd = "./umat -cin " + outFileName2 + " > test.eps";
        system((char*)cmd.c_str());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    MMAPFILE.close();
    delete mr;
    MPI_Finalize();

    
    
    return 0;
}
 
/** MR-MPI Map function - batch training2
 * @param itask - number of work items
 * @param kv
 * @param ptr
 */
 
void mr_train_batch(int itask, KeyValue *kv, void *ptr)
{    
    /// Debug
    //cout << "MyId, R = " << MYID << "," << R << endl;
    //for (size_t r = 0; r < NVECSPERRANK; r++) {
        //for (size_t d = 0; d < NDIMEN; d++) {
            //cout << *((FDATA + itask*NDIMEN*NVECSPERRANK) + r*NDIMEN + d) << " ";
        //}
        //cout << endl;
    //}
    //cout << endl;
    ///
    
    int *p1 = new int[SOM_D];
    int *p2 = new int[SOM_D];
    ARRAY_3D_T NUMER(boost::extents[SOM_Y][SOM_X][NDIMEN]);
    ARRAY_2D_T DENOM(boost::extents[SOM_Y][SOM_X]);  
    
    for (uint64_t n = 0; n < NVECSPERRANK; n++) {
        
        get_bmu_coord(p1, itask, n);
        //cout << "bmu = " << p1[0] << " " << p1[1] << endl;
        
        /// Accumulate denoms and numers
        for (size_t som_y = 0; som_y < SOM_Y; som_y++) { 
            for (size_t som_x = 0; som_x < SOM_X; som_x++) {
                p2[0] = som_x;
                p2[1] = som_y;
                float dist = 0.0f;
                for (size_t p = 0; p < SOM_D; p++)
                    dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
                dist = sqrt(dist);
                
                float neighbor_fuct = 0.0f;
                neighbor_fuct = exp(-(1.0f * dist * dist) / (R * R));
                
                ///AT denom is not a vector
                for (size_t w = 0; w < NDIMEN; w++) {
                    NUMER[som_y][som_x][w] += 
                        1.0f * neighbor_fuct * (*((FDATA + itask*NDIMEN*NVECSPERRANK) + n*NDIMEN + w));
                }
                DENOM[som_y][som_x] += neighbor_fuct;
            }
        }        
    }     
    delete [] p1;
    delete [] p2;
    p1 = NULL;
    p2 = NULL;
    
    ///
    /// Make FLOAT_T upate[NDIMEN+1] for NDIMEN numerator values one denominator
    /// value and add to KV
    ///
    FLOAT_T* update = new FLOAT_T[NDIMEN+1];    
    for (size_t som_y = 0; som_y < SOM_Y; som_y++) { 
        for (size_t som_x = 0; som_x < SOM_X; som_x++) {
            uint64_t iKey = som_y*SOM_X + som_x;

            size_t i = 0;
            for (; i < NDIMEN; i++) 
                update[i] = NUMER[som_y][som_x][i];
            update[i] = DENOM[som_y][som_x];
            
            kv->add((char*)&iKey, sizeof(uint64_t), (char*)update, (NDIMEN+1)*SZFLOAT);    
        }
    }
    delete [] update;
    update = NULL;
}
     

/** MR-MPI Map function - Get node coords for the best matching unit (BMU)
 * @param p
 * @param n
 */
 
void get_bmu_coord(int* coords, int itask, uint64_t n)
{
    ///AT coords should be int
    float mindist = 9999.99;
    float dist = 0.0f;
    
    ///
    /// Check SOM_X * SOM_Y nodes one by one and compute the distance 
    /// D(W_K, Fvec) and get the mindist and get the coords for the BMU.
    ///
    for (size_t som_y = 0; som_y < SOM_Y; som_y++) { 
        for (size_t som_x = 0; som_x < SOM_X; som_x++) {
            /// dist = get_distance(CODEBOOK[som_y][som_x], 
            /// (FDATA + itask*NDIMEN*NVECSPERRANK) + n*NDIMEN, DISTOPT);
            dist = get_distance(som_y, som_x, itask, n, DISTOPT);
            //cout << "dist = " << dist << endl;
            if (dist < mindist) { 
                mindist = dist;
                coords[0] = som_x;
                coords[1] = som_y;
            }
        }
    }
}


/** MR-MPI Map function - Distance b/w a feature vector and a weight vector
 * = Euclidean
 * @param som_y
 * @param som_x
 * @param distance_metric: 
 */

//AT why vect1 is not const?
float get_distance(size_t som_y, size_t som_x, int itask, size_t r, unsigned int distance_metric)
{
    float distance = 0.0f;
    //float n1 = 0.0f, n2 = 0.0f;
    switch (distance_metric) {
    default:
    case 0: /// EUCLIDIAN
        for (size_t w = 0; w < NDIMEN; w++)
            //distance += (vec1[w] - vec2[w]) * (vec1[w] - vec2[w]);
            distance += (CODEBOOK[som_y][som_x][w] - *((FDATA + itask*NDIMEN*NVECSPERRANK) + r*NDIMEN + w))
                        *
                        (CODEBOOK[som_y][som_x][w] - *((FDATA + itask*NDIMEN*NVECSPERRANK) + r*NDIMEN + w));
                    
        return sqrt(distance);
    //case 1: /// SOSD: //SUM OF SQUARED DISTANCES
        ////if (m_weights_number >= 4) {
        ////distance = mse(vec, m_weights, m_weights_number);
        ////} else {
        //for (unsigned int w = 0; w < NDIMEN; w++)
            //distance += (vec[w] - wvec[w]) * (vec[w] - wvec[w]);
        ////}
        //return distance;
    //case 2: /// TXCB: //TAXICAB
        //for (unsigned int w = 0; w < NDIMEN; w++)
            //distance += fabs(vec[w] - wvec[w]);
        //return distance;
    //case 3: /// ANGL: //ANGLE BETWEEN VECTORS
        //for (unsigned int w = 0; w < NDIMEN; w++) {
            //distance += vec[w] * wvec[w];
            //n1 += vec[w] * vec[w];
            //n2 += wvec[w] * wvec[w];
        //}
        //return acos(distance / (sqrt(n1) * sqrt(n2)));
    //case 4: /// MHLN:   //mahalanobis
        ////distance = sqrt(m_weights * cov * vec)
        ////return distance
    }
}


/** MR-MPI Map function - Distance b/w vec1 and vec2, default distance_metric
 * = Euclidean
 * @param vec1
 * @param vec2
 * @param distance_metric: 
 */
 
float get_distance(const float *vec1, const float *vec2, unsigned int distance_metric)
{
    float distance = 0.0f;
    float n1 = 0.0f, n2 = 0.0f;
    switch (distance_metric) {
    default:
    case 0: /// EUCLIDIAN
        for (unsigned int w = 0; w < NDIMEN; w++)
            distance += (vec1[w] - vec2[w]) * (vec1[w] - vec2[w]);
        return sqrt(distance);
    //case 1: /// SOSD: //SUM OF SQUARED DISTANCES
        ////if (m_weights_number >= 4) {
        ////distance = mse(vec, m_weights, m_weights_number);
        ////} else {
        //for (unsigned int w = 0; w < NDIMEN; w++)
            //distance += (vec[w] - wvec[w]) * (vec[w] - wvec[w]);
        ////}
        //return distance;
    //case 2: /// TXCB: //TAXICAB
        //for (unsigned int w = 0; w < NDIMEN; w++)
            //distance += fabs(vec[w] - wvec[w]);
        //return distance;
    //case 3: /// ANGL: //ANGLE BETWEEN VECTORS
        //for (unsigned int w = 0; w < NDIMEN; w++) {
            //distance += vec[w] * wvec[w];
            //n1 += vec[w] * vec[w];
            //n2 += wvec[w] * wvec[w];
        //}
        //return acos(distance / (sqrt(n1) * sqrt(n2)));
    //case 4: /// MHLN:   //mahalanobis
        ////distance = sqrt(m_weights * cov * vec)
        ////return distance
    }
}

/** User-defined Reduce function - Sum numer and denom
 * (Qid,DBid) key into Qid for further aggregating.
 * @param key
 * @param keybytes
 * @param multivalue: collected blast result strings.  
 * @param nvalues
 * @param valuebytes
 * @param kv
 * @param ptr
 */

///AT cast to float
void mr_sum(char *key, int keybytes, char *multivalue, int nvalues, int *valuebytes, 
            KeyValue *kv, void *ptr)
{   
    FLOAT_T* newUpdate = new FLOAT_T[NDIMEN+1];
    size_t i = 0;
    for (; i < NDIMEN; i++) {
        FLOAT_T numer = 0.0;
        for (size_t n = 0; n < nvalues; n++) { 
            //cout << *((FLOAT_T*)multivalue + i + n*(NDIMEN+1)) << endl;                   
            numer += *((FLOAT_T*)multivalue + i + n*(NDIMEN+1));            
        }
        newUpdate[i] = numer;
    }
    FLOAT_T denom = 0.0;
    for (size_t n = 0; n < nvalues; n++)
        denom += *((FLOAT_T*)multivalue + NDIMEN + n*(NDIMEN+1));
    newUpdate[i] = denom;
      
    /// Debug
    //for (size_t i = 0; i < NDIMEN+1; i++) {
        //cout << *((FLOAT_T*)newUpdate + i) << " ";
    //}
    //cout << endl;
    ///    
    
    kv->add(key, sizeof(uint64_t), (char*)newUpdate, (NDIMEN+1)*SZFLOAT);           
    delete [] newUpdate;
    newUpdate = NULL;
}
 

/** Update CODEBOOK numer and denom
 * (Qid,DBid) key into Qid for further aggregating.
 * @param itask
 * @param key
 * @param keybytes
 * @param value
 * @param valuebytes
 * @param kv
 * @param ptr
 */
 
void mr_update_weight(uint64_t itask, char *key, int keybytes, char *value,
                      int valuebytes, KeyValue *kv, void *ptr)
{
    uint64_t iKey = *((uint64_t *)key);
    unsigned int row = iKey/SOM_X;
    unsigned int col = iKey%SOM_X;
    
    /// Debug
    //for (size_t i = 0; i < NDIMEN+1; i++) {
        //cout << *((FLOAT_T*)value + i) << " ";
    //}
    //cout << endl;
    ///
    
    FLOAT_T finalDenom = *((FLOAT_T*)value + NDIMEN);
    
    if (finalDenom != 0) {
        for (size_t i = 0; i < NDIMEN; i++) {
            FLOAT_T aNumer = *((FLOAT_T*)value + i);
            FLOAT_T newWeight = aNumer / finalDenom;
            /// Should check newWeight > 0.0
            if (newWeight > 0.0) 
                CODEBOOK[row][col][i] = newWeight;                           
        }    
    }
}

 
 
/** MR-MPI Map function - Get weight vector from CODEBOOK using x, y index
 * @param som_y - y coordinate of a node in the map 
 * @param som_x - x coordinate of a node in the map 
 * @param fvec
 */

float *get_wvec(size_t som_y, size_t som_x)
{
    //AT do not create temporaries, return pointers instead
    FLOAT_T *wvec = (FLOAT_T *)malloc(SZFLOAT * NDIMEN);
    for (size_t d = 0; d < NDIMEN; d++)
        wvec[d] = CODEBOOK[som_y][som_x][d]; /// CAUTION: (y,x) order
    
    return wvec;
}

/** Save u-mat
 * @param CODEBOOK
 * @param fname
 */
 
int save_umat(char *fname)
{
    int D = 2;
    float min_dist = 1.5f;
    FILE *fp = fopen(fname, "wt");
    if (fp != 0) {
        unsigned int n = 0;
        for (size_t som_y1 = 0; som_y1 < SOM_Y; som_y1++) {
            for (size_t som_x1 = 0; som_x1 < SOM_X; som_x1++) {
                float dist = 0.0f;
                unsigned int nodes_number = 0;
                 int coords1[2];
                coords1[0] = som_x1;
                coords1[1] = som_y1;               
                
                for (size_t som_y2 = 0; som_y2 < SOM_Y; som_y2++) {   
                    for (size_t som_x2 = 0; som_x2 < SOM_X; som_x2++) {
                        unsigned int coords2[2];
                        coords2[0] = som_x2;
                        coords2[1] = som_y2;    

                        if (som_x1 == som_x2 && som_y1 == som_y2) continue;
                            
                        float tmp = 0.0;
                        for (size_t d = 0; d < D; d++) {
                            tmp += pow(coords1[d] - coords2[d], 2.0f);                            
                        }
                        tmp = sqrt(tmp);
                        if (tmp <= min_dist) {
                            nodes_number++;
                            FLOAT_T *vec1 = get_wvec(som_y1, som_x1);
                            FLOAT_T *vec2 = get_wvec(som_y2, som_x2);
                            dist += get_distance(vec1, vec2, DISTOPT);
                            free(vec1);
                            free(vec2);
                        }
                    }
                }
                dist /= (float)nodes_number;
                fprintf(fp, " %f", dist);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        return 0;
    }
    else
        return -2;
}
 
 
/** Utility - get time and date string for making file name
 * @param void
 */
 
string get_timedate(void)
{
   time_t now;
   char the_date[MAXSTR];
   the_date[0] = '\0';
   now = time(NULL);

   if (now != -1) {
      strftime(the_date, MAXSTR, "%R_%d_%m_%Y", gmtime(&now));
   }

   return std::string(the_date);
}
