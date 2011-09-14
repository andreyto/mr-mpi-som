//### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
//#
//#   See COPYING file distributed along with the MGTAXA package for the
//#   copyright and license terms.
//#
//### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

#ifndef MRSOM2_HPP
#define MRSOM2_HPP

/// MPI and MapReduce-MPI
#include "mpi.h"
#include "mrmpi/mapreduce.h"
#include "mrmpi/keyvalue.h"

#include <math.h>
#include <limits>
#include <stdint.h>

/// For save
#include <fstream>

/// Processing command line arguments
#include <boost/program_options.hpp>
namespace po = boost::program_options;

/// For Boost memory mapped file
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/filesystem/operations.hpp>         /// for real file size
boost::iostreams::mapped_file_source MMAPBINFILE;  /// Read-only Boost mmap file for input bin file
boost::iostreams::mapped_file_source MMAPIDXFILE;  /// Read-only Boost mmap file fr input index file (sparse)

#define FLOAT_T float
//#define FLOAT_T double
#define SZFLOAT sizeof(FLOAT_T)
#define MAXSTR 255

/// For syncronized timing
#ifndef MPI_WTIME_IS_GLOBAL
#define MPI_WTIME_IS_GLOBAL 1
#endif

/// For Boost Multidimensional Array
#include <boost/multi_array.hpp>
typedef boost::multi_array<FLOAT_T, 3> ARRAY_3D_T;    /// 3D array
typedef boost::multi_array<FLOAT_T, 1> ARRAY_1D_T;    /// 1D array

/// Configuration file processing
#include <boost/program_options/detail/config_file.hpp>
namespace pod = boost::program_options::detail;

/// For CODEBOOK
ARRAY_3D_T CODEBOOK;
ARRAY_1D_T NUMER1;
ARRAY_1D_T DENOM1;
ARRAY_1D_T NUMER2;
ARRAY_1D_T DENOM2;

using namespace MAPREDUCE_NS;
using namespace std;

enum DISTTYPE   { EUCL, SOSD, TXCB, ANGL, MHLN };   /// distance metrics
enum RUNMODE    { TRAIN, TEST };                    /// running mode

/// GLOBALS
int SZPAGE = 64;                /// Page size (MB), default = 64MB
int g_RANKID;
size_t SOM_X;                   /// Width of SOM MAP
size_t SOM_Y;                   /// Height of SOM MAP
size_t SOM_D;                   /// Dimension of SOM MAP, 2=2D
string OUTPREFIX;               /// output file name prefix
uint32_t NDIMEN = 0;            /// Num of dimensionality
uint32_t NVECS = 0;             /// Total num of feature vectors
uint32_t NVECSPERRANK = 0;      /// Num of feature vectors per task
uint32_t NVALSSPERRANK = 0;     /// Num of values per task (to split work item for sparse matrix)
uint32_t NVECSLEFT = 0;         /// Num of feature vectors for the last task (lefty)
uint32_t NBLOCKS = 0;           /// Total num of blocks divided
unsigned int NEPOCHS;           /// Iterations (=epochs)
unsigned int DISTOPT = EUCL;    /// Distance metric: 0=EUCL, 1=SOSD, 2=TXCB, 3=ANGL, 4=MHLN
unsigned int RUNMODE = TRAIN;   /// run mode: tain or test

FLOAT_T* FDATA = NULL;          /// Feature data
FLOAT_T R = 0.0;                /// SOM Map Radius

/// Sparse structures and routines
int bSPARSE = 0;                /// sparse matric or not
typedef struct item {
    uint32_t index;             /// column number index
    float value;                /// non-zero value in input matrix
} SPARSE_STRUCT_T;

typedef struct indextype {
    uint32_t position;          /// start pos of each row from tellp()
    uint32_t num_values;        /// num cols
    uint32_t num_values_accum;  /// num cols accumulated, this value is actually used to access a row in *.bin
} INDEX_STRUCT_T;

typedef struct sparseworkitem_row { 
    uint32_t start;             /// start row num for each work item 
    uint32_t end;               /// end row num for each work item
} SPARSEWORKITEM_STRUCT_T;
vector<SPARSEWORKITEM_STRUCT_T> g_vecSparseWorkItem;

SPARSE_STRUCT_T* FDATASPARSE = NULL; 
INDEX_STRUCT_T*  INDEXSPARSE = NULL; 

/// MR-MPI fuctions and related functions
void    mpireduce_train_batch(int itask, KeyValue* kv, void* ptr);
void    get_bmu_coord(int* p, int itask, uint32_t n);
float   get_distance(size_t y, size_t x, int itask, size_t row, unsigned int distance_metric);
float   get_distance(size_t y, size_t x, const FLOAT_T* vec, unsigned int distance_metric);
float   get_distance(const FLOAT_T* vec1, const FLOAT_T* vec2, unsigned int distance_metric);
float*  get_wvec(size_t y, size_t x);

/// I/O functions
int     init_codebook(unsigned int seed);
int     load_codebook(const char *mapFilename);
int     save_codebook(const char* cbFileName);
int     save_umat(const char* fname);
int     read_matrix(const char *binfilename, const char *indexilename);

/// Classification
void    test(const char* codebook, const char* binFileName);
void    classify(const FLOAT_T* vec, int* p);

/// For sparse
void    mpireduce_train_batch_sparse(int itask, KeyValue* kv, void* ptr);


#endif
