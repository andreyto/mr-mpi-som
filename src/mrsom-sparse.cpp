////////////////////////////////////////////////////////////////////////////////
//
//  Batch SOM on MR-MPI with a row-first sparse data structure
//
//  Author: Peter Wittek (peterwittek@acm.org)
//
//  Based on the code by Seung-Jin Sul (ssul@jcvi.org)
//
//  Last updated: 08.26.2011
//
////////////////////////////////////////////////////////////////////////////////


/// MPI and MapReduce-MPI
#include "mpi.h"
#include "mrmpi/mapreduce.h"
#include "mrmpi/keyvalue.h"

#include <string>
#include <sstream>
#include <iostream>

#include <math.h>

/// For save
#include <fstream>

/// Processing command line arguments
#include <boost/program_options.hpp>
namespace po = boost::program_options;

/// Configuration file processing
#include <boost/program_options/detail/config_file.hpp>
namespace pod = boost::program_options::detail;

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
typedef boost::multi_array<FLOAT_T, 2> ARRAY_2D_T;    /// 2D array
typedef boost::multi_array<FLOAT_T, 1> ARRAY_1D_T;    /// 1D array

/// For CODEBOOK
ARRAY_3D_T CODEBOOK;
ARRAY_1D_T NUMER1;
ARRAY_1D_T DENOM1;
ARRAY_1D_T NUMER2;
ARRAY_1D_T DENOM2;
int *rbuf;

int init_codebook(unsigned int seed);
int load_codebook(const char *mapFilename);
int save_codebook(char* cbFileName);

using namespace MAPREDUCE_NS;
using namespace std;

/// GLOBALS
size_t SOM_X;                 /// Width of SOM MAP
size_t SOM_Y;                 /// Height of SOM MAP
size_t SOM_D;                 /// Dimension of SOM MAP, 2=2D
size_t NDIMEN = 0;            /// Num of dimensionality
uint64_t NVECS = 0;           /// Total num of feature vectors
uint64_t NVECSPERRANK = 0;    /// Num of feature vectors per task
int SAVE_INTERIM=0;			  /// Whether to save interim states

enum TRAINORTEST    { TRAIN, TEST };                    /// running mode
unsigned int TRAINORTEST = TRAIN;

FLOAT_T R = 0.0;
int SZPAGE = 64;              /// Page size (MB), default = 64MB
unsigned int NEPOCHS;         /// Iterations (=epochs)
string OUTPREFIX;

/// MR-MPI fuctions and related functions
void    mpireduce_train_batch(int itask, KeyValue* kv, void* ptr);
void    get_bmu_coord(int* p, int itask, uint64_t n);
float   get_distance(size_t som_y, size_t som_x, int itask, size_t row);
float   get_distance(const FLOAT_T* vec1, const FLOAT_T* vec2);
float*  get_wvec(size_t somy, size_t somx);
int     get_best_matching_instance(int som_x, int somy);

/// Save U-matrix
int     save_umat(char* fname);

/// Sparse structures and routines
struct svm_node
{
    int index;
    float value;
};
struct svm_node *x_space;
struct svm_node **x_matrix;

int read_matrix(const char *filename, bool with_classes);

/// Classification
void    classify(const svm_node* vec, int* p);
float   get_distance(size_t somy, size_t somx, const svm_node* vec);


/* -------------------------------------------------------------------------- */
int main(int argc, char** argv)
/* -------------------------------------------------------------------------- */
{

    ///
    /// Read conf file, mrblast.ini and set parameters
    ///
    ifstream config("mrsom.ini");
    if (!config) {
        cerr << "ERROR: configuration file, mrsom.ini, not found" << endl;
        return 1;
    }

    /// parameters
    set<string> options;
    map<string, string> parameters;
    options.insert("*");

    try {
        for (pod::config_file_iterator i(config, options), e ; i != e; ++i) {
            parameters[i->string_key] = i->value[0];
        }

        try {
            SOM_X = boost::lexical_cast<size_t>(parameters["SOMX"]);
            SOM_Y = boost::lexical_cast<size_t>(parameters["SOMY"]);
            SOM_D = boost::lexical_cast<size_t>(parameters["SOMD"]);
        }
        catch(const boost::bad_lexical_cast &) {
            cerr << "Exception: bad_lexical_cast" << endl;
        }
    }
    catch(exception& e) {
        cerr<< "Exception: " << e.what() << endl;
    }

    po::options_description generalDesc("General options");
    generalDesc.add_options()
    ("help", "print help message")
    ("mode,m", po::value<string>(),
     "set train/test mode, \"train or test\"")
    ;

    po::options_description trainnigDesc("Options for training");
    trainnigDesc.add_options()
    ("infile,i", po::value<string>(),
     "set input train feature vector file name")
    ("outfile,o", po::value<string>(&OUTPREFIX)->default_value("result"),
     "set a prefix for outout file name")
    ("nepochs,e", po::value<unsigned int>(), "set the number of iterations")
    ("page-size,p", po::value<int>(&SZPAGE)->default_value(64),
     "[OPTIONAL] set page size of MR-MPI (default=64MB)")
    ("save-interim,s", po::value<int>(&SAVE_INTERIM)->default_value(0),
     "[OPTIONAL] save interim states (default=0)")

    ;

    po::options_description testingDesc("Options for testing");
    testingDesc.add_options()
    ("codebook,c", po::value<string>(),
     "set saved codebook file name")
    //("infile,i", po::value<string>(),
    //"set input feature vector file name for testing")
    //("outfile,o", po::value<string>(&OUTPREFIX)->default_value("result"),
    //"set a prefix for outout file name")
    //("nvecs,n", po::value<unsigned int>(),
    //"set the number of feature vectors")
    //("ndim,d", po::value<unsigned int>(),
    //"set the number of dimension of input feature vector")
    ;

    po::options_description allDesc("Allowed options");
    allDesc.add(generalDesc).add(trainnigDesc).add(testingDesc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, allDesc), vm);
    po::notify(vm);

    string inFileName, somMapFileName;
    string ex = "Examples\n";
    ex += "  Training: mpirun -np 4 mrsom-sparse -m train ";
    ex += "-i rgbs.svm -o rgbs -e 10 \n";
    ex += "  Testing: mrsom-sparse -m test -c rgbs-codebook.txt -i rgbs.svm ";
    ex += "-o rgbs\n\n";

    if (argc < 2 || (!strcmp(argv[1], "-?") || !strcmp(argv[1], "--?")
                     || !strcmp(argv[1], "/?") || !strcmp(argv[1], "/h")
                     || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--h")
                     || !strcmp(argv[1], "--help") || !strcmp(argv[1], "/help")
                     || !strcmp(argv[1], "-help")  || !strcmp(argv[1], "help") )) {
        cout << "MR-MPI Batch SOM for sparse data\n"
             << "\nAuthor: Peter Wittek (peterwittek@acm.org)\n\n"
             << allDesc << "\n" << ex;
        return 1;
    }
    else {
        if (vm.count("mode")) {
            string trainOrTest = vm["mode"].as<string>();
            if (trainOrTest.compare("train")) TRAINORTEST = TEST;
            else if (trainOrTest.compare("test")) TRAINORTEST = TRAIN;
            else {
                cerr << "ERROR: the mode should be 'train' or 'test'\n";
                return 1;
            }
            if (vm.count("infile")) inFileName = vm["infile"].as<string>();
            else {
                cout << allDesc << "\n" << ex;
                return 1;
            }
            if (vm.count("outfile")) OUTPREFIX = vm["outfile"].as<string>();
            if (vm.count("page-size")) SZPAGE = vm["page-size"].as<int>();
            if (vm.count("save-interim")) SAVE_INTERIM = vm["save-interim"].as<int>();

            if (TRAINORTEST == TRAIN) {

                if (vm.count("nepochs"))
                    NEPOCHS = vm["nepochs"].as<unsigned int>();
                else {
                    cout << allDesc << "\n" << ex;
                    return 1;
                }
            }
            else {
                if (vm.count("codebook"))
                    somMapFileName = vm["codebook"].as<string>();
                else {
                    cout << allDesc << "\n" << ex;
                    return 1;
                }
            }
        }
    }

    ///
    /// Open sparse data file
    ///
    read_matrix(inFileName.c_str(), false);

    ///
    /// Codebook
    ///
    CODEBOOK.resize(boost::extents[SOM_Y][SOM_X][NDIMEN]);
    NUMER1.resize(boost::extents[SOM_Y*SOM_X*NDIMEN]);
    DENOM1.resize(boost::extents[SOM_Y*SOM_X]);
    NUMER2.resize(boost::extents[SOM_Y*SOM_X*NDIMEN]);
    DENOM2.resize(boost::extents[SOM_Y*SOM_X]);

    ///
    /// TESTING MODE
    ///
    if (TRAINORTEST == TEST) {
        ///
        /// Load codebook
        ///
        if (load_codebook(somMapFileName.c_str())>0) {
            exit(0);
        }

        ///
        /// Classification: get the coords of the trained SOM MAP for new
        /// vectors for testing.
        ///
        string classFileName = OUTPREFIX + "-class.txt";
        FILE* classOutFile = fopen(classFileName.c_str(), "w");
        int p[SOM_D];
        if (classOutFile) {
            for (size_t nvec = 0; nvec < NVECS; nvec++) {
                /////////////////
                classify(x_matrix[nvec], p);
                /////////////////
                fprintf(classOutFile, "%d\t%d\n", p[0], p[1]); /// somx,somy
            }
        }
        else {
            cerr << "ERROR: file open error.\n";
            exit(0);
        }
        fclose(classOutFile);

        return 0;
    }


    init_codebook((unsigned int)time(0));
    //init_codebook(0);

    ///
    /// MPI init
    ///
    MPI_Init(&argc, &argv);

    char MPI_procName[MAXSTR];
    int MPI_myId, MPI_nProcs, MPI_length;
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_myId);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_nProcs);
    MPI_Get_processor_name(MPI_procName, &MPI_length);

    NVECSPERRANK = ceil(NVECS / (1.0*MPI_nProcs));

    MPI_Barrier(MPI_COMM_WORLD);
    double profile_time = MPI_Wtime();

    ///
    /// MR-MPI
    ///
    MapReduce* mr = new MapReduce(MPI_COMM_WORLD);
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
    mr->verbosity = 0;
    mr->timer = 0;
    mr->mapstyle = 0;       /// chunk. NOTE: MPI_reduce() does not work with
    /// master/slave mode
    mr->memsize = SZPAGE;   /// page size
    mr->keyalign = 8;       /// default: key type = uint_64t = 8 bytes

    MPI_Barrier(MPI_COMM_WORLD);

    ///
    /// Parameters for SOM
    ///
    float N = (float)NEPOCHS;       /// iterations
    float nrule, nrule0 = 0.9f;     /// learning rate factor
    float R0;
    R0 = SOM_X / 2.0f;              /// init radius for updating neighbors
    R = R0;
    unsigned int x = 0;             /// 0...N-1

    if (MPI_myId == 0) {
        rbuf = new int[3*NVECSPERRANK*MPI_nProcs];
    }
    ///
    /// Training
    ///
    while (NEPOCHS && R > 1.0) {

        if (MPI_myId == 0) {
            R = R0 * exp(-10.0f * (x * x) / (N * N));
            x++;
            printf("BATCH-  epoch: %d   R: %.2f \n", (NEPOCHS - 1), R);
        }

        MPI_Bcast(&R, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);


        if (SZFLOAT == 4)
            MPI_Bcast((void*)CODEBOOK.data(), SOM_Y*SOM_X*NDIMEN, MPI_FLOAT,
                      0, MPI_COMM_WORLD);
        else if (SZFLOAT == 8)
            MPI_Bcast((void*)CODEBOOK.data(), SOM_Y*SOM_X*NDIMEN, MPI_DOUBLE,
                      0, MPI_COMM_WORLD);

        /// v9 using MPI_reduce
        ///
        /// 1. Each task fills NUMER1 and DENOM1
        /// 2. MPI_reduce sums up each tasks NUMER1 and DENOM1 to the root's
        ///    NUMER2 and DENOM2.
        /// 3. Update CODEBOOK using NUMER2 and DENOM2
        if (MPI_myId == 0) {
            for (size_t som_y = 0; som_y < SOM_Y; som_y++) {
                for (size_t som_x = 0; som_x < SOM_X; som_x++) {
                    DENOM2[som_y*SOM_X + som_x] = 0.0;
                    for (size_t d = 0; d < NDIMEN; d++) {
                        NUMER2[som_y*SOM_X*NDIMEN + som_x*NDIMEN + d] = 0.0;
                    }
                }
            }
        }

        uint64_t nRes = mr->map(MPI_nProcs, &mpireduce_train_batch, NULL);

        if (MPI_myId == 0) {
            for (size_t som_y = 0; som_y < SOM_Y; som_y++) {
                for (size_t som_x = 0; som_x < SOM_X; som_x++) {
                    FLOAT_T denom = DENOM2[som_y*SOM_X + som_x];
                    for (size_t d = 0; d < NDIMEN; d++) {
                        FLOAT_T newWeight = NUMER2[som_y*SOM_X*NDIMEN
                                                   + som_x*NDIMEN + d] / denom;
                        if (newWeight > 0.0)
                            CODEBOOK[som_y][som_x][d] = newWeight;

                    }
                }
            }
        }
        ///

        if (MPI_myId == 0 && SAVE_INTERIM!=0) {
            printf("INFO: Saving interim U-Matrix...\n");
            char umatInterimFileName[50];
            sprintf(umatInterimFileName, "%s-umat-%03d.txt", OUTPREFIX.c_str(),  x);
            int ret = save_umat(umatInterimFileName);
            char codebookInterimFileName[50];
            sprintf(codebookInterimFileName, "%s-codebook-%03d.txt", OUTPREFIX.c_str(),  x);
            save_codebook(codebookInterimFileName);
            char bmuInterimFileName[50];
            sprintf(bmuInterimFileName, "%s-bmu-%03d.txt", OUTPREFIX.c_str(),  x);
            ofstream bmuFile(bmuInterimFileName);
            if (bmuFile.is_open()) {
                for (int i=0; i<NVECS; i++) {
                    bmuFile << rbuf[3*i] << " " << rbuf[3*i+1] << " " << rbuf[3*i+2] << endl;
                }
            }
        }
        NEPOCHS--;

    }

    MPI_Barrier(MPI_COMM_WORLD);

    ///
    /// Save SOM map and u-mat
    ///
    if (MPI_myId == 0) {
        ///
        /// Save U-mat
        ///
        printf("INFO: Saving SOM map and U-Matrix...\n");
        string umatFileName = OUTPREFIX + "-umat.txt";
        cout << "    Saving U-mat file = " << umatFileName << endl;
        int ret = save_umat((char*)umatFileName.c_str());
        if (ret < 0)
            printf("    Failed to save u-matrix. !\n");
        string cbFileName = OUTPREFIX + "-codebook.txt";
        save_codebook((char *)cbFileName.c_str());
    }
    MPI_Barrier(MPI_COMM_WORLD);


    delete mr;

    profile_time = MPI_Wtime() - profile_time;
    if (MPI_myId == 0) {
        cerr << "Total Execution Time: " << profile_time << endl;
    }

    MPI_Finalize();

    delete(x_matrix);
    delete(x_space);
    delete(rbuf);

    return 0;
}



/** MR-MPI user-defined map function - batch training with MPI_reduce()
 * @param itask - number of work items
 * @param kv
 * @param ptr
 */

void mpireduce_train_batch(int itask,
                           KeyValue* kv,
                           void* ptr)
{
    int p1[SOM_D];
    int p2[SOM_D];

    /// v2
    for (size_t som_y = 0; som_y < SOM_Y; som_y++) {
        for (size_t som_x = 0; som_x < SOM_X; som_x++) {
            DENOM1[som_y*SOM_X + som_x] = 0.0;
            for (size_t d = 0; d < NDIMEN; d++)
                NUMER1[som_y*SOM_X*NDIMEN + som_x*NDIMEN + d] = 0.0;
        }
    }
    int bmu_list[3*NVECSPERRANK];
    int bmu_index=0;
    for (uint64_t n = 0; n < NVECSPERRANK; n++) {
        if (itask*NVECSPERRANK+n<NVECS) {
            /// get the best matching unit
            get_bmu_coord(p1, itask, n);
            bmu_list[bmu_index++]=p1[0];
            bmu_list[bmu_index++]=p1[1];
            bmu_list[bmu_index++]=itask*NVECSPERRANK+n;
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
                    size_t offset = 0;
                    while((x_matrix[itask*NVECSPERRANK+n]+offset)->index != -1) {
                        NUMER1[som_y*SOM_X*NDIMEN + som_x*NDIMEN +
                               (x_matrix[itask*NVECSPERRANK+n]+offset)->index ] +=
                                   1.0f * neighbor_fuct
                                   * (x_matrix[itask*NVECSPERRANK+n]+offset)->value;
                        ++offset;
                    }

                    DENOM1[som_y*SOM_X + som_x] += neighbor_fuct;
                }
            }
        }
    }

    if (SZFLOAT == 4) { /// 4 bytes float
        MPI_Reduce((void*)NUMER1.data(), (void*)NUMER2.data(),
                   SOM_Y*SOM_X*NDIMEN, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce((void*)DENOM1.data(), (void*)DENOM2.data(),
                   SOM_Y*SOM_X, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else if (SZFLOAT == 8) { /// 8 bytes double
        MPI_Reduce((void*)NUMER1.data(), (void*)NUMER2.data(),
                   SOM_Y*SOM_X*NDIMEN, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce((void*)DENOM1.data(), (void*)DENOM2.data(),
                   SOM_Y*SOM_X, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    MPI_Gather( bmu_list, 3*NVECSPERRANK, MPI_INT, rbuf, 3*NVECSPERRANK, MPI_INT, 0, MPI_COMM_WORLD);
}


/** MR-MPI Map function - Get node coords for the best matching unit (BMU)
 * @param coords - BMU coords
 * @param itask - #task
 * @param n - row num in the input feature file
 */

void get_bmu_coord(int* coords,
                   int itask,
                   uint64_t n)
{
    float mindist = 9999.99;
    float dist = 0.0f;

    ///
    /// Check SOM_X * SOM_Y nodes one by one and compute the distance
    /// D(W_K, Fvec) and get the mindist and get the coords for the BMU.
    ///
    for (size_t som_y = 0; som_y < SOM_Y; som_y++) {
        for (size_t som_x = 0; som_x < SOM_X; som_x++) {
            dist = get_distance(som_y, som_x, itask, n);
            if (dist < mindist) {
                mindist = dist;
                coords[0] = som_x;
                coords[1] = som_y;
            }
        }
    }
}

/** Get best matching instance for a given unit
 * @param som_x, som_y - the coordinates of the unit
 * @param itask - #task
 */

int get_best_matching_instance(int som_y, int som_x)
{
    float mindist = 9999.99;
    float dist = 0.0f;
    int result = -1;
    ///
    /// Check SOM_X * SOM_Y nodes one by one and compute the distance
    /// D(W_K, Fvec) and get the mindist and get the coords for the BMU.
    ///
    for (uint64_t i = 0; i < NVECS/NVECSPERRANK; i++) {
        for (uint64_t n = 0; n < NVECSPERRANK; n++) {
            if (i*NVECSPERRANK+n<NVECS) {
                dist = get_distance(som_y, som_x, i, n);
                if (dist < mindist) {
                    mindist = dist;
                    result = i*NVECSPERRANK+n;
                }
            }
        }
    }
    return result;
}

/** MR-MPI Map function - Distance b/w vec and weight vector in the codebook
 * ,default distance_metric = Euclidean
 * @param somy
 * @param somx
 * @param vec2
 */

float get_distance(size_t somy,
                   size_t somx,
                   const svm_node* vec)
{
    float distance = 0.0f;
    size_t offset=0;
    size_t d=0;
    float n1 = 0.0f, n2 = 0.0f;
    while ( d < NDIMEN ) {
        if ( d == (vec+offset)->index ) {
            distance += (CODEBOOK[somy][somx][d]-
                         (vec+offset)->value)
                        *
                        (CODEBOOK[somy][somx][d]-
                         (vec+offset)->value);
            ++offset;
            ++d;
        } else {
            distance += CODEBOOK[somy][somx][d]*CODEBOOK[somy][somx][d];
            ++d;
        }
    }
    return sqrt(distance);
}

/** MR-MPI Map function - Distance b/w a feature vector and a weight vector
 * = Euclidean
 * @param som_y
 * @param som_x
 * @param r - row number in the input feature file
 * @param distance_metric
 */
float get_distance(size_t som_y,
                   size_t som_x,
                   int itask,
                   size_t r)
{
    return get_distance(som_y, som_x, x_matrix[itask*NVECSPERRANK+r]);
}

/** MR-MPI Map function - Distance b/w vec1 and vec2, default distance_metric
 * = Euclidean
 * @param vec1
 * @param vec2
 * @param distance_metric:
 */

float get_distance(const float* vec1,
                   const float* vec2)
{
    float distance = 0.0f;
    float n1 = 0.0f, n2 = 0.0f;
    for (size_t d = 0; d < NDIMEN; d++)
        distance += (vec1[d] - vec2[d]) * (vec1[d] - vec2[d]);
    return sqrt(distance);
}


/** MR-MPI Map function - Get weight vector from CODEBOOK using x, y index
 * @param som_y - y coordinate of a node in the map
 * @param som_x - x coordinate of a node in the map
 */

float* get_wvec(size_t som_y,
                size_t som_x)
{
    FLOAT_T* wvec = (FLOAT_T*)malloc(SZFLOAT * NDIMEN);
    for (size_t d = 0; d < NDIMEN; d++)
        wvec[d] = CODEBOOK[som_y][som_x][d]; /// CAUTION: (y,x) order

    return wvec;
}

/** Save u-matrix
 * @param fname
 */

int save_umat(char* fname)
{
    int D = 2;
    float min_dist = 1.5f;
    FILE* fp = fopen(fname, "wt");
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
                            FLOAT_T* vec1 = get_wvec(som_y1, som_x1);
                            FLOAT_T* vec2 = get_wvec(som_y2, som_x2);
                            dist += get_distance(vec1, vec2);
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

int read_matrix(const char *filename, bool with_classes)
{
    ifstream file;
    file.open(filename);
    string line;
    int elements = 0;
    while(getline(file,line))
    {
        stringstream   linestream(line);
        string         value;
        while(getline(linestream,value,' '))
        {
            elements++;
        }
        if (!with_classes) {
            elements++;
        }
        ++NVECS;
    }
    file.close();
    file.open(filename);
    x_matrix = new svm_node *[NVECS];
    x_space = new svm_node[elements];
    int max_index=-1;
    int j=0;
    for(int i=0; i<NVECS; i++)
    {

        x_matrix[i] = &x_space[j];
        getline(file, line);
        if (with_classes) {
            int first_separator=line.find_first_of(" ");;
            //string label = line.substr(0,first_separator);
            line=line.substr(first_separator+1);
        }
        stringstream   linestream(line);
        string         value;
        while(getline(linestream,value,' '))
        {
            int separator=value.find(":");
            istringstream myStream(value.substr(0,separator));
            myStream >> x_space[j].index;
            if(x_space[j].index > max_index)
                max_index = x_space[j].index;
            istringstream myStream2(value.substr(separator+1));
            myStream2 >> x_space[j].value;
            j++;
        }
        x_space[j++].index = -1;
    }
    NDIMEN=max_index+1;
    file.close();
    return 0;
}

int init_codebook(unsigned int seed)
{
    ///
    /// Fill initial random weights
    ///
    srand(seed);
    for (size_t som_y = 0; som_y < SOM_Y; som_y++) {
        for (size_t som_x = 0; som_x < SOM_X; som_x++) {
            for (size_t d = 0; d < NDIMEN; d++) {
                int w = 0xFFF & rand();
                w -= 0x800;
                CODEBOOK[som_y][som_x][d] = (FLOAT_T)w / 4096.0f;
            }
        }
    }

}

int load_codebook(const char *mapFilename)
{
    FILE* somMapFile = fopen(mapFilename, "r");
    if (somMapFile) {
        for (size_t y = 0; y < SOM_Y; y++) {
            for (size_t x = 0; x < SOM_X; x++) {
                for (size_t d = 0; d < NDIMEN; d++) {
                    FLOAT_T tmp = 0.0f;
                    fscanf(somMapFile, "%f", &tmp);
                    CODEBOOK[y][x][d] = tmp;
                }
            }
        }
    }
    else {
        cerr << "ERROR: codebook file does not exist.\n";
        return 1;
    }
    fclose(somMapFile);
    return 0;
}

int save_codebook(char* cbFileName)
{
    char temp[80];
    cout << "    Codebook file = " << cbFileName << endl;
    ofstream mapFile2(cbFileName);
    printf("    Saving Codebook...\n");
    if (mapFile2.is_open()) {
        for (size_t som_y = 0; som_y < SOM_Y; som_y++) {
            for (size_t som_x = 0; som_x < SOM_X; som_x++) {
                for (size_t d = 0; d < NDIMEN; d++) {
                    sprintf(temp, "%0.10f", CODEBOOK[som_y][som_x][d]);
                    mapFile2 << temp << "\t";
                }
            }
            mapFile2 << endl;
        }
        mapFile2.close();
        return 0;
    }
    else
    {
        return 1;
    }
}

/** Classify - Compute BMU for new test vectors on the trained SOM MAP. The
 * output is the coords (x, y) in the som map.
 * @param vec
 * @param p
 */

void classify(const svm_node* vec,
              int* p)
{
    float mindist = 9999.99;
    float dist = 0.0f;

    ///
    /// Check SOM_X * SOM_Y nodes one by one and compute the distance
    /// D(W_K, Fvec) and get the mindist and get the coords for the BMU.
    ///
    for (size_t som_y = 0; som_y < SOM_Y; som_y++) {
        for (size_t som_x = 0; som_x < SOM_X; som_x++) {
            dist = get_distance(som_y, som_x, vec);
            if (dist < mindist) {
                mindist = dist;
                p[0] = som_x;
                p[1] = som_y;
            }
        }
    }
}

/// EOF
