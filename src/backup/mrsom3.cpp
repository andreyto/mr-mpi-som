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
//
////////////////////////////////////////////////////////////////////////////////

//      This program is free software; you can redistribute it and/or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation; either version 2 of the License, or
//      (at your option) any later version.
//
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//      GNU General Public License for more details.
//
//      You should have received a copy of the GNU General Public License
//      along with this program; if not, write to the Free Software
//      Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//      MA 02110-1301, USA.


/// MPI and MapReduce-MPI
#include "mpi.h"
#include "./mrmpi/mapreduce.h"
#include "./mrmpi/keyvalue.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>

#include <vector>
#include <iostream>
#include <numeric>

/// For save
#include <fstream>

/// For split
#include <sstream>

/// For shuffle
#include <algorithm>

using namespace std;
using namespace MAPREDUCE_NS;

#define MAXSTR 255
#define SZFLOAT sizeof(float)
//#define NDEBUG
#define _DEBUG
//#ifdef _DEBUG
//#endif

/// SOM NODE
typedef struct node {
    vector<float>   weights;
    vector<float>   coords;     /// 2D: (x,y)
} NODE;
typedef vector<NODE *> V_NODEP_T;

/// SOM
typedef struct SOM {
    V_NODEP_T nodes;
} SOM;

typedef struct {
    int             m, n;       /// ROWS, COLS
    float          *data;       /// DATA, ORDERED BY ROW, THEN BY COL
    float         **rows;       /// POINTERS TO ROWS IN DATA
} DMatrix;

typedef vector<vector<vector<float> > > VVV_FLOAT_T;

struct GIFTBOX {
    SOM            *som;
    float           R;          /// RADIUS
    const DMatrix *codebook;
};

enum TRAINTYPE      { BATCH, ONLINE, BATCH2 };
enum DISTTYPE       { EUCL, SOSD, TXCB, ANGL, MHLN };
enum NORMALIZETYPE  { NONE, MNMX, ZSCR, SIGM, ENRG };
enum TRAINSPEED     { SLOW, FAST };

/* ------------------------------------------------------------------------ */
//void    train_online(SOM *som, float R, float Alpha, char *file, DMatrix *codebook);
void    train_online(SOM *som, float R, float Alpha, char *file);
void    train_batch_serial(SOM *som, float R, float Alpha, char* file, DMatrix *codebook);
//void    MR_train_batch(MapReduce *mr, SOM *som, DMatrix &F, DMatrix &codebook,
                       //int *loc, int *scattered,
                       //float R, int argc, char* argv[], int MPI_myId,
                       //char *myname, int MPI_nProcs
                       //);
//void    MR_train_batch_serial(MapReduce *mr, SOM *som, DMatrix &codebook,
                       //int *loc, int *scattered,
                       //float R, int argc, char* argv[], int MPI_myId,
                       //char *myname, int MPI_nProcs
                       //);                       
float   *normalize(DMatrix &F, int n, int normalizeopt);
//float   *get_coords(NODE *node);
float   get_distance(float *, vector<float> &, int);
float   get_distance2(vector<float> &vec, vector<float> &wvec, int distance_metric);
void    updatew_online(NODE *node, float *vec, float Alpha_x_Hck);
void    updatew_batch(NODE *, float *);
void    updatew_batch_index(NODE *node, float new_weight, int k);
void    get_file_name(char *path, char *name);
int     save_umat(SOM *som, char *fname);
NODE    *get_node(SOM *, int);
NODE    *get_bmu(SOM *, float *);

/// MATRIX
DMatrix createMatrix(const unsigned int rows, const unsigned int cols);
DMatrix initMatrix(void);
void    freeMatrix(DMatrix *matrix);
void    printMatrix(DMatrix A);
int     validMatrix(DMatrix matrix);

/// MAPPER AND REDUCER
//void    MR_compute_weight(int itask, KeyValue *kv, void *ptr);
//void    MR_compute_weight2(int itask, char *file, KeyValue *kv, void *ptr);
//void    MR_accumul_weight(char *key, int keybytes, char *multivalue,
                          //int nvalues, int *valuebytes, KeyValue *kv,
                          //void *ptr);
//void    MR_update_weight(uint64_t itask, char *key, int keybytes, char *value,
                         //int valuebytes, KeyValue *kv, void *ptr);

/// MR fuctions
void    mr_train_batch(int itask, char *file, KeyValue *kv, void *ptr);
void    mr_update_weight(uint64_t itask, char *key, int keybytes, char *value, int valuebytes, KeyValue *kv, void *ptr);
void    mr_sum(char *key, int keybytes, char *multivalue, int nvalues, int *valuebytes, KeyValue *kv, void *ptr);
         
float   *get_bmu_coord(const DMatrix *codebook, const float *fvec);
//float   *get_wvec(unsigned int som_x, unsigned int som_y, const DMatrix *codebook);
float   *get_wvec2(unsigned int som_x, unsigned int som_y, const DMatrix *codebook);
float   get_distance3(float *vec1, const float *vec2, int distance_metric);

/// Util
string float2str(float number);
float str2float(string str);
string uint2str(uint64_t number);
uint64_t str2uint(string str);

/// Tokenizer routines
vector<string>  &split(const string &s, char delim, vector<string> &vecElems);
vector<string>  split(const string &s, char delim);

/// GLOBALS
int     NDIMEN;                 /// NUM OF DIMENSIONALITY
int     NVECS;                  /// NUM OF FEATURE VECTORS
int     SOM_X = 50;
int     SOM_Y = 50;
int     SOM_D = 2;              /// 2=2D
int     NNODES = SOM_X * SOM_Y; /// TOTAL NUM OF SOM NODES
int     NEPOCHS;                /// ITERATIONS
int     DISTOPT = EUCL;         /// 0=EUCL, 1=SOSD, 2=TXCB, 3=ANGL, 4=MHLN
int     TRAINMODE = ONLINE;     /// 0=BATCH, 1=ONLINE
int     TRAINOPT = SLOW;        /// 0=SLOW, 1=FAST
int     NORMALIZEOPT = NONE;    /// 0=NONE, 1=MNMX, 2=ZSCR, 3=SIGM, 4=ENRG
int     SHUFFLE = 0;
/* ------------------------------------------------------------------------ */
int main(int argc, char *argv[])
/* ------------------------------------------------------------------------ */
{
    SOM *som;
    if (argc == 6) {  /// READ FEATURE DATA FROM FILE
        /// syntax: mrsom FILE NEPOCHS TRAINMODE NVECS NDIMEN
        NEPOCHS = atoi(argv[2]);
        TRAINMODE = atoi(argv[3]);
        NVECS = atoi(argv[4]);
        NDIMEN = atoi(argv[5]);
    }
    else if (argc == 8) {  /// READ FEATURE DATA FROM FILE
        /// syntax: mrsom FILE NEPOCHS TRAINMODE NVECS NDIMEN X Y
        NEPOCHS = atoi(argv[2]);
        TRAINMODE = atoi(argv[3]);
        NVECS = atoi(argv[4]);
        NDIMEN = atoi(argv[5]);
        SOM_X = atoi(argv[6]);
        SOM_Y = atoi(argv[7]);
        NNODES = SOM_X * SOM_Y;
    }
    else {
        printf("         mrsom FILE NEPOCHS TRAINMODE NVECS NDIMEN [X Y]\n\n");
        printf("         FILE    = feature vector file.\n");
        printf("         NEPOCHS = number of iterations.\n");
        printf("         TRAINMODE   = 0-batch, 1-online.\n");
        printf("         NVECS   = number of feature vectors.\n");
        printf("         NDIMEN  = number of dimensionality of feature vector.\n");
        printf("         [X Y]   = optional, SOM map size. Default = [50 50]\n");
        exit(0);
    }
    
    /// MAKE SOM//////////////////////////////////////////////////////////
    som = (SOM *)malloc(sizeof(SOM));
    som->nodes = V_NODEP_T(NNODES);
    for (int i = 0; i < SOM_X * SOM_Y; i++) {
        som->nodes[i] = (NODE *)malloc(sizeof(NODE*));
    }
    
    /// FILL INITIAL RANDOM WEIGHTS///////////////////////////////////////
    srand((unsigned int)time(0));
    for (int x = 0; x < NNODES; x++) {
        NODE *node = (NODE *)malloc(sizeof(NODE));
        node->weights.resize(NDIMEN);
        node->coords.resize(SOM_D, 0.0);
        for (int i = 0; i < NDIMEN; i++) {
            int w = 0xFFF & rand();
            w -= 0x800;
            node->weights[i] = (float)w / 4096.0f;
        }
        som->nodes[x] = node;
    }
    
    /// FILE COORDS (2D RECT)/////////////////////////////////////////////
    for (int y = 0; y < SOM_Y; y++) {
        for (int x = 0; x < SOM_X; x++) {        
            //som->nodes[(x*SOM_Y)+y]->coords[0] = y * 1.0f;
            //som->nodes[(x*SOM_Y)+y]->coords[1] = x * 1.0f;
            som->nodes[y*SOM_X+x]->coords[0] = x * 1.0f;
            som->nodes[y*SOM_X+x]->coords[1] = y * 1.0f;
        }
    }
    //for (int y = 0; y < SOM_Y; y++) {
        //for (int x = 0; x < SOM_X; x++) {        
            //cout << som->nodes[y*SOM_X+x]->coords[0] << " "
                 //<< som->nodes[y*SOM_X+x]->coords[1] << endl;
        //}   
    //}
    //for (int y = 0; y < SOM_Y; y++) {
        //for (int x = 0; x < SOM_X; x++) {        
            //som->nodes[y*SOM_X + x]->coords[0] = x * 1.0f;
            //som->nodes[y*SOM_X + x]->coords[1] = y * 1.0f;
        //}
    //}
    //int x1 = 1;
    //int y1 = 2;
    //cout << som->nodes[y1*SOM_X + x1]->coords[0] << " " << som->nodes[y1*SOM_X + x1]->coords[1] << endl;
    
    /// create weight matrix //////////////////////////////////////////////
    DMatrix codebook;
    codebook = initMatrix();
    codebook = createMatrix(SOM_Y, SOM_X*NDIMEN);
    if (!validMatrix(codebook)) {
        printf("FATAL: not valid codebook matrix.\n");
        exit(0);
    }
    
    /// MPI ///////////////////////////////////////////////////////////////
    int MPI_myId, MPI_nProcs, MPI_length;
    char myname[MAXSTR];
    int ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "MPI initsialization failed !\n");
        exit(0);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_myId);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_nProcs);
    MPI_Get_processor_name(myname, &MPI_length);
    
    //double time0, time1;
    //struct timeval t1_start;
    //struct timeval t1_end;
    //double t1_time;
    //gettimeofday(&t1_start, NULL);
    
    /// MR-MPI////////////////////////////////////////////////////////////
    MapReduce *mr = new MapReduce(MPI_COMM_WORLD);
    mr->verbosity = 0;
    mr->timer = 0;
    mr->mapstyle = 2;  /// master/slave mode

    /// Prepare random vectors if random gen is chosen
    /// and prepare an index vector to scatter among tasks
    if (MPI_myId == 0) {
        printf("INFO: %d x %d SOM, num epochs = %d, num features = %d, dimensionality = %d\n", SOM_X, SOM_Y, NEPOCHS, NVECS, NDIMEN);
        printf("Reading (%d x %d) feature vectors from %s...\n", NVECS, NDIMEN, argv[1]);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    float N = (float)NEPOCHS;       /// ITERATIONS
    float nrule, nrule0 = 0.9f;     /// LEARNING RATE FACTOR
    float R, R0;
    R0 = SOM_X / 2.0f;              /// INIT RADIUS FOR UPDATING NEIGHBORS
    R = R0;
    int x = 0;                      /// 0...N-1
    
    /*
     * SHOULD ADD A ROUTINE TO CHECK WHEN TO STOP THE UPDATE IN BATCH
     * MODE. OTHERWISE, THE RESULTING MAP WILL BE GETTING DETERIORATED
     * IN TERMS OF THE QUALITY.
     *
     * UPDATE: THERE IS NO STOPPING CRITERIA IN SOM. 07.13.2010
     */     
    GIFTBOX gf;
        
    if (TRAINMODE == BATCH) {  
        while (NEPOCHS && R >= 1.0) {
            if (MPI_myId == 0) {
                /// R to broadcast
                R = R0 * exp(-10.0f * (x * x) / (N * N));
                x++;
                
                /// Prepare weight vectors to broadcast
                for (int y = 0; y < SOM_Y; y++) 
                    for (int x = 0; x < SOM_X; x++)                     
                        for (int d = 0; d < NDIMEN; d++)
                            codebook.rows[y][x*NDIMEN + d] = som->nodes[y*SOM_X+x]->weights[d];
                
                //for (int y = 0; y < SOM_Y; y++) 
                    //for (int x = 0; x < SOM_X; x++)                 
                        //for (int d = 0; d < NDIMEN; d++)
                            //cout << codebook.rows[y][x*NDIMEN + d] << " " << som->nodes[y*SOM_X+x]->weights[d] << endl;
                
                printf("BATCH-  epoch: %d   R: %.2f \n", (NEPOCHS - 1), R);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast(&R, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            
            /// Broadcast weight vectors for the next epoch
            MPI_Bcast((void *)codebook.data, SOM_Y*SOM_X*NDIMEN, MPI_FLOAT, 0, MPI_COMM_WORLD);
            
            //MR_train_batch(mr, som, codebook, loc,
                           //scattered, R, argc, argv, MPI_myId, myname, MPI_nProcs);
    
            gf.som = som;       
            gf.R = R;            
            gf.codebook = &codebook;
            
            uint64_t nRes = mr->map(argv[1], &mr_train_batch, &gf);
            //mr->print(-1, 1, 5, 5);
            //cout << "nRes = " << nRes << endl;
            mr->collate(NULL);
            //mr->print(-1, 1, 5, 5);            
            nRes = mr->reduce(&mr_sum, NULL);
            //mr->print(-1, 1, 5, 5);  
            mr->gather(1);
            //mr->print(0, 1, 5, 5);
            nRes = mr->map(mr, &mr_update_weight, &gf);
            //cout << "nRes = " << nRes << endl;
                       
         
            NEPOCHS--;
        }
    }
    else if (TRAINMODE == ONLINE) {  
        while (NEPOCHS) {                                       
            if (MPI_myId == 0) {
                R = R0 * exp(-10.0f * (x * x) / (N * N));
                
                /// Learning rule shrinks over time, for online som
                nrule = nrule0 * exp(-10.0f * (x * x) / (N * N));  
                x++;                 
                
                /// Test
                //for (int y = 0; y < SOM_Y; y++) 
                    //for (int x = 0; x < SOM_X; x++)                     
                        //for (int d = 0; d < NDIMEN; d++)
                            //codebook.rows[y][x*NDIMEN + d] = som->nodes[y*SOM_X+x]->weights[d];
                
                //for (int y = 0; y < SOM_Y; y++) 
                    //for (int x = 0; x < SOM_X; x++)                 
                        //for (int d = 0; d < NDIMEN; d++)
                            //cout << codebook.rows[y][x*NDIMEN + d] << " " << som->nodes[y*SOM_X+x]->weights[d] << endl;
             
                //train_online(som, R, nrule, argv[1], &codebook); 
                //printMatrix(codebook);
                ///
                
                train_online(som, R, nrule, argv[1]); 

                printf("ONLINE-  epoch: %d   R: %.2f \n", (NEPOCHS - 1), R);
            }
        
            NEPOCHS--;
        }
    }
    else if (TRAINMODE == BATCH2) {  
        while (NEPOCHS && R >= 1.0) {
        //while (NEPOCHS) {
            if (MPI_myId == 0) {
                R = R0 * exp(-10.0f * (x * x) / (N * N));
                x++;                 
                
                for (int y = 0; y < SOM_Y; y++) 
                    for (int x = 0; x < SOM_X; x++)                     
                        for (int d = 0; d < NDIMEN; d++)
                            codebook.rows[y][x*NDIMEN + d] = som->nodes[y*SOM_X+x]->weights[d];
                
                //for (int y = 0; y < SOM_Y; y++) 
                    //for (int x = 0; x < SOM_X; x++)                 
                        //for (int d = 0; d < NDIMEN; d++)
                            //cout << codebook.rows[y][x*NDIMEN + d] << " " << som->nodes[y*SOM_X+x]->weights[d] << endl;
             
                train_batch_serial(som, R, nrule, argv[1], &codebook); 
                printf("BATCH2-  epoch: %d   R: %.2f \n", (NEPOCHS - 1), R);
            }
        
            NEPOCHS--;                        
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    //gettimeofday(&t1_end, NULL);
    //t1_time = t1_end.tv_sec - t1_start.tv_sec + (t1_end.tv_usec - t1_start.tv_usec) / 1.e6;
    //fprintf(stderr,"\n**** Processing time: %g seconds **** \n",t1_time);
    //if (MPI_myId == 0) {
    //for (int i=0; i<NNODES; i++)
    //printf("%f %f %f, %f %f %f\n", som->nodes[i]->weights[0],som->nodes[i]->weights[1],som->nodes[i]->weights[2], codebook.rows[i][0], codebook.rows[i][1], codebook.rows[i][2]);
    //}
    
    freeMatrix(&codebook);
        
    /// SAVE SOM//////////////////////////////////////////////////////////
    if (MPI_myId == 0) {
        printf("### Saving SOM map and U-Matrix...\n");
        char som_map[MAXSTR] = "";
        strcat(som_map, "result.umat.txt");
        save_umat(som, som_map);
        printf("    Converting SOM map to U-map...\n");
        //system("python ./show2.py result.umat.txt");
        printf("    Done1 !\n");
        
        ///
        /// Save SOM map for umat tool
        /// Usage: /tools/umat -cin result.map > result.eps
        ///
        printf("    Saving SOM map...\n");
        char *outFileName2 = "result.map.txt";
        ofstream mapFile(outFileName2);
        char temp[80];
        if (mapFile.is_open()) {
            mapFile << NDIMEN << " rect " << SOM_X << " " << SOM_Y << endl;
            for (int x = 0; x < NNODES; x++) {
                for (int j = 0; j < NDIMEN; j++) {
                    sprintf(temp, "%0.10f", som->nodes[x]->weights[j]);
                    mapFile << temp << " ";                    
                }
                mapFile << endl;
            }            
        }
        mapFile.close();        
        //system("../tools/umat -cin result.map.txt > test.eps");
        printf("    Done2 !\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int x = 0; x < SOM_X * SOM_Y; x++) {
        delete som->nodes[x];
    }
    
    delete som;
    delete mr;
    MPI_Finalize();

    
    return 0;
}

void train_batch_serial(SOM *som, float R, float Alpha, char* file, DMatrix *codebook)
{
    ///
    /// Read feature chunk file distributed
    ///
    DMatrix data;
    data = initMatrix();
    data = createMatrix(NVECS, NDIMEN);
    if (!validMatrix(data)) {
        printf("FATAL: not valid data matrix.\n");
        exit(0);
    }
    
    ///
    /// Read file and fill the matrix
    ///
    FILE *fp;   
    fp = fopen(file, "r");
    if (SHUFFLE) {
        vector<vector<float> > vvFeature(NVECS, vector<float> (NDIMEN));
        vector<uint64_t> vRowIdx;
        for (uint64_t row = 0; row < NVECS; row++) { 
            vRowIdx.push_back(row);
            for (uint64_t col = 0; col < NDIMEN; col++) {
                float tmp = 0.0f;
                fscanf(fp, "%f", &tmp);
                vvFeature[row][col] = tmp;
            }
        }
        random_shuffle(vRowIdx.begin(), vRowIdx.end());
        
        ///
        /// Set data matrix with shuffled rows
        ///
        for (uint64_t row = 0; row < NVECS; row++)
            for (uint64_t col = 0; col < NDIMEN; col++)
                data.rows[row][col] = vvFeature[vRowIdx[row]][col];
                
        vvFeature.clear();
        vRowIdx.clear();
    }
    else {
        for (uint64_t row = 0; row < NVECS; row++) { 
            for (uint64_t col = 0; col < NDIMEN; col++) {
                float tmp = 0.0f;
                fscanf(fp, "%f", &tmp);
                data.rows[row][col] = tmp;
            }
        }           
    }
    fclose(fp);            
    
    
    ///
    /// Read data one by one and compute denom and numer and add to KV
    ///
    float p2[SOM_D];
    VVV_FLOAT_T numer;    
    numer = VVV_FLOAT_T(SOM_Y, vector<vector<float> > (SOM_X,
                        vector<float>(NDIMEN, 0.0)));
    VVV_FLOAT_T denom;
    denom = VVV_FLOAT_T(SOM_Y, vector<vector<float> > (SOM_X,
                        vector<float>(NDIMEN, 0.0)));
                        
    for (uint64_t n = 0; n < NVECS; n++) {
        
        /// Normalize
        float *normalized = normalize(data, n, NORMALIZEOPT);
        
        /// Get the best matching unit
        /// p1[0] = x, p1[1] = y
        const float *p1 = get_bmu_coord(codebook, normalized);
        //cout << "p1 = " << p1[0] << " " << p1[1] << endl;
        
        for (uint64_t som_y = 0; som_y < SOM_Y; som_y++) { 
            for (uint64_t som_x = 0; som_x < SOM_X; som_x++) {
                p2[0] = (float) som_x;
                p2[1] = (float) som_y;
                float dist = 0.0f;
                for (int p = 0; p < NDIMEN; p++)
                    dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
                dist = sqrt(dist);
                
                float neighbor_fuct = 0.0f;
                neighbor_fuct = exp(-(1.0f * dist * dist) / (R * R));
                
                for (int w = 0; w < NDIMEN; w++) {
                    float tempNumer = 1.0f * neighbor_fuct * normalized[w];
                    float tempDenom = neighbor_fuct;
                    numer[som_y][som_x][w] += 1.0f * neighbor_fuct * normalized[w];
                    denom[som_y][som_x][w] += neighbor_fuct;
                }                
            }
        }
        
        delete p1;
        delete normalized; 
    }
    
    for (uint64_t som_y = 0; som_y < SOM_Y; som_y++) { 
        for (uint64_t som_x = 0; som_x < SOM_X; som_x++) {
            cout << "numer, denom, newweight = ";
            for (int w = 0; w < NDIMEN; w++) {
                 cout << numer[som_y][som_x][w] << " "
                     << denom[som_y][som_x][w] << " "
                     << numer[som_y][som_x][w] / denom[som_y][som_x][w] << endl;
            }            
        }
    }
    
    for (uint64_t som_y = 0; som_y < SOM_Y; som_y++) { 
        for (uint64_t som_x = 0; som_x < SOM_X; som_x++) {
            for (int w = 0; w < NDIMEN; w++) {
                if (denom[som_y][som_x][w] != 0) /// NOTE: THIS IS MANDATORY!!
                    som->nodes[som_y*SOM_X + som_x]->weights[w] = numer[som_y][som_x][w] / denom[som_y][som_x][w];
            }
        }
    }
    
    numer.clear();
    denom.clear();
    freeMatrix(&data);
}


/** A Reduce user-defined function - Sum numer and denom
 * (Qid,DBid) key into Qid for further aggregating.
 * @param key
 * @param keybytes
 * @param multivalue: collected blast result strings.  
 * @param nvalues
 * @param valuebytes
 * @param kv
 * @param ptr
 */
void mr_sum(char *key, int keybytes, char *multivalue, int nvalues, int *valuebytes, 
         KeyValue *kv, void *ptr)
{   
    float numer = 0.0;
    float denom = 0.0;

    for (int i = 0; i < nvalues; i++) {
        stringstream ss;
        ss << multivalue;
        //cout << ss.str() << endl;
        vector<string> vValue = split(ss.str(), ',');
        assert(vValue.size() == 2);
        //cout << vValue[0] << " " << vValue[1] << endl;
        numer += str2float(vValue[0]);
        denom += str2float(vValue[1]);
        
        multivalue += valuebytes[i];        
    }
    string value = float2str(numer) + "," + float2str(denom);
    kv->add(key, strlen(key)+1, (char*)value.c_str(), value.length()+1);
}

/* ------------------------------------------------------------------------ */
void mr_update_weight(uint64_t itask, char *key, int keybytes, char *value,
                   int valuebytes, KeyValue *kv, void *ptr)
/* ------------------------------------------------------------------------ */
{
    GIFTBOX *gf = (GIFTBOX *) ptr;
    vector<string> vKey = split(string(key), ',');
    assert(vKey.size() == 3);
    vector<string> vValue = split(string(value), ',');
    assert(vValue.size() == 2);

    uint64_t row = str2uint(vKey[0]);
    uint64_t col = str2uint(vKey[1]);
    unsigned int d = str2uint(vKey[2]);    

    float numer = str2float(vValue[0]);
    float denom = str2float(vValue[1]);
    float newWeight = 0.0;
    if (denom != 0)
        newWeight = numer / denom;
    //char temp[80];
    //sprintf(temp, "%0.5f", newWeight);   
    
    //cout << "row, col, col*NDIMEN+d, numer, denom, newWeight = " 
         //<< row << ","
         //<< col << ","
         ////<< col*NDIMEN + d << ","
         //<< d << ","
         //<< numer << ","
         //<< denom << ","
         //<< newWeight << endl;         
         
    //gf->codebook->rows[row][col*NDIMEN + d] = newWeight;
    //updatew_online(som->nodes[k], normalized, Alpha * neighbor_fuct);
    if (newWeight > 0.0) /// NOTE: THIS IS MANDATORY!!
        gf->som->nodes[row*SOM_X + col]->weights[d] = newWeight;
}

/** MR-MPI Map function - batch training
 * @param itask
 * @param file
 * @param kv
 * @param ptr
 */
/* ------------------------------------------------------------------------ */
void mr_train_batch(int itask, char *file, KeyValue *kv, void *ptr)
/* ------------------------------------------------------------------------ */
{
    GIFTBOX *gf = (GIFTBOX *) ptr;
    
    ///
    /// Read feature chunk file distributed
    ///
    string workItem(file);
    cout << "workItem = " << workItem << endl;

    DMatrix data;
    data = initMatrix();
    data = createMatrix(NVECS, NDIMEN);
    if (!validMatrix(data)) {
        printf("FATAL: not valid data matrix.\n");
        exit(0);
    }
    
    ///
    /// Read file and fill the matrix
    ///
    FILE *fp;   
    fp = fopen(file, "r");
    if (SHUFFLE) {
        vector<vector<float> > vvFeature(NVECS, vector<float> (NDIMEN));
        vector<uint64_t> vRowIdx;
        for (uint64_t row = 0; row < NVECS; row++) { 
            vRowIdx.push_back(row);
            for (uint64_t col = 0; col < NDIMEN; col++) {
                float tmp = 0.0f;
                fscanf(fp, "%f", &tmp);
                vvFeature[row][col] = tmp;
            }
        }
        random_shuffle(vRowIdx.begin(), vRowIdx.end());
        
        ///
        /// Set data matrix with shuffled rows
        ///
        for (uint64_t row = 0; row < NVECS; row++)
            for (uint64_t col = 0; col < NDIMEN; col++)
                data.rows[row][col] = vvFeature[vRowIdx[row]][col];
                
        vvFeature.clear();
        vRowIdx.clear();
    }
    else {
        for (uint64_t row = 0; row < NVECS; row++) { 
            for (uint64_t col = 0; col < NDIMEN; col++) {
                float tmp = 0.0f;
                fscanf(fp, "%f", &tmp);
                data.rows[row][col] = tmp;
            }
        }           
    }
    fclose(fp);            
    
    
    ///
    /// Read data one by one and compute denom and numer and add to KV
    ///
    float p2[SOM_D];
    VVV_FLOAT_T numer;    
    numer = VVV_FLOAT_T(SOM_Y, vector<vector<float> > (SOM_X,
                        vector<float>(NDIMEN, 0.0)));
    VVV_FLOAT_T denom;
    denom = VVV_FLOAT_T(SOM_Y, vector<vector<float> > (SOM_X,
                        vector<float>(NDIMEN, 0.0)));
                        
    for (uint64_t n = 0; n < NVECS; n++) {
        
        /// Normalize
        float *normalized = normalize(data, n, NORMALIZEOPT);
        
        /// Get the best matching unit
        /// p1[0] = x, p1[1] = y
        const float *p1 = get_bmu_coord(gf->codebook, normalized);
        //cout << "p1 = " << p1[0] << " " << p1[1] << endl;
        
        for (uint64_t som_y = 0; som_y < SOM_Y; som_y++) { 
            for (uint64_t som_x = 0; som_x < SOM_X; som_x++) {
                p2[0] = (float) som_x;
                p2[1] = (float) som_y;
                float dist = 0.0f;
                for (int p = 0; p < NDIMEN; p++)
                    dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
                dist = sqrt(dist);
                
                float neighbor_fuct = 0.0f;
                neighbor_fuct = exp(-(1.0f * dist * dist) / (gf->R * gf->R));
                
                for (int w = 0; w < NDIMEN; w++) {
                    float tempNumer = 1.0f * neighbor_fuct * normalized[w];
                    float tempDenom = neighbor_fuct;
                    numer[som_y][som_x][w] += 1.0f * neighbor_fuct * normalized[w];
                    denom[som_y][som_x][w] += neighbor_fuct;
                }                
            }
        }
        
        delete p1;
        delete normalized; 
    }
    
    for (uint64_t som_y = 0; som_y < SOM_Y; som_y++) { 
        for (uint64_t som_x = 0; som_x < SOM_X; som_x++) {
            for (int w = 0; w < NDIMEN; w++) {
                string key = uint2str(som_y) + "," + uint2str(som_x) + "," + uint2str(w);
                string value = float2str(numer[som_y][som_x][w]) + "," + float2str(denom[som_y][som_x][w]);
                kv->add((char*)key.c_str(), key.length()+1, (char*)value.c_str(), value.length()+1);
            }
        }
    }
    
    numer.clear();
    denom.clear();
    freeMatrix(&data);
}


/* ------------------------------------------------------------------------ */
//void train_online(SOM *som, float R, float Alpha, char* file, DMatrix *codebook)
void train_online(SOM *som, float R, float Alpha, char* file)
/* ------------------------------------------------------------------------ */
{
    ///
    /// Create a matrix for feature vectors
    ///
    DMatrix f;
    f = initMatrix();
    f = createMatrix(NVECS, NDIMEN);
    if (!validMatrix(f)) {
        printf("FATAL: not valid data matrix.\n");
        exit(0);
    }
    
    ///
    /// Read file and fill the matrix
    ///
    FILE *fp;   
    fp = fopen(file, "r");
    if (SHUFFLE) {
        vector<vector<float> > vvFeature(NVECS, vector<float> (NDIMEN));
        vector<uint64_t> vRowIdx;
        for (uint64_t row = 0; row < NVECS; row++) { 
            vRowIdx.push_back(row);
            for (uint64_t col = 0; col < NDIMEN; col++) {
                float tmp = 0.0f;
                fscanf(fp, "%f", &tmp);
                vvFeature[row][col] = tmp;
            }
        }
        random_shuffle(vRowIdx.begin(), vRowIdx.end());
        
        ///
        /// Set data matrix with shuffled rows
        ///
        for (uint64_t row = 0; row < NVECS; row++)
            for (uint64_t col = 0; col < NDIMEN; col++)
                f.rows[row][col] = vvFeature[vRowIdx[row]][col];
                
        vvFeature.clear();
        vRowIdx.clear();
    }
    else {
        for (uint64_t row = 0; row < NVECS; row++) { 
            for (uint64_t col = 0; col < NDIMEN; col++) {
                float tmp = 0.0f;
                fscanf(fp, "%f", &tmp);
                f.rows[row][col] = tmp;
            }
        }           
    }
    fclose(fp);            
    
    for (int n = 0; n < NVECS; n++) {
        
        /// Normalize vector
        float *normalized = normalize(f, n, NORMALIZEOPT);
        //cout << "Normalized = ";
        //for (int i = 0; i < NDIMEN; i++)
            //cout << normalized[i] << " ";
        //cout << endl;
        
        /// Get best node using d_k (t) = || x(t) = w_k (t) || ^2
        /// and d_c (t) == min d_k (t)
        /// p1[0] = x, p1[1] = y
        NODE *bmu_node = get_bmu(som, normalized);
        //const float *p1 = get_coords(bmu_node);
        float p1[SOM_D];
        p1[0] = bmu_node->coords[0];
        p1[1] = bmu_node->coords[1];
        cout << "BMU coords = " << p1[0] << "," << p1[1] << endl;
        
        /// test
        //const float *p12 = get_bmu_coord(codebook, normalized);
        //cout << "p12 = " << p12[0] << " " << p12[1] << endl;
        ///
        
        if (R <= 1.0f) { /// ADJUST BMU NODE ONLY
            updatew_online(bmu_node, normalized, Alpha);
        }
        else {   /// Adjust weight vectors of the neighbors to bmu node
            for (int k = 0; k < NNODES; k++) { //ADJUST WEIGHTS OF ALL K NODES IF DISTOPT <= R
                //const float *p2 = get_coords(som->nodes[k]);
                float p2[SOM_D];
                p2[0] = som->nodes[k]->coords[0];
                p2[1] = som->nodes[k]->coords[1];
                float dist = 0.0f;
                
                /// dist = sqrt((x1-y1)^2 + (x2-y2)^2 + ...)  DISTANCE TO NODE
                for (int p = 0; p < NDIMEN; p++)
                    dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
                dist = sqrt(dist);
                
                if (TRAINOPT == FAST && dist > R) continue;
                
                /// Gaussian neighborhood function
                float neighbor_fuct = exp(-(1.0f * dist * dist) / (R * R));
                
                updatew_online(som->nodes[k], normalized, Alpha * neighbor_fuct);
                
                //delete p2;
            }
        }
        
        //delete p1;
        delete normalized;
    }
    
    freeMatrix(&f);    
}

/** MR-MPI Map function - Get BMU node coords
 * @param const DMatrix *codebook
 * @param const float *fvec
 */
/* ------------------------------------------------------------------------ */
float *get_bmu_coord(const DMatrix *codebook, const float *fvec)
/* ------------------------------------------------------------------------ */
{
    float *coords = (float *)malloc(SZFLOAT*SOM_D);
    //float *wvec = get_wvec(0, 0, codebook);
    //float mindist = get_distance(wvec, fvec, DISTOPT);
    float mindist = 9999.99;
    float dist = 0.0f;
    
    ///
    /// Check SOM_X * SOM_Y nodes one by one and compute the distance 
    /// D(W_K, Fvec) and get the mindist and get the coords for the BMU.
    ///
    for (uint64_t som_y = 0; som_y < SOM_Y; som_y++) { 
        for (uint64_t som_x = 0; som_x < SOM_X; som_x++) {        
        
            float *tempVec = get_wvec2(som_y, som_x, codebook); 
            //cout << "wvec = ";
            //for (int i = 0; i < NDIMEN; i++)
                //cout << tempVec[i] << " ";
            //cout << endl;
            
            dist = get_distance3(tempVec, fvec, DISTOPT);
            
            if (dist < mindist) { 
                mindist = dist;
                coords[0] = (float) som_x;
                coords[1] = (float) som_y;
                //cout << "Coords = " << coords[0] << "," <<  coords[1] << endl; 
            }
            
            delete tempVec;
        }
    }
    //delete wvec;
    
    /// CAN ADD A FEATURE FOR VOTING AMONG BMUS.    
    
    return coords;
}

/** MR-MPI Map function - Get weight vector from codebook using x, y index
 * @param unsigned int somx
 * @param unsigned int somy
 * @param const DMatrix *codebook
 */
///* ------------------------------------------------------------------------ */
//float *get_wvec(unsigned int som_x, unsigned int som_y, const DMatrix *codebook)
///* ------------------------------------------------------------------------ */
//{
    //float *wvec = (float *)malloc(SZFLOAT * NDIMEN);
    //for (unsigned int i = 0; i < NDIMEN; i++)
        //wvec[i] = codebook->rows[som_y][som_x*NDIMEN + i]; /// CAUTION: (y,x) order
    
    //return wvec;
//}

/* ------------------------------------------------------------------------ */
float *get_wvec2(unsigned int som_y, unsigned int som_x, const DMatrix *codebook)
/* ------------------------------------------------------------------------ */
{
    float *wvec = (float *)malloc(SZFLOAT * NDIMEN);
    for (unsigned int d = 0; d < NDIMEN; d++)
        wvec[d] = codebook->rows[som_y][som_x*NDIMEN + d]; /// CAUTION: (y,x) order
    
    return wvec;
}

///* ------------------------------------------------------------------------ */
//void train_online(SOM *som, DMatrix &f, float R, float Alpha, int *loc)
///* ------------------------------------------------------------------------ */
//{
    //for (int n = 0; n < NVECS; n++) {
        //float *normalized = normalize(f, loc[n]);
        
        ///// GET BEST NODE USING d_k (t) = || x(t) = w_k (t) || ^2
        ///// AND d_c (t) == min d_k (t)
        //NODE *bmu_node = get_bmu(som, normalized);
        //const float *p1 = get_coords(bmu_node);
        //if (R <= 1.0f) { /// ADJUST BMU NODE ONLY
            //updatew_online(bmu_node, normalized, Alpha);
        //}
        //else {   /// ADJUST WEIGHT VECTORS OF THE NEIGHBORS TO BMU NODE
            //for (int k = 0; k < NNODES; k++) { //ADJUST WEIGHTS OF ALL K NODES IF DISTOPT <= R
                //const float *p2 = get_coords(som->nodes[k]);
                //float dist = 0.0f;
                ///// dist = sqrt((x1-y1)^2 + (x2-y2)^2 + ...)  DISTANCE TO NODE
                //for (int p = 0; p < NDIMEN; p++)
                    //dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
                //dist = sqrt(dist);
                //if (TRAINOPT == 1 && dist > R)
                    //continue;
                ///// GAUSSIAN NEIGHBORHOOD FUNCTION
                //float neighbor_fuct = exp(-(1.0f * dist * dist) / (R * R));
                //updatew_online(som->nodes[k], normalized, Alpha * neighbor_fuct);
            //}
        //}
        
        //delete normalized;
    //}    
//}

///* ------------------------------------------------------------------------ */
//void MR_train_batch_serial(MapReduce *mr, SOM *som, DMatrix &w,
                    //int *loc, int *scattered,
                    //float R, int argc, char* argv[], int MPI_myId,
                    //char *myname, int MPI_nProcs)
///* ------------------------------------------------------------------------ */
//{
    //if (R > 1.0f) {
        //int new_nvecs = NVECS / MPI_nProcs; /// CHUNNK SIZE
        //GIFTBOX gfbox;
        //gfbox.som = som;
        //gfbox.new_nvecs = new_nvecs;
        ////gfbox.f_vectors = &f;
        ////gfbox.w_vectors = &w;
        ////gfbox.numer = &numer_vec;
        ////gfbox.denom = &denom_vec;
        //gfbox.R = R;
        //gfbox.idx_start = scattered[0];
        //gfbox.loc = loc;
        ////double tstart = MPI_Wtime();
        //MPI_Barrier(MPI_COMM_WORLD);
        ////fprintf(stderr,"[Node %d]: %s train_batch_MR FEATURES %0.2f %0.2f %0.2f **** \n", MPI_myId, myname, f.rows[0][0],f.rows[0][1],f.rows[0][2]);
        //int num_keywords = mr->map(argv[1], &MR_compute_weight2, &gfbox);
        ////fprintf(stderr,"[Node %d]: %s, mapper1 ends! -- num_keywords = %d! -------\n", MPI_myId, myname, num_keywords);
        ////mr->print(-1, 1, 5, 3);
        //int total_num_KM = mr->collate(NULL);     /// AGGREGATES A KEYVALUE OBJECT ACROSS PROCESSORS AND CONVERTS IT INTO A KEYMULTIVALUE OBJECT.
        ////int total_num_KM = mr->clone();         /// CONVERTS A KEYVALUE OBJECT DIRECTLY INTO A KEYMULTIVALUE OBJECT.
        ////fprintf(stderr,"[Node %d]: %s ------ COLLATE ends! total_num_KM = %d -------\n", MPI_myId, myname, total_num_KM);
        ////mr->print(-1, 1, 5, 3);
        ////MPI_Barrier(MPI_COMM_WORLD);
        //int num_unique = mr->reduce(&MR_accumul_weight, (void *)NULL);
        //////int num_unique = mr->reduce(&accumul_weight, &gfbox);
        ////fprintf(stderr,"[Node %d]: %s ------ reducer ends! total_num_KM = %d -------\n", MPI_myId, myname, num_unique);
        ////mr->print(-1, 1, 5, 3);
        //MPI_Barrier(MPI_COMM_WORLD);
        ////int num_keywords2 = mr->map(mr, &MR_update_weight, &gfbox);
        ////fprintf(stderr,"[Node %d]: %s, mapper2 ends! -- num_keywords = %d! -------\n", MPI_myId, myname, num_keywords2);
        //////mr->print(-1, 1, 5, 3);
        ///*
         //* gather(MPI_nProcs):
         //* MPI_nProcs CAN BE 1 OR ANY NUMBER SMALLER THAN P, THE TOTAL
         //* NUMBER OF PROCESSORS. THE GATHERING IS DONE TO THE LOWEST ID
         //* PROCESSORS, FROM 0 TO MPI_nProcs-1. PROCESSORS WITH ID >= MPI_nProcs
         //* END UP WITH AN EMPTY KEYVALUE OBJECT CONTAINING NO KEY/VALUE
         //* PAIRS.
         //*/
        //mr->gather(1);
        ////mr->print(-1, 1, 5, 3);
        //int num_keywords2 = mr->map(mr, &MR_update_weight, &gfbox);
        ////fprintf(stderr,"[Node %d]: %s, mapper2 ends! -- num_keywords = %d! -------\n", MPI_myId, myname, num_keywords2);
        ////mr->print(-1, 1, 5, 3);
        ////double tstop = MPI_Wtime();
    //}
    ////else {
    ///* IN BATCH MODE, THERE IS NO LEARNING RATE FACTOR. THUS NOTHING
     //* IS UPDATED WHEN R <= 1.0.
     //* OR
     //* COMPUTE neighbor_fuct * PREVIOUS_WEIGHT AND SET THE WEIGHT AS
     //* NEW WEGITH, BECAUSE 1/neighbor_fuct ACTS LIKE LEARNING RATE
     //* FACTOR IN BATCH MODE?
     //* LET'S TRY THE LATTER CASE.
     //*
     //* I HAVE TRIED THE LATTER BUT NOT WORKING. 07.13.2010
     //* SEE SOM.CPP FOR DETAILS.
     //*/
    ////}
//}

///* ------------------------------------------------------------------------ */
//void MR_train_batch(MapReduce *mr, SOM *som, DMatrix &f, DMatrix &w,
                    //int *loc, int *scattered,
                    //float R, int argc, char* argv[], int MPI_myId,
                    //char *myname, int MPI_nProcs)
///* ------------------------------------------------------------------------ */
//{
    //if (R > 1.0f) {
        //int new_nvecs = NVECS / MPI_nProcs; /// CHUNNK SIZE
        //GIFTBOX gfbox;
        //gfbox.som = som;
        //gfbox.new_nvecs = new_nvecs;
        //gfbox.f_vectors = &f;
        ////gfbox.w_vectors = &w;
        ////gfbox.numer = &numer_vec;
        ////gfbox.denom = &denom_vec;
        //gfbox.R = R;
        //gfbox.idx_start = scattered[0];
        //gfbox.loc = loc;
        ////double tstart = MPI_Wtime();
        //MPI_Barrier(MPI_COMM_WORLD);
        ////fprintf(stderr,"[Node %d]: %s train_batch_MR FEATURES %0.2f %0.2f %0.2f **** \n", MPI_myId, myname, f.rows[0][0],f.rows[0][1],f.rows[0][2]);
        //int num_keywords = mr->map(MPI_nProcs, &MR_compute_weight, &gfbox);
        ////fprintf(stderr,"[Node %d]: %s, mapper1 ends! -- num_keywords = %d! -------\n", MPI_myId, myname, num_keywords);
        ////mr->print(-1, 1, 5, 3);
        //int total_num_KM = mr->collate(NULL);     /// AGGREGATES A KEYVALUE OBJECT ACROSS PROCESSORS AND CONVERTS IT INTO A KEYMULTIVALUE OBJECT.
        ////int total_num_KM = mr->clone();         /// CONVERTS A KEYVALUE OBJECT DIRECTLY INTO A KEYMULTIVALUE OBJECT.
        ////fprintf(stderr,"[Node %d]: %s ------ COLLATE ends! total_num_KM = %d -------\n", MPI_myId, myname, total_num_KM);
        ////mr->print(-1, 1, 5, 3);
        ////MPI_Barrier(MPI_COMM_WORLD);
        //int num_unique = mr->reduce(&MR_accumul_weight, (void *)NULL);
        //////int num_unique = mr->reduce(&accumul_weight, &gfbox);
        ////fprintf(stderr,"[Node %d]: %s ------ reducer ends! total_num_KM = %d -------\n", MPI_myId, myname, num_unique);
        ////mr->print(-1, 1, 5, 3);
        //MPI_Barrier(MPI_COMM_WORLD);
        ////int num_keywords2 = mr->map(mr, &MR_update_weight, &gfbox);
        ////fprintf(stderr,"[Node %d]: %s, mapper2 ends! -- num_keywords = %d! -------\n", MPI_myId, myname, num_keywords2);
        //////mr->print(-1, 1, 5, 3);
        ///*
         //* gather(MPI_nProcs):
         //* MPI_nProcs CAN BE 1 OR ANY NUMBER SMALLER THAN P, THE TOTAL
         //* NUMBER OF PROCESSORS. THE GATHERING IS DONE TO THE LOWEST ID
         //* PROCESSORS, FROM 0 TO MPI_nProcs-1. PROCESSORS WITH ID >= MPI_nProcs
         //* END UP WITH AN EMPTY KEYVALUE OBJECT CONTAINING NO KEY/VALUE
         //* PAIRS.
         //*/
        //mr->gather(1);
        ////mr->print(-1, 1, 5, 3);
        //int num_keywords2 = mr->map(mr, &MR_update_weight, &gfbox);
        ////fprintf(stderr,"[Node %d]: %s, mapper2 ends! -- num_keywords = %d! -------\n", MPI_myId, myname, num_keywords2);
        ////mr->print(-1, 1, 5, 3);
        ////double tstop = MPI_Wtime();
    //}
    ////else {
    ///* IN BATCH MODE, THERE IS NO LEARNING RATE FACTOR. THUS NOTHING
     //* IS UPDATED WHEN R <= 1.0.
     //* OR
     //* COMPUTE neighbor_fuct * PREVIOUS_WEIGHT AND SET THE WEIGHT AS
     //* NEW WEGITH, BECAUSE 1/neighbor_fuct ACTS LIKE LEARNING RATE
     //* FACTOR IN BATCH MODE?
     //* LET'S TRY THE LATTER CASE.
     //*
     //* I HAVE TRIED THE LATTER BUT NOT WORKING. 07.13.2010
     //* SEE SOM.CPP FOR DETAILS.
     //*/
    ////}
//}

///* ------------------------------------------------------------------------ */
//void MR_compute_weight2(int itask, char *file, KeyValue *kv, void *ptr)
///* ------------------------------------------------------------------------ */
//{
    //GIFTBOX *gb = (GIFTBOX *) ptr;
    ////DMatrix f = *(gb->f_vectors);
    //VVV_FLOAT_T numer;    
    //numer = VVV_FLOAT_T(NVECS, vector<vector<float> > (NNODES,
                        //vector<float>(NDIMEN, 0.0)));
    //VVV_FLOAT_T denom;
    //denom = VVV_FLOAT_T(NVECS, vector<vector<float> > (NNODES,
                        //vector<float>(NDIMEN, 0.0)));
                      
                      
    //string workItem(file);
    ////cout << "workItem = " << workItem << endl;
    //FILE *fp;   
    //fp = fopen(file, "r");
    //DMatrix f;
    //f = initMatrix();
    //f = createMatrix(NVECS, NDIMEN);
    //if (!validMatrix(f)) {
        //printf("FATAL: not valid data matrix.\n");
        //exit(0);
    //}
    //for (uint64_t row = 0; row < gb->new_nvecs; row++) { 
        //for (uint64_t col = 0; col < NDIMEN; col++) {
            //float tmp = 0.0f;
            //fscanf(fp, "%f", &tmp);
            //f.rows[row][col] = tmp;
        //}
    //}           
    //fclose(fp);            
    
    
    //for (int n = 0; n < gb->new_nvecs; n++) {
        ////float *normalized = normalize(f, gb->idx_start+n); //n-th feature vector in the scatered list.
        //float *normalized = normalize(f, gb->loc[gb->idx_start+n]); //n-th feature vector in the scatered list.
        ////printf("%d %d %d %d \n", gb->idx_start, n, gb->idx_start+n, gb->loc[gb->idx_start+n]);
        
        ////bottleneck///////////////////////////////////////////////////////
        ///// GET THE BEST MATCHING UNIT
        //NODE *bmu_node = get_bmu(gb->som, normalized);
        ///// BOTTLENECK////////////////////////////////////////////////////
        
        ///// GET THE COORDS FOR THE BMU
        //const float *p1 = get_coords(bmu_node);
        
        //for (int k = 0; k < NNODES; k++) {
            //NODE *tp = gb->som->nodes[k];
            //const float *p2 = get_coords(tp);
            //float dist = 0.0f;
            //for (int p = 0; p < NDIMEN; p++)
                //dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
            //dist = sqrt(dist);
            //float neighbor_fuct = 0.0f;
            //neighbor_fuct = exp(-(1.0f * dist * dist) / (gb->R * gb->R));
            //for (int w = 0; w < NDIMEN; w++) {
                ////(*numer)[n][k][w] += 1.0f * neighbor_fuct * normalized[w];
                ////(*denom)[n][k][w] += neighbor_fuct;
                //numer[n][k][w] += 1.0f * neighbor_fuct * normalized[w];
                //denom[n][k][w] += neighbor_fuct;
            //}
            //delete p2;
        //}
        //delete normalized;
        //delete p1;
    //}
    ////for (int n = 0; n < gb->new_nvecs; n++)
    ////for (int k = 0; k < gb->num_som_nodes; k++)
    ////printf("%f %f %f\n", numer[n][k][0], numer[n][k][1], numer[n][k][2]);
    ////printf("\n");
    ////for (int n = 0; n < gb->new_nvecs; n++)
    ////for (int k = 0; k < gb->num_som_nodes; k++)
    ////printf("%f %f %f\n", numer[n][k][0], numer[n][k][1], numer[n][k][2]);
    ///// UPDATE codebook-DIMENSINAL WEIGHTS FOR EACH NODE
    ////float *sum_numer = (float *)malloc(SZFLOAT * gb->num_weights_per_node);
    ////float *sum_demon = (float *)malloc(SZFLOAT * gb->num_weights_per_node);
    //float sum_numer[NDIMEN];
    //float sum_demon[NDIMEN];
    //for (int k = 0; k < NNODES; k++) {
        //for (int w = 0; w < NDIMEN; w++) {
            //float temp_numer = 0.0f;
            //float temp_demon = 0.0f;
            //for (int n = 0; n < gb->new_nvecs; n++) {
                //temp_numer += numer[n][k][w];
                //temp_demon += denom[n][k][w];
            //}
            //sum_numer[w] = temp_numer; /// LOCAL SUM VECTOR FOR K-TH NODE
            //sum_demon[w] = temp_demon; /// LOCAL SUM VECTOR FOR K-TH NODE
        //}
        //for (int w = 0; w < NDIMEN; w++) {
            ////char weightnum[MAXSTR];
            ////sprintf(weightnum, "N %d %d", k, w);
            //////cout << weightnum << " " << sum_numer[w] << endl;
            //////nodes[k]->updatew_batch(new_weights);
            ////char bkey[strlen(weightnum)+1], bvalue[SZFLOAT], bvalue2[SZFLOAT];
            //////float f_key, f_value;
            ////memcpy(bkey, &weightnum, strlen(weightnum)+1);
            ////memcpy(bvalue, &sum_numer[w], SZFLOAT);
            ////kv->add(bkey, strlen(weightnum)+1, bvalue, SZFLOAT);
            ////sprintf(weightnum, "D %d %d", k, w);
            //////cout << weightnum << " " << sum_demon[w] << endl;
            ////memcpy(bkey, &weightnum, strlen(weightnum)+1);
            ////memcpy(bvalue2, &sum_demon[w], SZFLOAT);
            ////kv->add(bkey, strlen(weightnum)+1, bvalue2, SZFLOAT);
            //char weightnum[MAXSTR];
            //sprintf(weightnum, "%d %d", k, w);
            /////////////////////////////////////////////
            ///// SHOULD BE CAREFUL ON KEY BYTE ALIGN!
            /////////////////////////////////////////////
            //char bkey[strlen(weightnum)+1];
            //char bvalue[SZFLOAT];
            //char bvalue2[SZFLOAT];
            //char bconcat[SZFLOAT*2];
            ////char byte_value_cancat2[sizeof(int)*2];
            //memcpy(bkey, &weightnum, strlen(weightnum) + 1);
            //memcpy(bvalue, &sum_numer[w], SZFLOAT);
            //memcpy(bvalue2, &sum_demon[w], SZFLOAT);
            //for (int i = 0; i < (int)SZFLOAT; i++) {
                //bconcat[i] = bvalue[i];
                //bconcat[i+SZFLOAT] = bvalue2[i];
            //}//total 4*2 = 8bytes for two floats.
            
            ///// DEBUG
            ////printf("orig floats = %g, %g\n", sum_numer[w], sum_demon[w]);
            ////char *c1 = (char *)malloc(SZFLOAT);
            ////char *c2 = (char *)malloc(SZFLOAT);
            ////for (int i = 0; i < (int)SZFLOAT; i++) {
            ////c1[i] = bconcat[i];
            ////c2[i] = bconcat[i+SZFLOAT];
            ////}
            ////printf("parsed floats = %g, %g\n", *(float *)c1, *(float *)c2);
            //////////////////////////////////////////////////////////////
            //kv->add(bkey, strlen(weightnum) + 1, bconcat, SZFLOAT * 2);
            //////////////////////////////////////////////////////////////
            ////try encoded key ==> fail!
            ////char byte_value3[sizeof(int)];
            ////char byte_value4[sizeof(int)];
            ////memcpy(byte_value3, &k, sizeof(int));
            ////memcpy(byte_value4, &w, sizeof(int));
            ////for (int i = 0; i < (int)sizeof(int); i++) {
            ////byte_value_cancat2[i] = byte_value3[i];
            ////byte_value_cancat2[i+sizeof(int)] = byte_value4[i];
            ////}
            ////kv->add(byte_value_cancat2, sizeof(int)*2, bconcat, SZFLOAT*2);
        //}
    //}
    ////free(sum_numer);
    ////free(sum_demon);
    //numer.clear();
    //denom.clear();
    ////fprintf(stderr,"[Node %d]: %s end mapper **** \n", gb->MPI_myId, gb->myname);
//}

///* ------------------------------------------------------------------------ */
//void MR_compute_weight(int itask, KeyValue *kv, void *ptr)
///* ------------------------------------------------------------------------ */
//{
    //GIFTBOX *gb = (GIFTBOX *) ptr;
    //DMatrix f = *(gb->f_vectors);
    //VVV_FLOAT_T numer;
    
    //numer = VVV_FLOAT_T(gb->new_nvecs, vector<vector<float> > (NNODES,
                        //vector<float>(NDIMEN, 0.0)));
    //VVV_FLOAT_T denom;
    //denom = VVV_FLOAT_T(gb->new_nvecs, vector<vector<float> > (NNODES,
                        //vector<float>(NDIMEN, 0.0)));
                        
    //for (int n = 0; n < gb->new_nvecs; n++) {
        ////float *normalized = normalize(f, gb->idx_start+n); //n-th feature vector in the scatered list.
        //float *normalized = normalize(f, gb->loc[gb->idx_start+n]); //n-th feature vector in the scatered list.
        ////printf("%d %d %d %d \n", gb->idx_start, n, gb->idx_start+n, gb->loc[gb->idx_start+n]);
        
        ////bottleneck///////////////////////////////////////////////////////
        ///// GET THE BEST MATCHING UNIT
        //NODE *bmu_node = get_bmu(gb->som, normalized);
        ///// BOTTLENECK////////////////////////////////////////////////////
        
        ///// GET THE COORDS FOR THE BMU
        //const float *p1 = get_coords(bmu_node);
        
        //for (int k = 0; k < NNODES; k++) {
            //NODE *tp = gb->som->nodes[k];
            //const float *p2 = get_coords(tp);
            //float dist = 0.0f;
            //for (int p = 0; p < NDIMEN; p++)
                //dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
            //dist = sqrt(dist);
            //float neighbor_fuct = 0.0f;
            //neighbor_fuct = exp(-(1.0f * dist * dist) / (gb->R * gb->R));
            //for (int w = 0; w < NDIMEN; w++) {
                ////(*numer)[n][k][w] += 1.0f * neighbor_fuct * normalized[w];
                ////(*denom)[n][k][w] += neighbor_fuct;
                //numer[n][k][w] += 1.0f * neighbor_fuct * normalized[w];
                //denom[n][k][w] += neighbor_fuct;
            //}
            //delete p2;
        //}
        //delete normalized;
        //delete p1;
    //}
    ////for (int n = 0; n < gb->new_nvecs; n++)
    ////for (int k = 0; k < gb->num_som_nodes; k++)
    ////printf("%f %f %f\n", numer[n][k][0], numer[n][k][1], numer[n][k][2]);
    ////printf("\n");
    ////for (int n = 0; n < gb->new_nvecs; n++)
    ////for (int k = 0; k < gb->num_som_nodes; k++)
    ////printf("%f %f %f\n", numer[n][k][0], numer[n][k][1], numer[n][k][2]);
    ///// UPDATE codebook-DIMENSINAL WEIGHTS FOR EACH NODE
    ////float *sum_numer = (float *)malloc(SZFLOAT * gb->num_weights_per_node);
    ////float *sum_demon = (float *)malloc(SZFLOAT * gb->num_weights_per_node);
    //float sum_numer[NDIMEN];
    //float sum_demon[NDIMEN];
    //for (int k = 0; k < NNODES; k++) {
        //for (int w = 0; w < NDIMEN; w++) {
            //float temp_numer = 0.0f;
            //float temp_demon = 0.0f;
            //for (int n = 0; n < gb->new_nvecs; n++) {
                //temp_numer += numer[n][k][w];
                //temp_demon += denom[n][k][w];
            //}
            //sum_numer[w] = temp_numer; /// LOCAL SUM VECTOR FOR K-TH NODE
            //sum_demon[w] = temp_demon; /// LOCAL SUM VECTOR FOR K-TH NODE
        //}
        //for (int w = 0; w < NDIMEN; w++) {
            ////char weightnum[MAXSTR];
            ////sprintf(weightnum, "N %d %d", k, w);
            //////cout << weightnum << " " << sum_numer[w] << endl;
            //////nodes[k]->updatew_batch(new_weights);
            ////char bkey[strlen(weightnum)+1], bvalue[SZFLOAT], bvalue2[SZFLOAT];
            //////float f_key, f_value;
            ////memcpy(bkey, &weightnum, strlen(weightnum)+1);
            ////memcpy(bvalue, &sum_numer[w], SZFLOAT);
            ////kv->add(bkey, strlen(weightnum)+1, bvalue, SZFLOAT);
            ////sprintf(weightnum, "D %d %d", k, w);
            //////cout << weightnum << " " << sum_demon[w] << endl;
            ////memcpy(bkey, &weightnum, strlen(weightnum)+1);
            ////memcpy(bvalue2, &sum_demon[w], SZFLOAT);
            ////kv->add(bkey, strlen(weightnum)+1, bvalue2, SZFLOAT);
            //char weightnum[MAXSTR];
            //sprintf(weightnum, "%d %d", k, w);
            /////////////////////////////////////////////
            ///// SHOULD BE CAREFUL ON KEY BYTE ALIGN!
            /////////////////////////////////////////////
            //char bkey[strlen(weightnum)+1];
            //char bvalue[SZFLOAT];
            //char bvalue2[SZFLOAT];
            //char bconcat[SZFLOAT*2];
            ////char byte_value_cancat2[sizeof(int)*2];
            //memcpy(bkey, &weightnum, strlen(weightnum) + 1);
            //memcpy(bvalue, &sum_numer[w], SZFLOAT);
            //memcpy(bvalue2, &sum_demon[w], SZFLOAT);
            //for (int i = 0; i < (int)SZFLOAT; i++) {
                //bconcat[i] = bvalue[i];
                //bconcat[i+SZFLOAT] = bvalue2[i];
            //}//total 4*2 = 8bytes for two floats.
            
            ///// DEBUG
            ////printf("orig floats = %g, %g\n", sum_numer[w], sum_demon[w]);
            ////char *c1 = (char *)malloc(SZFLOAT);
            ////char *c2 = (char *)malloc(SZFLOAT);
            ////for (int i = 0; i < (int)SZFLOAT; i++) {
            ////c1[i] = bconcat[i];
            ////c2[i] = bconcat[i+SZFLOAT];
            ////}
            ////printf("parsed floats = %g, %g\n", *(float *)c1, *(float *)c2);
            //////////////////////////////////////////////////////////////
            //kv->add(bkey, strlen(weightnum) + 1, bconcat, SZFLOAT * 2);
            //////////////////////////////////////////////////////////////
            ////try encoded key ==> fail!
            ////char byte_value3[sizeof(int)];
            ////char byte_value4[sizeof(int)];
            ////memcpy(byte_value3, &k, sizeof(int));
            ////memcpy(byte_value4, &w, sizeof(int));
            ////for (int i = 0; i < (int)sizeof(int); i++) {
            ////byte_value_cancat2[i] = byte_value3[i];
            ////byte_value_cancat2[i+sizeof(int)] = byte_value4[i];
            ////}
            ////kv->add(byte_value_cancat2, sizeof(int)*2, bconcat, SZFLOAT*2);
        //}
    //}
    ////free(sum_numer);
    ////free(sum_demon);
    //numer.clear();
    //denom.clear();
    ////fprintf(stderr,"[Node %d]: %s end mapper **** \n", gb->MPI_myId, gb->myname);
//}

/* ------------------------------------------------------------------------ */
void updatew_online(NODE *node, float *vec, float Alpha_x_Hck)
/* ------------------------------------------------------------------------ */
{
    for (int w = 0; w < NDIMEN; w++)
        node->weights[w] += Alpha_x_Hck * (vec[w] - node->weights[w]);
}

/* ------------------------------------------------------------------------ */
void updatew_batch(NODE *node, float *new_w)
/* ------------------------------------------------------------------------ */
{
    for (int w = 0; w < NDIMEN; w++) {
        //if (new_w[w] > 0) //????
        node->weights[w] = new_w[w];
    }
}

/* ------------------------------------------------------------------------ */
float *normalize(DMatrix &f, int n, int normalizeopt)
/* ------------------------------------------------------------------------ */
{
    float *m_data = (float *)malloc(SZFLOAT * NDIMEN);
    switch (normalizeopt) {
    default:
    case 0: //NONE
        for (int x = 0; x < NDIMEN; x++) {
            m_data[x] = f.rows[n][x];
        }
        break;
    case 1: //MNMX
        //for (int x = 0; x < NDIMEN; x++)
        //m_data[x] = (0.9f - 0.1f) * (vec[x] + m_add[x]) * m_mul[x] + 0.1f;
        //break;
    case 2: //ZSCR
        //for (int x = 0; x < NDIMEN; x++)
        //m_data[x] = (vec[x] + m_add[x]) * m_mul[x];
        //break;
    case 3: //SIGM
        //for (int x = 0; x < NDIMEN; x++)
        //m_data[x] = 1.0f / (1.0f + exp(-((vec[x] + m_add[x]) * m_mul[x])));
        //break;
    case 4: //ENRG
        float energy = 0.0f;
        for (int x = 0; x < NDIMEN; x++)
            energy += f.rows[n][x] * f.rows[n][x];
        energy = sqrt(energy);
        for (int x = 0; x < NDIMEN; x++)
            m_data[x] = f.rows[n][x] / energy;
        break;
    }
    return m_data;
}

/* ------------------------------------------------------------------------ */
NODE *get_bmu(SOM *som, float *fvec)
/* ------------------------------------------------------------------------ */
{
    NODE *pbmu_node = som->nodes[0];
    float mindist = get_distance(fvec, pbmu_node->weights, DISTOPT);
    float dist;
    for (int x = 1; x < NNODES; x++) {
        if ((dist = get_distance(fvec, som->nodes[x]->weights, DISTOPT)) < mindist) {
            mindist = dist;
            pbmu_node = som->nodes[x];
        }
    }
    /// CAN ADD A FEATURE FOR VOTING AMONG BMUS.
    return pbmu_node;
}

/* ------------------------------------------------------------------------ */
float get_distance(float *vec, vector<float> &wvec, int distance_metric)
/* ------------------------------------------------------------------------ */
{
    float distance = 0.0f;
    float n1 = 0.0f, n2 = 0.0f;
    switch (distance_metric) {
    default:
    case 0: /// EUCLIDIAN
        //if (NDIMEN >= 4) {
        //distance = mse(vec, distance_metric, wvec);
        //} else {
        for (int w = 0; w < NDIMEN; w++)
            distance += (vec[w] - wvec[w]) * (vec[w] - wvec[w]);
        //}
        return sqrt(distance);
    case 1: /// SOSD: //SUM OF SQUARED DISTANCES
        //if (m_weights_number >= 4) {
        //distance = mse(vec, m_weights, m_weights_number);
        //} else {
        for (int w = 0; w < NDIMEN; w++)
            distance += (vec[w] - wvec[w]) * (vec[w] - wvec[w]);
        //}
        return distance;
    case 2: /// TXCB: //TAXICAB
        for (int w = 0; w < NDIMEN; w++)
            distance += fabs(vec[w] - wvec[w]);
        return distance;
    case 3: /// ANGL: //ANGLE BETWEEN VECTORS
        for (int w = 0; w < NDIMEN; w++) {
            distance += vec[w] * wvec[w];
            n1 += vec[w] * vec[w];
            n2 += wvec[w] * wvec[w];
        }
        return acos(distance / (sqrt(n1) * sqrt(n2)));
        //case 4: /// MHLN:   //mahalanobis
        //distance = sqrt(m_weights * cov * vec)
        //return distance
    }
}

/* ------------------------------------------------------------------------ */
float get_distance2(vector<float> &vec, vector<float> &wvec, int distance_metric)
/* ------------------------------------------------------------------------ */
{
    float distance = 0.0f;
    float n1 = 0.0f, n2 = 0.0f;
    switch (distance_metric) {
    default:
    case 0: /// EUCLIDIAN
        for (int w = 0; w < NDIMEN; w++)
            distance += (vec[w] - wvec[w]) * (vec[w] - wvec[w]);
        return sqrt(distance);
    case 1: /// SOSD: //SUM OF SQUARED DISTANCES
        //if (m_weights_number >= 4) {
        //distance = mse(vec, m_weights, m_weights_number);
        //} else {
        for (int w = 0; w < NDIMEN; w++)
            distance += (vec[w] - wvec[w]) * (vec[w] - wvec[w]);
        //}
        return distance;
    case 2: /// TXCB: //TAXICAB
        for (int w = 0; w < NDIMEN; w++)
            distance += fabs(vec[w] - wvec[w]);
        return distance;
    case 3: /// ANGL: //ANGLE BETWEEN VECTORS
        for (int w = 0; w < NDIMEN; w++) {
            distance += vec[w] * wvec[w];
            n1 += vec[w] * vec[w];
            n2 += wvec[w] * wvec[w];
        }
        return acos(distance / (sqrt(n1) * sqrt(n2)));
        //case 4: /// MHLN:   //mahalanobis
        //distance = sqrt(m_weights * cov * vec)
        //return distance
    }
}

/** MR-MPI Map function - Distance b/w vec1 and vec2, default: Euclidean
 * @param float *vec1
 * @param const float *vec2
 * @param int distance_metric: 
 */
/* ------------------------------------------------------------------------ */
float get_distance3(float *vec1, const float *vec2, int distance_metric)
/* ------------------------------------------------------------------------ */
{
    float distance = 0.0f;
    float n1 = 0.0f, n2 = 0.0f;
    switch (distance_metric) {
    default:
    case 0: /// EUCLIDIAN
        for (int w = 0; w < NDIMEN; w++)
            distance += (vec1[w] - vec2[w]) * (vec1[w] - vec2[w]);
        return sqrt(distance);
    //case 1: /// SOSD: //SUM OF SQUARED DISTANCES
        ////if (m_weights_number >= 4) {
        ////distance = mse(vec, m_weights, m_weights_number);
        ////} else {
        //for (int w = 0; w < NDIMEN; w++)
            //distance += (vec[w] - wvec[w]) * (vec[w] - wvec[w]);
        ////}
        //return distance;
    //case 2: /// TXCB: //TAXICAB
        //for (int w = 0; w < NDIMEN; w++)
            //distance += fabs(vec[w] - wvec[w]);
        //return distance;
    //case 3: /// ANGL: //ANGLE BETWEEN VECTORS
        //for (int w = 0; w < NDIMEN; w++) {
            //distance += vec[w] * wvec[w];
            //n1 += vec[w] * vec[w];
            //n2 += wvec[w] * wvec[w];
        //}
        //return acos(distance / (sqrt(n1) * sqrt(n2)));
    ////case 4: /// MHLN:   //mahalanobis
        ////distance = sqrt(m_weights * cov * vec)
        ////return distance
    }
}

///* ------------------------------------------------------------------------ */
//float *get_wvec(SOM *som, int n)
///* ------------------------------------------------------------------------ */
//{
    //float *weights = (float *)malloc(SZFLOAT * NDIMEN);
    //for (int i = 0; i < NDIMEN; i++)
        //weights[i] = som->nodes[n]->weights[i];
    //return weights;
//}

///* ------------------------------------------------------------------------ */
//float *get_coords(NODE *node)
///* ------------------------------------------------------------------------ */
//{
    //float *coords = (float *)malloc(SZFLOAT * SOM_D);
    //for (int i = 0; i < SOM_D; i++)
        //coords[i] = node->coords[i];
    //return coords;
//}

/* ------------------------------------------------------------------------ */
void get_file_name(char *path, char *name)
/* ------------------------------------------------------------------------ */
{
    int sl = 0, dot = (int)strlen(path);
    int i;
    for (i = 0; i < (int)strlen(path); i++) {
        if (path[i] == '.') break;
        if (path[i] == '\\') break;
    }
    if (i >= (int)strlen(path)) {
        strcpy(name, path);
        return;
    }
    for (i = (int)strlen(path) - 1; i >= 0; i--) {
        if (path[i] == '.')
            dot = i;
        if (path[i] == '\\') {
            sl = i + 1;
            break;
        }
    }
    memcpy(name, &path[sl], (dot - sl) * 2);
    name[dot-sl] = 0;
}

/* ------------------------------------------------------------------------ */
int save_umat(SOM *som, char *fname)
/* ------------------------------------------------------------------------ */
{
    int D = 2;
    float min_dist = 1.5f;
    FILE *fp = fopen(fname, "wt");
    if (fp != 0) {
        int n = 0;
        for (int i = 0; i < SOM_X; i++) {
            for (int j = 0; j < SOM_Y; j++) {
                float dist = 0.0f;
                int nodes_number = 0;
                NODE *pnode = get_node(som, n++);
                
                for (int m = 0; m < NNODES; m++) {
                    NODE *node = get_node(som, m);
                    if (node == pnode)
                        continue;
                    float tmp = 0.0;
                    //float *temp1 = get_coords(pnode);
                    //float *temp2 = get_coords(node);
                    float temp1[SOM_D];
                    float temp2[SOM_D];
                    temp1[0] = pnode->coords[0];
                    temp1[1] = pnode->coords[1];
                    temp2[0] = node->coords[0];
                    temp2[1] = node->coords[1];
                    for (int d = 0; d < D; d++) {
                        //cout << "1 " << *(get_coords(pnode)) << endl;
                        //cout << x << " " << *(get_coords(pnode) + x) << endl;
                        //tmp += pow(*(get_coords(pnode) + x) - *(get_coords(node) + x), 2.0f);
                        //tmp += pow(*(temp1 + x) - *(temp2 + x), 2.0f);
                        tmp += pow(temp1[d] - temp2[d], 2.0f);
                    }
                    //delete temp1;
                    //delete temp2;
                    tmp = sqrt(tmp);
                    if (tmp <= min_dist) {
                        nodes_number++;
                        dist += get_distance2(node->weights, pnode->weights, DISTOPT);
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

/* ------------------------------------------------------------------------ */
NODE *get_node(SOM *som, int n)
/* ------------------------------------------------------------------------ */
{
    return som->nodes[n];
}


///* ------------------------------------------------------------------------ */
//void MR_accumul_weight(char *key, int keybytes, char *multivalue,
                       //int nvalues, int *valuebytes, KeyValue *kv,
                       //void *ptr)
///* ------------------------------------------------------------------------ */
//{
    //GIFTBOX *gb = (GIFTBOX *) ptr;
    ///// A SZFLOAT*2-BYTE STRUCTURE FOR TWO FLOATS => TWO FLOAT VALUES
    //vector<float> vec_numer(nvalues, 0.0);
    //vector<float> vec_denom(nvalues, 0.0);
    //for (int j = 0; j < nvalues; j++) {
        //char *c1 = (char *)malloc(SZFLOAT);
        //char *c2 = (char *)malloc(SZFLOAT);
        //for (int i = 0; i < (int)SZFLOAT; i++) {
            //c1[i] = multivalue[i];
            //c2[i] = multivalue[i+SZFLOAT];
        //}
        ////printf("parsed floats = %g, %g\n", *(float *)c1, *(float *)c2);
        //vec_numer[j] = *(float *)c1;
        //vec_denom[j] = *(float *)c2;
        ////printf("%g ",*(float *) multivalue);
        //multivalue += SZFLOAT * 2;
        
        //delete c1;
        //delete c2;        
    //}
    ////printf("summed %s, %f, %f, %f, %f\n", key, vec_numer[0],vec_numer[0],vec_denom[0],vec_denom[0]);
    ////fprintf(stderr,"[Node %d]: %s reduce %s, %f, %f \n", gb->MPI_myId, gb->myname, key, vec_numer[0],vec_numer[0],vec_denom[0],vec_denom[0]);
    
    ///// New weights for node K and dimension D
    ////cout << accumulate(vec_numer.begin(), vec_numer.end(), 0.0f) << " ";
    ////cout << accumulate(vec_denom.begin(), vec_denom.end(), 0.0f) << endl;
    //float temp_numer = accumulate(vec_numer.begin(), vec_numer.end(), 0.0f);
    //float temp_demon = accumulate(vec_denom.begin(), vec_denom.end(), 0.0f);
    //float new_weight = 0.0f;
    //if (temp_demon != 0)
        //new_weight = temp_numer / temp_demon;
    ////float new_weight = accumulate(vec_numer.begin(), vec_numer.end(), 0.0f ) /
    ////accumulate(vec_denom.begin(), vec_denom.end(), 0.0f );
    ////nodes[K]->updatew_batch_index(new_weight, codebook);
    ////(*gb->nodes)[K]->updatew_batch_index(new_weight, codebook);
    ////gb->new_weights[K][codebook] = new_weight;
    //char bvalue[SZFLOAT];
    //memcpy(bvalue, &new_weight, SZFLOAT);
    //////////////////////////////////////////
    //kv->add(key, keybytes, bvalue, SZFLOAT);
    //////////////////////////////////////////
    ////(*gb->nodes)[K]->updatew_batch_index(new_weight, codebook);
    ////gb->weight_vectors[K][codebook] = new_weight;
    //vec_numer.clear();
    //vec_denom.clear();
    ////fprintf(stderr,"[Node %d]: %s, %s, %d, %d \n", gb->MPI_myId, gb->myname, key, keybytes, nvalues);
    ////kv->add(key, keybytes, (char *) &nvalues, sizeof(int));
//}

///* ------------------------------------------------------------------------ */
//void MR_update_weight(uint64_t itask, char *key, int keybytes, char *value,
                      //int valuebytes, KeyValue *kv, void *ptr)
///* ------------------------------------------------------------------------ */
//{
    //GIFTBOX *gb = (GIFTBOX *) ptr;
    //char *whitespace = " \t\n\f\r\0";
    ////char *whitespace = " ";
    //char *key_tokens = strtok(key, whitespace);
    //int K = atoi(key_tokens);
    //key_tokens = strtok(NULL, whitespace);
    //int codebook = atoi(key_tokens);
    //float new_weight = *(float *)value;
    //updatew_batch_index(gb->som->nodes[K], new_weight, codebook); /// UPDATE WEIGHT OF NODE K
//}

/* ------------------------------------------------------------------------ */
void updatew_batch_index(NODE *node, float new_weight, int w)
/* ------------------------------------------------------------------------ */
{
    if (new_weight > 0)
        node->weights[w] = new_weight;
}


/* ------------------------------------------------------------------------ */
DMatrix createMatrix(const unsigned int rows, const unsigned int cols)
/* ------------------------------------------------------------------------ */
{
    DMatrix matrix;
    unsigned long int m, n;
    unsigned int i;
    m = rows;
    n = cols;
    matrix.m = rows;
    matrix.n = cols;
    matrix.data = (float *) malloc(sizeof(float) * m * n);
    matrix.rows = (float **) malloc(sizeof(float *) * m);
    if (validMatrix(matrix)) {
        matrix.m = rows;
        matrix.n = cols;
        for (i = 0; i < rows; i++) {
            matrix.rows[i] = matrix.data + (i * cols);
        }
    }
    else {
        freeMatrix(&matrix);
    }
    return matrix;
}


/* ------------------------------------------------------------------------ */
void freeMatrix(DMatrix *matrix)
/* ------------------------------------------------------------------------ */
{
    if (matrix == NULL) return;
    if (matrix -> data) {
        free(matrix -> data);
        matrix -> data = NULL;
    }
    if (matrix -> rows) {
        free(matrix -> rows);
        matrix -> rows = NULL;
    }
    matrix -> m = 0;
    matrix -> n = 0;
}


/* ------------------------------------------------------------------------ */
int validMatrix(DMatrix matrix)
/* ------------------------------------------------------------------------ */
{
    if ((matrix.data == NULL) || (matrix.rows == NULL) ||
            (matrix.m == 0) || (matrix.n == 0))
        return 0;
    else return 1;
}


/* ------------------------------------------------------------------------ */
DMatrix initMatrix()
/* ------------------------------------------------------------------------ */
{
    DMatrix matrix;
    matrix.m = 0;
    matrix.n = 0;
    matrix.data = NULL;
    matrix.rows = NULL;
    return matrix;
}


/* ------------------------------------------------------------------------ */
void printMatrix(DMatrix A)
/* ------------------------------------------------------------------------ */
{
    unsigned int i, j;
    if (validMatrix(A)) {
        for (i = 0; i < A.m; i++) {
            for (j = 0; j < A.n; j++) printf("%7.3f ", A.rows[i][j]);
            printf("\n");
        }
    }
}

/* ------------------------------------------------------------------------ */
string uint2str(uint64_t number)
/* ------------------------------------------------------------------------ */
{
    stringstream ss;
    ss << number;
    return ss.str();
}

/* ------------------------------------------------------------------------ */
uint64_t str2uint(string str)
/* ------------------------------------------------------------------------ */
{
    std::stringstream ss;
    ss << str;
    uint64_t f;
    ss >> f;
    return f;
}

/* ------------------------------------------------------------------------ */
string float2str(float number)
/* ------------------------------------------------------------------------ */
{
    stringstream ss;
    ss << number;
    return ss.str();
}

/* ------------------------------------------------------------------------ */
float str2float(string str)
/* ------------------------------------------------------------------------ */
{
    std::stringstream ss;
    ss << str;
    float f;
    ss >> f;
    return f;
}

/// TOKENIZER ROUTINES
/* ------------------------------------------------------------------------ */
vector<string> &split(const string &s, char delim, vector<string> &vecElems)
/* ------------------------------------------------------------------------ */
{
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        vecElems.push_back(item);
    }
    return vecElems;
}

/* ------------------------------------------------------------------------ */
vector<string> split(const string &s, char delim)
/* ------------------------------------------------------------------------ */
{
    vector<string> vecElems;
    return split(s, delim, vecElems);
}

///* ------------------------------------------------------------------------ */
//float mse(float *vec1, vector<float> &vec2, int size)
///* ------------------------------------------------------------------------ */
//{
//float z = 0.0f, fres = 0.0f;
//float ftmp[4];
//__m128 mv1, mv2, mres;
//mres = _mm_load_ss(&z);

//for (int i = 0; i < size / 4; i++) {
//mv1 = _mm_loadu_ps(&vec1[4*i]);
//mv2 = _mm_loadu_ps(&vec2[4*i]);
//mv1 = _mm_sub_ps(mv1, mv2);
//mv1 = _mm_mul_ps(mv1, mv1);
//mres = _mm_add_ps(mres, mv1);
//}
//if (size % 4) {
//for (int i = size - size % 4; i < size; i++)
//fres += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
//}

////mres = a,b,c,d
//mv1 = _mm_movelh_ps(mres, mres);   //a,b,a,b
//mv2 = _mm_movehl_ps(mres, mres);   //c,d,c,d
//mres = _mm_add_ps(mv1, mv2);       //res[0],res[1]

//_mm_storeu_ps(ftmp, mres);

//return fres + ftmp[0] + ftmp[1];
//}


/* ------------------------------------------------------------------------ */
//void mr_train_batch(SOM *som, DMatrix &f, float R)
/* ------------------------------------------------------------------------ */
/*{
    float *numer = (float *)malloc(NDIMEN * SZFLOAT);
    float *denom = (float *)malloc(NDIMEN * SZFLOAT);
    float *new_weights = (float *)malloc(NDIMEN * SZFLOAT);

    for (int k = 0; k < NNODES; k++) {  //ADJUST WEIGHTS OF ALL K NODES IF DISTOPT <= R
        const float *p2 = get_coords(som->nodes[k]);
        float neighbor_fuct=0.0f;
        for (int w = 0; w < NDIMEN; w++) {
            numer[w] = 0.0f;
            denom[w] = 0.0f;
            new_weights[w] = 0.0f;
        }
        for (int n = 0; n < NVECS; n++) {
            //COMPUTE DISTANCES AND SELECT WINNING NODE (BMU)
            float *normalized = normalize(f, n);
            //GET BEST NODE USING d_k (t) = || x(t) = w_k (t_0) || ^2
            //AND d_c (t) == min d_k (t)
            //FIRST BMU IS SELECTED. NO VOTING.
//bottleneck///////////////////////////////////////////////
            NODE *bmu_node = get_bmu(som, normalized);
//bottleneck///////////////////////////////////////////////
            const float *p1 = get_coords(bmu_node);
            float dist = 0.0f;
            //dist = sqrt((x1-y1)^2 + (x2-y2)^2 + ...)  distance to node
            for (int p = 0; p < NDIMEN; p++)
                dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
            dist = sqrt(dist);
            if (TRAINOPT == 1 && dist > R)
                continue;
            //GAUSSIAN NEIGHBORHOOD FUNCTION
            neighbor_fuct = exp(-(1.0f * dist * dist) / (R * R));
            //NEW WEIGHT, w_k(t_f) = (SUM(h_ck (t') * x(t')) / (SUM()h_ck(t'))
            for (int w = 0; w < NDIMEN; w++) {
                numer[w] += 1.0f * neighbor_fuct * normalized[w];
                denom[w] += neighbor_fuct;
            }
            for (int w = 0; w < NDIMEN; w++)
                if (denom[w] != 0)
                    new_weights[w] = numer[w] / denom[w];
        }
        //UPDATE WEIGHTS
        updatew_batch(som->nodes[k], new_weights);
    }
    free(numer);
    free(denom);
    free(new_weights);
}*/

/* ------------------------------------------------------------------------ */
//void train_batch_serial(SOM* som, DMatrix &f, float R)
//VVV_FLOAT_T &numer, VVV_FLOAT_T &denom)
/* ------------------------------------------------------------------------ */ /*
{
    VVV_FLOAT_T numer = VVV_FLOAT_T (NVECS, vector<vector<float> > (NNODES,
                                   vector<float>(NDIMEN, 0.0)));
    VVV_FLOAT_T denom = VVV_FLOAT_T (NVECS, vector<vector<float> > (NNODES,
                                     vector<float>(NDIMEN, 0.0)));

    for (int n = 0; n < NVECS; n++) {
        //printf("orig       %f %f %f \n", FEATURE[n][0],FEATURE[n][1],FEATURE[n][2]);
        float *normalized = normalize(f, n);
        //printf("normalized %f %f %f \n", normalized[0],normalized[1],normalized[2]);
        NODE *bmu_node = get_bmu(som, normalized);
        //cout << bmu_node->coords[0] << " " << bmu_node->coords[1] << endl;
        const float *p1 = get_coords(bmu_node);
        //printf("coord 1 %f %f \n", p1[0], p1[1]);
        for (int k = 0; k < NNODES; k++) {
            const float *p2 = get_coords(som->nodes[k]);
            //printf("coord 2 %f %f \n", p2[0], p2[1]);
            float dist = 0.0f;
            for (int p = 0; p < NDIMEN; p++)
                dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
            dist = sqrt(dist);
            float neighbor_fuct = exp(-(1.0f * dist * dist) / (R * R));
            for (int w = 0; w < NDIMEN; w++) {
                numer[n][k][w] += 1.0f * neighbor_fuct * normalized[w];
                denom[n][k][w] += neighbor_fuct;
            }
        }
    }
    //UPDATE codebook-DIMENSINAL WEIGHTS FOR EACH NODE
    float *new_weights = (float *)malloc(NDIMEN * SZFLOAT);
    for (int i = 0; i < NDIMEN; i++)
        new_weights[i] = 0.0f;
    for (int k = 0; k < NNODES; k++) {
        for (int w = 0; w < NDIMEN; w++) {
            float temp_numer = 0.0f;
            float temp_demon = 0.0f;
            for (int n = 0; n < NVECS; n++) {
                temp_numer += numer[n][k][w];
                temp_demon += denom[n][k][w];
                //printf("temp_numer, temp_demon = %f %f\n", temp_numer, temp_demon);
            }
            if (temp_demon != 0)
                new_weights[w] = temp_numer / temp_demon;
            else if (temp_numer != 0)
                new_weights[w] = temp_numer;
            else {
                //printf("temp_numer temp_demon = %0.2f %0.2f\n", temp_numer, temp_demon);
                new_weights[w] = 0.0f;
            }
        }
        //printf("%f %f %f\n", som->nodes[k]->weights[0], som->nodes[k]->weights[1], som->nodes[k]->weights[2]);
        //printf("%f %f %f\n", new_weights[0], new_weights[1], new_weights[2]);
        updatew_batch(som->nodes[k], new_weights);
        //printf("%f %f %f\n", som->nodes[k]->weights[0], som->nodes[k]->weights[1], som->nodes[k]->weights[2]);
    }
    free(new_weights);
    numer.clear();
    denom.clear();
}*/


///* ------------------------------------------------------------------------ */
//float sqroot(float m)
///* ------------------------------------------------------------------------ */
//{
//float i = 0;
//float x1, x2;
//while ( (i*i) <= m ) i+=0.1;
//x1=i;
//for (int j = 0; j < 10; j++) {
//x2=m;
//x2/=x1;
//x2+=x1;
//x2/=2;
//x1=x2;
//}
//return x2;
//}
