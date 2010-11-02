/// MPI and MapReduce-MPI
#include "mpi.h"
#include "./mrmpi/mapreduce.h"
#include "./mrmpi/keyvalue.h"

//#include <stdlib.h>
//#include <stdio.h>
#include <math.h>
#include <assert.h>

#include <vector>
//#include <iostream>
//#include <numeric>

/// For split
#include <sstream>

/// For save
#include <fstream>

/// For timing
//#include <sys/time.h>

using namespace MAPREDUCE_NS;
using namespace std;

#define _DEBUG
#define SZFLOAT sizeof(float)
#define MAXSTR 255

enum TRAINTYPE      { BATCH, ONLINE };
enum DISTTYPE       { EUCL, SOSD, TXCB, ANGL, MHLN };
enum NORMALIZETYPE  { NONE, MNMX, ZSCR, SIGM, ENRG };
enum TRAINSPEED     { SLOW, FAST };

/// GLOBALS
int NDIMEN = 0;             /// NUM OF DIMENSIONALITY
int SOM_X = 50;
int SOM_Y = 50;
int SOM_D = 2;              /// 2=2D
int NNODES = SOM_X*SOM_Y;   /// TOTAL NUM OF SOM NODES
int NEPOCHS;                /// ITERATIONS
int DISTOPT = EUCL;         /// 0=EUCL, 1=SOSD, 2=TXCB, 3=ANGL, 4=MHLN
int TRAINMODE = 0;          /// 0=BATCH, 1=ONLINE
int TRAINOPT = 0;           /// 0=SLOW, 1=FAST
int NORMALOPT = NONE;       /// 0=NONE, 1=MNMX, 2=ZSCR, 3=SIGM, 4=ENRG
int SHUFFLE = 0;            /// 0=no shuffling of input vector, 1=shuffle
uint64_t NVECSPERFILE = 0;  /// NUM OF FEATURE VECTORS per file

/// Matrix
typedef struct {
    uint64_t m, n;          /// ROWS, COLS
    float *data;            /// DATA, ORDERED BY ROW, THEN BY COL
    float **rows;           /// POINTERS TO ROWS IN DATA
} DMatrix;

/// Matrix mani functions
DMatrix createMatrix(const unsigned int rows, const unsigned int cols);
DMatrix initMatrix(void);
void    freeMatrix(DMatrix *matrix);
void    printMatrix(DMatrix A);
int     validMatrix(DMatrix matrix);

struct GIFTBOX {
    float r;
    const DMatrix *codebook;
};

typedef vector<vector<vector<float> > > VVV_FLOAT_T;

/// MR-MPI map fuction and related functions
void train_batch(int itask, char *file, KeyValue *kv, void *ptr);
float *normalize(float *f, int normalopt);
float *get_bmu_coord(const DMatrix *codebook, const float *fvec);
float get_distance(float *vec1, const float *vec2, int distance_metric);
float *get_wvec(unsigned int somx, unsigned int somy, const DMatrix *codebook);
void update_weight(uint64_t itask, char *key, int keybytes, char *value, int valuebytes, KeyValue *kv, void *ptr);

/// MR-MPI reduce function
void sum(char *key, int keybytes, char *multivalue, int nvalues, int *valuebytes, KeyValue *kv, void *ptr);

/// Util
string float2str(float number);
float str2float(string str);
string uint2str(uint64_t number);
uint64_t str2uint(string str);

/// Tokenizer routines
vector<string>  &split(const string &s, char delim, vector<string> &vecElems);
vector<string>  split(const string &s, char delim);

/// Save U-matrix
int save_2D_distance_map(DMatrix *codebook, char *fname);

/// Online training
void train_online(char* file, DMatrix *codebook, float R, float Alpha);

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    if (argc == 6) {         
        /// syntax: mrsom FILE NEPOCHS TRAINMODE NVECSPERFILE NDIMEN
        NEPOCHS = atoi(argv[2]);
        TRAINMODE = atoi(argv[3]);
        NVECSPERFILE = atoi(argv[4]);
        NDIMEN = atoi(argv[5]);
    }
    else if (argc == 8) {  
        /// syntax: mrsom FILE NEPOCHS TRAINMODE NVECSPERFILE NDIMEN SOMX SOMY
        NEPOCHS = atoi(argv[2]);
        TRAINMODE = atoi(argv[3]);
        NVECSPERFILE = atoi(argv[4]);
        NDIMEN = atoi(argv[5]);
        SOM_X = atoi(argv[6]);
        SOM_Y = atoi(argv[7]);
        NNODES = SOM_X * SOM_Y;
    }
    else {
        printf("    mrsom FILE NEPOCHS TRAINMODE NVECSPERFILE NDIMEN [X Y]\n\n");
        printf("    FILE    = master file.\n");
        printf("    NEPOCHS = number of iterations.\n");
        printf("    TRAINMODE = 0-batch, 1-online.\n");
        printf("    NVECSPERFILE = number of feature vectors.\n");
        printf("    NDIMEN  = number of dimensionality of feature vector.\n");
        printf("    [X Y]   = optional, SOM map size. Default = [50 50]\n");
        exit(0);
    }
    
    ///
    /// Create codebook
    /// 
    ///   dimension
    /// |-----------|
    /// (1d 2d 3d...d), (1d 2d 3d..d), ..., (1d 2d 3d..d)  -
    /// (1d 2d 3d...d), (1d 2d 3d..d), ..., (1d 2d 3d..d)  |
    /// ...                                                |  SOM_Y
    /// (1d 2d 3d...d), (1d 2d 3d..d), ..., (1d 2d 3d..d)  -
    /// |-----------------------------------------------|
    ///                      SOM_X
    ///
    DMatrix codebook;
    codebook = initMatrix();
    codebook = createMatrix(SOM_Y, SOM_X * NDIMEN);
    if (!validMatrix(codebook)) {
        printf("FATAL: not valid codebook matrix.\n");
        exit(0);
    }
    
    ///
    /// Fill initial random weights
    ///
    srand((unsigned int)time(0));
    for (int som_y = 0; som_y < SOM_Y; som_y++) {        
        for (int som_x = 0; som_x < SOM_X; som_x++) {
            for (int d = 0; d < NDIMEN; d++) {
                int w = 0xFFF & rand();
                w -= 0x800;
                codebook.rows[som_y][som_x*NDIMEN + d] = (float)w / 4096.0f;
            }
        }
    }
    //printMatrix(codebook);
    
    /// MPI
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
    mr->verbosity = 0;
    mr->timer = 0;
    mr->mapstyle = 2;  /// master/slave mode
    MPI_Barrier(MPI_COMM_WORLD);
    
    ///
    /// Parameters for SOM
    ///
    float N = (float)NEPOCHS;       /// iterations
    float nrule, nrule0 = 0.9f;     /// learning rate factor
    float R, R0;
    R0 = SOM_X / 2.0f;              /// init radius for updating neighbors
    R = R0;
    int x = 0;                      /// 0...N-1
    
    ///
    /// Training
    ///
    if (TRAINMODE == ONLINE) {
        while (NEPOCHS) {
            if (MPI_myId == 0) {
                R = R0 * exp(-10.0f * (x * x) / (N * N));
                nrule = nrule0 * exp(-10.0f * (x * x) / (N * N));  
                x++;
                train_online(argv[1], &codebook, R, nrule); 
                printf("ONLINE-  epoch: %d   R: %.2f \n", (NEPOCHS - 1), R);
            }
            NEPOCHS--;
        }
    }
    else if (TRAINMODE == BATCH) {
        while (NEPOCHS && R > 1.0) {
            if (MPI_myId == 0) {
                R = R0 * exp(-10.0f * (x * x) / (N * N));
                x++;
                printf("### BATCH-  epoch: %d   R: %.2f \n", (NEPOCHS - 1), R);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast(&R, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast((void *)codebook.data, SOM_Y * SOM_X * NDIMEN, MPI_FLOAT, 0,
                      MPI_COMM_WORLD);
            
            GIFTBOX gf;
            gf.r = R;
            gf.codebook = &codebook;
            
            uint64_t nRes = mr->map(argv[1], &train_batch, &gf);
            //mr->print(-1, 1, 5, 5);
            mr->collate(NULL);
            //mr->print(-1, 1, 5, 5);
            nRes = mr->reduce(&sum, NULL);
            //cout << "nRes = " << nRes << endl;
            //mr->print(-1, 1, 5, 5);            
            mr->gather(1);
            //mr->print(0, 1, 5, 5);
            nRes = mr->map(mr, &update_weight, &gf);
            //cout << "nRes = " << nRes << endl;
            
            
            MPI_Barrier(MPI_COMM_WORLD); 

            NEPOCHS--;
        }  
        //cout << "MPI_myId = " << MPI_myId << endl;
        //printMatrix(codebook);
    }
    
    ///
    /// Save SOM map in a file
    ///
    if (MPI_myId == 0) {
        printf("Saving SOM map and U-Matrix...\n");
        char *outFileName = "result.umat.txt";
        save_2D_distance_map(&codebook, outFileName);
        system("python ./show2.py result.umat.txt");
        
        ///
        /// Save SOM map for umat tool
        /// Usage: /tools/umat -cin result.map > result.eps
        ///
        char *outFileName2 = "result.map.txt";
        ofstream mapFile(outFileName2);
        char temp[80];
        if (mapFile.is_open()) {
            mapFile << NDIMEN << " rect " << SOM_X << " " << SOM_Y << endl;
            for (uint64_t som_y = 0; som_y < SOM_Y; som_y++) { 
                for (uint64_t som_x = 0; som_x < SOM_X; som_x++) { 
                    for (uint64_t d = 0; d < NDIMEN; d++) {
                        //mapFile << codebook.rows[som_y][SOM_X*NDIMEN+i] << " ";
                        sprintf(temp, "%0.10f", codebook.rows[som_y][som_x*NDIMEN + d]);
                        mapFile << temp << " ";
                    }
                    mapFile << endl;
                }
            }
        }
        mapFile.close();
        system("./umat -cin result.map.txt > test.eps");
        printf("Done!\n");
    }
    
    freeMatrix(&codebook);
    delete mr;
    MPI::Finalize();
    return 0;
}
 
/* ------------------------------------------------------------------------ */
void train_online(char* file, DMatrix *codebook, float R, float Alpha)
/* ------------------------------------------------------------------------ */
{   
    ///
    /// Read feature chunk file
    ///
    //string workItem(file);
    //cout << "workItem = " << workItem << endl;
    FILE *fp;   
    fp = fopen(file, "r");
    DMatrix data;
    data = initMatrix();
    data = createMatrix(NVECSPERFILE, NDIMEN);
    if (!validMatrix(data)) {
        printf("FATAL: not valid data matrix.\n");
        exit(0);
    }
    
    if (SHUFFLE) {
        vector<vector<float> > vvFeature(NVECSPERFILE, vector<float> (NDIMEN));
        vector<uint64_t> vRowIdx;
        for (uint64_t row = 0; row < NVECSPERFILE; row++) { 
            vRowIdx.push_back(row);
            for (uint64_t d = 0; d < NDIMEN; d++) {
                float tmp = 0.0f;
                fscanf(fp, "%f", &tmp);
                vvFeature[row][d] = tmp;
            }
        }
        random_shuffle(vRowIdx.begin(), vRowIdx.end());
        //for (uint64_t i = 0; i < gf->nvecs; i++)
            //cout << vRowIdx[i] << " ";
        //cout << endl;
        
        ///
        /// Set data matrix with shuffled rows
        ///
        for (uint64_t row = 0; row < NVECSPERFILE; row++)
            for (uint64_t col = 0; col < NDIMEN; col++)
                data.rows[row][col] = vvFeature[vRowIdx[row]][col];
                
        vvFeature.clear();
        vRowIdx.clear();
    }
    else {
        for (uint64_t row = 0; row < NVECSPERFILE; row++) { 
            for (uint64_t d = 0; d < NDIMEN; d++) {
                float tmp = 0.0f;
                fscanf(fp, "%f", &tmp);
                data.rows[row][d] = tmp;
            }
        }           
    }
    //printMatrix(data);
    //cout << "data.rows[7][0] = " << data.rows[7][1] << endl;
    
    fclose(fp); 
    
    for (int n = 0; n < NVECSPERFILE; n++) {
        const float *normalized = normalize(data.rows[n], NORMALOPT); 
        //cout << "normalized = ";
        //for (int i = 0; i < NDIMEN; i++)
            //cout << normalized[i] << " ";
        //cout << endl;
        
        /// GET BEST NODE USING d_k (t) = || x(t) = w_k (t) || ^2
        /// AND d_c (t) == min d_k (t)
        /// p1[0] = x, p1[1] = y
        //NODE *bmu_node = get_BMU(som, normalized);
        //const float *p1 = get_coords(bmu_node);
        const float *p1 = get_bmu_coord(codebook, normalized);
        //cout << "BMU coords = " << p1[0] << "," << p1[1] << endl;
        
        float *wVec = get_wvec((int)p1[0], (int)p1[1], codebook); 
        //cout << "Alpha = " << Alpha << endl;
        
        if (R <= 1.0f) { /// ADJUST BMU NODE ONLY
            //updatew_online(bmu_node, normalized, Alpha);            
            for (int w = 0; w < NDIMEN; w++)
                codebook->rows[(int)p1[1]][(int)p1[0]*NDIMEN + w] +=  /// CAUTION: y, x order
                    Alpha * (normalized[w] - wVec[w]);
        }
        else {   /// ADJUST WEIGHT VECTORS OF THE NEIGHBORS TO BMU NODE
            for (uint64_t som_y = 0; som_y < SOM_Y; som_y++) { 
                for (uint64_t som_x = 0; som_x < SOM_X; som_x++) { 
                    float p2[NDIMEN];
                    p2[0] = (float) som_x;
                    p2[1] = (float) som_y;
                    //const float *p2 = get_coords(som->nodes[k]);
                    float dist = 0.0f;
                    
                    /// dist = sqrt((x1-y1)^2 + (x2-y2)^2 + ...)  DISTANCE TO NODE
                    for (int d = 0; d < NDIMEN; d++)
                        dist += (p1[d] - p2[d]) * (p1[d] - p2[d]);
                    dist = sqrt(dist);
                    
                    if (TRAINOPT == FAST && dist > R)
                        continue;
                        
                    /// GAUSSIAN NEIGHBORHOOD FUNCTION
                    float neighbor_fuct = exp(-(1.0f * dist * dist) / (R * R));
                    //cout << "neighbor_fuct = " << neighbor_fuct << endl;
                    //updatew_online(som->nodes[k], normalized, Alpha * neighbor_fuct);
                    
                    for (int d = 0; d < NDIMEN; d++)
                        codebook->rows[som_y][som_x*NDIMEN + d] += 
                            Alpha * neighbor_fuct * (normalized[d] - wVec[d]);
                }
            }
        }
        delete wVec;
        delete p1;
        delete normalized;
    }
}
  
/** MR-MPI Map function - batch training
 * @param itask
 * @param file
 * @param kv
 * @param ptr
 */
/* ------------------------------------------------------------------------ */
void train_batch(int itask, char *file, KeyValue *kv, void *ptr)
/* ------------------------------------------------------------------------ */
{
    GIFTBOX *gf = (GIFTBOX *) ptr;
    
    ///
    /// Read feature chunk file distributed
    ///
    string workItem(file);
    //cout << "workItem = " << workItem << endl;
    FILE *fp;   
    fp = fopen(file, "r");
    DMatrix data;
    data = initMatrix();
    data = createMatrix(NVECSPERFILE, NDIMEN);
    if (!validMatrix(data)) {
        printf("FATAL: not valid data matrix.\n");
        exit(0);
    }
    
    if (SHUFFLE) {
        vector<vector<float> > vvFeature(NVECSPERFILE, vector<float> (NDIMEN));
        vector<uint64_t> vRowIdx;
        for (uint64_t row = 0; row < NVECSPERFILE; row++) { 
            vRowIdx.push_back(row);
            for (uint64_t col = 0; col < NDIMEN; col++) {
                float tmp = 0.0f;
                fscanf(fp, "%f", &tmp);
                vvFeature[row][col] = tmp;
            }
        }
        random_shuffle(vRowIdx.begin(), vRowIdx.end());
        //for (uint64_t i = 0; i < gf->nvecs; i++)
            //cout << vRowIdx[i] << " ";
        //cout << endl;
        
        ///
        /// Set data matrix with shuffled rows
        ///
        for (uint64_t row = 0; row < NVECSPERFILE; row++)
            for (uint64_t col = 0; col < NDIMEN; col++)
                data.rows[row][col] = vvFeature[vRowIdx[row]][col];
        vvFeature.clear();
        vRowIdx.clear();
    }
    else {
        for (uint64_t row = 0; row < NVECSPERFILE; row++) { 
            for (uint64_t col = 0; col < NDIMEN; col++) {
                float tmp = 0.0f;
                fscanf(fp, "%f", &tmp);
                data.rows[row][col] = tmp;
            }
        }           
    }
    //printMatrix(data);
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
    
    for (uint64_t n = 0; n < NVECSPERFILE; n++) {

        /// Normalize
        const float *normalized = normalize(data.rows[n], NORMALOPT); 
        
        /// GET THE BEST MATCHING UNIT
        /// p1[0] = x, p1[1] = y
        const float *p1 = get_bmu_coord(gf->codebook, normalized);
        
        
        /// Accumulate denoms and numers
        for (uint64_t som_y = 0; som_y < SOM_Y; som_y++) { 
            for (uint64_t som_x = 0; som_x < SOM_X; som_x++) {
                p2[0] = (float) som_x;
                p2[1] = (float) som_y;
                float dist = 0.0f;
                for (int p = 0; p < NDIMEN; p++)
                    dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
                dist = sqrt(dist);
                //cout << "Dist b/w nodes = " << dist << endl;
                
                float neighbor_fuct = 0.0f;
                neighbor_fuct = exp(-(1.0f * dist * dist) / (gf->r * gf->r));
                
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
            float *tempVec = get_wvec(som_x, som_y, codebook);  
            
            //cout << "wvec = ";
            //for (int i = 0; i < NDIMEN; i++)
                //cout << tempVec[i] << " ";
            //cout << endl;
            
            dist = get_distance(tempVec, fvec, DISTOPT);
            if (dist < mindist) { 
                mindist = dist;
                coords[0] = (float) som_x;
                coords[1] = (float) som_y;
                //cout << "Coords = " << coords[0] << "," <<  coords[1] << endl;
            }
            //cout << "wVec = ";
            //for (int i = 0; i < NDIMEN; i++)
                //cout << tempVec[i] << " ";
            //cout << endl;
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
/* ------------------------------------------------------------------------ */
float *get_wvec(unsigned int som_x, unsigned int som_y, const DMatrix *codebook)
/* ------------------------------------------------------------------------ */
{
    float *wvec = (float *)malloc(SZFLOAT * NDIMEN);
    for (unsigned int i = 0; i < NDIMEN; i++)
        wvec[i] = codebook->rows[som_y][som_x*NDIMEN + i]; /// CAUTION: (y,x) order
    
    return wvec;
}


/* ------------------------------------------------------------------------ */
int save_2D_distance_map(DMatrix *codebook, char *fname)
/* ------------------------------------------------------------------------ */
{
    int D = 2;
    float min_dist = 1.5f;
    FILE *fp = fopen(fname, "wt");
    if (fp != 0) {
        int n = 0;
        for (int som_y1 = 0; som_y1 < SOM_Y; som_y1++) {
            for (int som_x1 = 0; som_x1 < SOM_X; som_x1++) {
            
                float dist = 0.0f;
                int nodes_number = 0;
                //NODE *pnode = get_node(som, n++);
                int coords1[2];
                coords1[0] = som_x1;
                coords1[1] = som_y1;               
                
                //for (int m = 0; m < NNODES; m++) {
                for (int som_y2 = 0; som_y2 < SOM_Y; som_y2++) {   
                    for (int som_x2 = 0; som_x2 < SOM_X; som_x2++) {
                    
                        //NODE *node = get_node(som, m);
                        int coords2[2];
                        coords2[0] = som_x2;
                        coords2[1] = som_y2;    
                        //if (node == pnode)
                            //continue;
                        if (som_x1 == som_x2 && som_y1 == som_y2)
                            continue;
                            
                        float tmp = 0.0;
                        for (int x = 0; x < D; x++) {
                            //tmp += pow(*(get_coords(pnode) + x) - *(get_coords(node) + x), 2.0f);
                            tmp += pow(coords1[x] - coords2[x], 2.0f);                            
                        }
                        tmp = sqrt(tmp);
                        if (tmp <= min_dist) {
                            nodes_number++;
                            //dist += get_distance2(node->weights, 0, pnode->weights);
                            float *vec1 = get_wvec(som_x1, som_y1, codebook);
                            float *vec2 = get_wvec(som_x2, som_y2, codebook);
                            dist += get_distance(vec1, vec2, DISTOPT);
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


/* ------------------------------------------------------------------------ */
void update_weight(uint64_t itask, char *key, int keybytes, char *value,
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
         //<< col*NDIMEN + d << ","
         //<< numer << ","
         //<< denom << ","
         //<< newWeight << endl;         
    
    if (newWeight > 0.0) 
        gf->codebook->rows[row][col*NDIMEN + d] = newWeight;
    //char *whitespace = " \t\n\f\r\0";
    ////char *whitespace = " ";
    //char *key_tokens = strtok(key, whitespace);
    //int K = atoi(key_tokens);
    //key_tokens = strtok(NULL, whitespace);
    //int W = atoi(key_tokens);
    //float new_weight = *(float *)value;
    //updatew_batch_index(gb->som->nodes[K], new_weight, W); /// UPDATE WEIGHT OF NODE K
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
void sum(char *key, int keybytes, char *multivalue, int nvalues, int *valuebytes, 
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
 
/** MR-MPI map related function - Normalize vector
 * @param float *f
 * @param const float *vec2
 * @param int distance_metric: 
 */
/* ------------------------------------------------------------------------ */
float *normalize(float *f, int normalopt)
/* ------------------------------------------------------------------------ */
{
    int d = NDIMEN;
    float *aVector = (float *)malloc(SZFLOAT * d);
    switch (normalopt) {
    default:
    case 0: /// NONE
        for (unsigned int x = 0; x < d; x++) {
            aVector[x] = f[x];
        }
        break;
    //case 1: //MNMX
        ////for (int x = 0; x < NDIMEN; x++)
        ////m_data[x] = (0.9f - 0.1f) * (vec[x] + m_add[x]) * m_mul[x] + 0.1f;
        ////break;
    //case 2: //ZSCR
        ////for (int x = 0; x < NDIMEN; x++)
        ////m_data[x] = (vec[x] + m_add[x]) * m_mul[x];
        ////break;
    //case 3: //SIGM
        ////for (int x = 0; x < NDIMEN; x++)
        ////m_data[x] = 1.0f / (1.0f + exp(-((vec[x] + m_add[x]) * m_mul[x])));
        ////break;
    case 4: /// ENRG
        float energy = 0.0f;
        for (unsigned int x = 0; x < d; x++)
            energy += f[x] * f[x];
        energy = sqrt(energy);
        for (unsigned int x = 0; x < d; x++)
            aVector[x] = f[x] / energy;
        break;
    }
    return aVector;
}

/** MR-MPI Map function - Distance b/w vec1 and vec2, default: Euclidean
 * @param float *vec1
 * @param const float *vec2
 * @param int distance_metric: 
 */
/* ------------------------------------------------------------------------ */
float get_distance(float *vec1, const float *vec2, int distance_metric)
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


/** Utility - Create matrix
 * @param unsigned int rows
 * @param unsigned int cols
 */
/* ------------------------------------------------------------------------ */
DMatrix createMatrix(const unsigned int rows, const unsigned int cols)
/* ------------------------------------------------------------------------ */
{
    DMatrix matrix;
    uint64_t m, n;
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

/** Utility - Free matrix
 * @param DMatrix *matrix
 */
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

/** Utility - Free matrix
 * @param DMatrix* matrix
 */
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



/// EOF