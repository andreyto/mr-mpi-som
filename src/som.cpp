////////////////////////////////////////////////////////////////////////////////
//
//  Serial SOM (Self-Organizing Map)
//
//  Author: Seung-Jin Sul
//          (ssul@jcvi.org)
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


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>

#include <vector>
#include <iostream>
#include <numeric>

using namespace std;
 
#define MAX_STR         255
#define SZFLOAT         sizeof(float)

//#define NDEBUG
//#define _DEBUG
//#ifdef _DEBUG
//#endif

//SOM NODE
typedef struct node {
    vector<float> weights;
    vector<float> coords;  //2D: (x,y)
} NODE;
typedef vector<NODE *>                  V_NODEP_T;
typedef vector<float>                   V_FLOAT_T;
typedef vector<vector<vector<float> > > VVV_FLOAT_T;

//SOM
typedef struct SOM {
    V_NODEP_T nodes;
} SOM;

typedef struct {
    unsigned int    m, n;       //ROWS, COLS
    float          *data;       //DATA, ORDERED BY ROW, THEN BY COL
    float         **rows;       //POINTERS TO ROWS IN DATA
} DMatrix;


/* ------------------------------------------------------------------------ */
void          train_online(SOM *som, DMatrix &F, float R, float Alpha);
//void        train_batch(SOM *som, DMatrix &F, float R);
void          train_batch2(SOM *som, DMatrix &F, float R);// VVV_FLOAT_T &numer, VVV_FLOAT_T &denom);
float        *normalize(float *vec);
float        *normalize2(DMatrix &F, int n);
float        *get_coords(NODE *);
float        *get_wvec(SOM *, int);
float         get_distance(float *, int, vector<float> &);
float         get_distance2(vector<float> &vec, int distance_metric, vector<float> &wvec);
void          updatew_online(NODE *node, float *vec, float Alpha_x_Hck);
void          updatew_batch(NODE *, float *);
//void        updatew_batch_index(NODE *node, float new_weight, int k);
void          get_file_name(char *path, char *name);
int           save_2D_distance_map(SOM *som, char *fname);
NODE         *get_node(SOM *, int);
NODE         *get_BMU(SOM *, float *);

//CLASSIFYING
NODE         *classify(SOM *som, float *vec);

//MATRIX
DMatrix       createMatrix(const unsigned int rows, const unsigned int cols);
DMatrix       initMatrix(void);
void          freeMatrix(DMatrix *matrix);
void          printMatrix(DMatrix A);
int           validMatrix(DMatrix matrix);

//BATCH3: IMPROVED BATCH////////////////////////////////////////////////
class BMUnode {
public:
    int i;
    float dist;
    BMUnode() : i(0), dist(0) {}
    BMUnode( int index, float dist_value ) : i( index ), dist( dist_value) {}
};
typedef vector<BMUnode>             V_BMUNODE_T;
typedef vector<vector<BMUnode> >    VV_BMUNODE_T;
VV_BMUNODE_T *create_BMUTable(SOM *som, DMatrix &F);
void          train_batch3(SOM* som, DMatrix &F, DMatrix &W, 
                           V_FLOAT_T &vn, DMatrix &S, DMatrix &D, float R);
float         nodes_distance(NODE *n1, NODE *n2);

/* ------------------------------------------------------------------------ */

//GLOBALS
int NDIMEN;             //NUM OF DIMENSIONALITY
int NVECS;              //NUM OF FEATURE VECTORS
int SOM_X=50;
int SOM_Y=50;
int SOM_D=2;            //2=2D  
int NNODES=SOM_X*SOM_Y; //TOTAL NUM OF SOM NODES
int NEPOCHS;            //ITERATIONS
int DOPT=0;             //0=EUCL, 1=SOSD, 2=TXCB, 3=ANGL, 4=MHLN
int TMODE=0;            //0=BATCH, 1=ONLINE
int TOPT=0;             //0=SLOW, 1=FAST
int NORMAL=0;           //0=NONE, 1=MNMX, 2=ZSCR, 3=SIGM, 4=ENR

/* ------------------------------------------------------------------------ */
int main(int argc, char *argv[])
/* ------------------------------------------------------------------------ */
{
    SOM *som;
    //vector<NODE *> NodeGarbageCan;
    //cout << argc << endl;
    if (argc == 5) { //RANDOM 
        //syntax: mrsom NEPOCHS TMODE NVECS NDIMEN
        NEPOCHS = atoi(argv[1]);    //# OF ITERATIONS
        TMODE = atoi(argv[2]);      //BATCH OR ONLINE
        NVECS = atoi(argv[3]);      //NUM FEATURE VECTORS
        NDIMEN = atoi(argv[4]);     //NUM DIMENSIONALITY OF DATA 
    }
    else if (argc == 7) { //RANDOM 
        //syntax: mrsom NEPOCHS TMODE NVECS NDIMEN X Y
        NEPOCHS = atoi(argv[1]);     
        TMODE = atoi(argv[2]);       
        NVECS = atoi(argv[3]);
        NDIMEN = atoi(argv[4]); 
        SOM_X = atoi(argv[5]);      //WIDTH OF SOM MAP
        SOM_Y = atoi(argv[6]);      //HEIGHT OF SOM MAP
        NNODES=SOM_X*SOM_Y;         //NUM NODES IN SOM
    }
    else if (argc == 6) {  ///READ FEATURE DATA FROM FILE
        //syntax: mrsom FILE NEPOCHS TMODE NVECS NDIMEN
        NEPOCHS = atoi(argv[2]);     
        TMODE = atoi(argv[3]);       
        NVECS = atoi(argv[4]);
        NDIMEN = atoi(argv[5]);        
    }
    else if (argc == 8) {  ///READ FEATURE DATA FROM FILE
        //syntax: mrsom FILE NEPOCHS TMODE NVECS NDIMEN X Y
        NEPOCHS = atoi(argv[2]);     
        TMODE = atoi(argv[3]);       
        NVECS = atoi(argv[4]);
        NDIMEN = atoi(argv[5]);        
        SOM_X = atoi(argv[6]);  
        SOM_Y = atoi(argv[7]);  
        NNODES=SOM_X*SOM_Y;
    }
    else {
        printf("         mrsom [FILE] NEPOCHS TMODE NVECS NDIMEN [X Y]\n\n");   
        printf("         [FILE]  = optional, feature vector file.\n");     
        printf("                   If not provided, (NVECS * NDIMEN) random\n");     
        printf("                   vectors are generated.\n");     
        printf("         NEPOCHS = number of iterations.\n");     
        printf("         TMODE   = 0-batch, 1-online.\n");
        printf("         NVECS   = number of feature vectors.\n");     
        printf("         NDIMEN  = number of dimensionality of feature vector.\n");     
        printf("         [X Y]   = optional, SOM map size. Default = [50 50]\n");     
        exit(0);
    }    
    
    //MAKE SOM//////////////////////////////////////////////////////////
    som = (SOM *)malloc(sizeof(SOM));
    som->nodes = V_NODEP_T(NNODES);
    for (int x = 0; x < NNODES; x++) {
        som->nodes[x] = (NODE *)malloc(sizeof(NODE));
    }
    
    //CREATE WEIGHT MATRIX////////////////////////////////////////////////
    DMatrix W; //WEIGHT VECTORS
    W = initMatrix();    
    W = createMatrix(NNODES, NDIMEN);
    if (!validMatrix(W)) {
        printf("FATAL: not valid W matrix.\n");
        exit(0);
    }
    
    //FILL WEIGHTS//////////////////////////////////////////////////////
    srand((unsigned int)time(0));
    for (int x = 0; x < NNODES; x++) {
        NODE *node = (NODE *)malloc(sizeof(NODE));
        //NodeGarbageCan.push_back(node);
        node->weights.resize(NDIMEN);
        node->coords.resize(SOM_D, 0.0);
        for (int i = 0; i < NDIMEN; i++) {
            int w1 = 0xFFF & rand();
            w1 -= 0x800;
            node->weights[i] = (float)w1 / 4096.0f;
        }
        som->nodes[x] = node;
    }

    //FILE COORDS (2D RECT)/////////////////////////////////////////////
    for (int x = 0; x < SOM_X; x++) {
        for (int y = 0; y < SOM_Y; y++) {
            som->nodes[(x*SOM_Y)+y]->coords[0] = y * 1.0f;
            som->nodes[(x*SOM_Y)+y]->coords[1] = x * 1.0f;
        }
    }
    
    //CREATE DATA MATRIX////////////////////////////////////////////////
    DMatrix F; //FEATURE VECTOR
    F = initMatrix();    
    F = createMatrix(NVECS, NDIMEN);
    if (!validMatrix(F)) {
        printf("FATAL: not valid F matrix.\n");
        exit(0);
    }

    //READ FEATURE DATA FROM FILE///////////////////////////////////////
    if (argc == 6 || argc == 8) {  ///READ FEATURE DATA FROM FILE
        printf("Reading (%d x %d) feature vectors from %s...\n", NVECS, NDIMEN, argv[1]);
        FILE *fp;
        fp = fopen(argv[1],"r");
        for(int i = 0; i < NVECS; i++) {
            for(int j = 0; j < NDIMEN; j++) {
                float tmp = 0.0f;
                fscanf(fp, "%f", &tmp);
                F.rows[i][j] = tmp;
            }
        }
        fclose(fp);
        //printMatrix(F);
    } 
    else if (argc == 5 || argc == 7) {
        printf("Generating (%d x %d) random feature vectors...\n", NVECS, NDIMEN);
        //RANDOM FEATURE VECTORS
        for(int i = 0; i < NVECS; i++) {
            for(int j = 0; j < NDIMEN; j++) {
                F.rows[i][j] = rand() % 100 / 100.0f;
            }
        }
    }
        
    float N = (float)NEPOCHS;       //ITERATIONS
    float nrule, nrule0 = 0.9f;     //LEARNING RATE FACTOR
    float R, R0;
    R0 = SOM_X / 2.0f;              //INIT RADIUS FOR UPDATING NEIGHBORS
    int x = 0;                      //0...N-1
    
    ////double time0, time1;
    struct timeval t1_start;
    struct timeval t1_end;
    double t1_time;
    gettimeofday(&t1_start, NULL);
    
    V_FLOAT_T vn = V_FLOAT_T(NNODES, 0.0);
    
    DMatrix S; 
    S = initMatrix();    
    S = createMatrix(NNODES, NDIMEN);
    if (!validMatrix(S)) {
        printf("FATAL: not valid S matrix.\n");
        exit(0);
    }
    
    //CREATE DISTANCE TABLE
    DMatrix D; 
    D = initMatrix();    
    D = createMatrix(NNODES, NNODES);
    if (!validMatrix(D)) {
        printf("FATAL: not valid D matrix.\n");
        exit(0);
    }
    for (int i = 0; i < NNODES; i++) {
        for (int j = i; j < NNODES; j++) {
            D.rows[i][j] = D.rows[j][i] = nodes_distance(som->nodes[i], som->nodes[j]);
        }
    }
    //printMatrix(D);
    
    //ITERATIONS////////////////////////////////////////////////////////
    while (NEPOCHS) {
        R = R0 * exp(-10.0f * (x * x) / (N * N));
        x++;  
        if (TMODE == 0) { //BATCH                          
            train_batch2(som, F, R); //, numer, denom);
            printf("BATCH2-  epoch: %d   R: %.2f \n", (NEPOCHS-1), R);
        } else if (TMODE == 1) { //ONLINE
            nrule = nrule0 * exp(-10.0f * (x * x) / (N * N));  //LEARNING RULE SHRINKS OVER TIME, FOR ONLINE SOM
            train_online(som, F, R, nrule);
            printf("ONLINE-  epoch: %d   R: %.2f \n", (NEPOCHS-1), R);
        }
        else if (TMODE == 2) { //NEW BATCH
            train_batch3(som, F, W, vn, S, D, R);  
            printf("BATCH3-  epoch: %d   R: %.2f \n", (NEPOCHS-1), R);
        }
        NEPOCHS--;
    }
    gettimeofday(&t1_end, NULL);
    t1_time = t1_end.tv_sec - t1_start.tv_sec + (t1_end.tv_usec - t1_start.tv_usec) / 1.e6;
    fprintf(stderr,"Processing time: %g seconds\n",t1_time);

    freeMatrix(&F);
    freeMatrix(&W); //FOR BATCH3
    freeMatrix(&D); //FOR BATCH3
    freeMatrix(&S); //FOR BATCH3
    vn.clear();
    //for (int i = 0; i < NodeGarbageCan.size(); i++) {
        //if (NodeGarbageCan[i] != NULL) {
            //NODE *temp = NodeGarbageCan[i];
            //delete temp;
        //}
    //}
 
    //SAVE SOM//////////////////////////////////////////////////////////
    printf("Saving SOM...\n");
    char som_map[MAX_STR] = "";
    //get_file_name(argv[1], som_map);
    strcat(som_map, "result.map");
    save_2D_distance_map(som, som_map);
    printf("Converting SOM map to distance map...\n");
    system("python ./show.py");
    printf("Done!\n");
 
    return 0;
}

/* ------------------------------------------------------------------------ */
float nodes_distance(NODE *n1, NODE *n2)
/* ------------------------------------------------------------------------ */
{
    float dist = 0.0f;
    const float *p1 = get_coords(n1); 
    const float *p2 = get_coords(n2); 
    for (int k = 0; k < NDIMEN; k++)
        dist += (p1[k] - p2[k]) * (p1[k] - p2[k]);
    //printf("dist = %f\n", sqrt(dist));
    return sqrt(dist);
}

/* ------------------------------------------------------------------------ */
VV_BMUNODE_T *create_BMUTable(SOM *som, DMatrix &F, DMatrix &W)
/* ------------------------------------------------------------------------ */
{
    int i, j, k, j_min, j_min2;  
    int d = NDIMEN;
    int n = NVECS;
    int m = NNODES;
    float dist, diff, dist_min, dist_min2;
    VV_BMUNODE_T *bmu_table = new VV_BMUNODE_T;
    bmu_table->clear();
    V_BMUNODE_T vec;
    
    for (i = 0; i < n; i++) {
        j_min = 0;  j_min2 = 0;
        dist_min = 9999999; dist_min2 = 9999999;        
        for (j = 0; j < m; j++) {
            dist = 0;    
            for (k = 0; k < d; k++) {
                diff = F.rows[i][k] - W.rows[j][k]; 
                dist += diff*diff;
                //printf("%f %f dist = %f\n", F.rows[i][k], W.rows[j][k], dist);
            }
            if (dist < dist_min) {
                j_min2 = j_min;
                dist_min2 = dist_min;
                j_min = j;
                dist_min = dist;            
            } else if ((j_min2 == 0)&&( dist < dist_min2)) {
                j_min2 = j;
                dist_min2 = dist;
            }
            //vec.push_back( BMUnode( j, sqrt(dist) )  );       
        }     
        vec.push_back(BMUnode(j_min, sqrt(dist_min)));
        vec.push_back(BMUnode(j_min2, sqrt(dist_min2)));
        //sort( vec.begin(), vec.end() );
        bmu_table->push_back(vec);      
        vec.clear();          
    }      
    return bmu_table;
}

/* ------------------------------------------------------------------------ */
void train_batch3(SOM* som, DMatrix &F, DMatrix &W, V_FLOAT_T &vn, 
                  DMatrix &S, DMatrix &D, float R) 
/* ------------------------------------------------------------------------ */
{
    int i, k, j, i1, i2, bmu;
    float h, r, htot;  
    int d = NDIMEN;
    int n = NVECS;
    int m = NNODES;  
    
    //READ WEIGHTS INTO W
    for (int i = 0; i < m; i++) 
        for (int k = 0; k < NDIMEN; k++) 
            W.rows[i][k] = som->nodes[i]->weights[k]; 
    //printf("W1 %0.2f %0.2f %0.2f\n", W.rows[0][0], W.rows[0][1], W.rows[0][2]);
    
    VV_BMUNODE_T *tmp_BMUTable = new VV_BMUNODE_T;
    tmp_BMUTable = create_BMUTable(som, F, W);
    assert(tmp_BMUTable->size() == NVECS);
    //printf("tmp_BMUTable->size() = %d\n", tmp_BMUTable->size());
    //for (i = 0; i < tmp_BMUTable->size(); i++)
        //for (j = 0; j < tmp_BMUTable[i].size(); j++)
            //for (k = 0; k < n; k++)
                //printf("%d ", tmp_BMUTable[0][k][0].i);
            //printf("\n");   
 
    //INITIATE THE VORONOI VECTOR AND THE SUMDATA VARIABLES
    for (i = 0; i < m; i++) { 
        vn[i] = 0; 
        for (k = 0; k < d; k++) 
            S.rows[i][k] = 0.0;
    }
    for (j = 0; j < n; j++ ) {   
        bmu = tmp_BMUTable[0][j][0].i;      
        //printf("bmu = %d\n", bmu);
        vn[bmu]++;  
        for (k = 0; k < d; k++) 
            S.rows[bmu][k] += F.rows[j][k]; 
    }

    //FILL WEIGHTS WITH ZEROS
    for (i = 0; i < m; i++) 
        for (k = 0; k < d; k++) 
            W.rows[i][k] = 0.0;
            
    for (i1 = 0; i1 < m; i1++) {
        htot = 0;   
        for (i2 = 0; i2 < m; i2++ ) {                               
            //h = getTopology()->H( delta[i1][i2], r, neighboor );
            h = exp(-(1.0f * D.rows[i1][i2] * D.rows[i1][i2]) / (R * R));
            for (k = 0; k < d; k++) 
                W.rows[i1][k] += h*S.rows[i2][k]; 
            htot += h*vn[i2];   
            //printf("h %0.2f %0.2f %0.2f %0.2f %0.2f\n", h, D.rows[i1][i2], R, vn[i2], htot);
        }   
        for (k = 0; k < d; k++) 
            if (htot != 0) 
                W.rows[i1][k] /= htot; 
    }
    //printf("W2 %0.2f %0.2f %0.2f\n", W.rows[0][0], W.rows[0][1], W.rows[0][2]);

    //UPDATE WEIGHTS
    for (i = 0; i < m; i++) {
        for (k = 0; k < d; k++) {
            if (W.rows[i][k] > 0)
                som->nodes[i]->weights[k] = W.rows[i][k];                  
        }
    }

    tmp_BMUTable->clear();
    delete tmp_BMUTable;
}

/* ------------------------------------------------------------------------ */
void train_online(SOM *som, DMatrix &F, float R, float Alpha)
/* ------------------------------------------------------------------------ */
{
    for (int n = 0; n < NVECS; n++) {
        float *normalized = normalize2(F, n);
        //GET BEST NODE USING d_k(t) = || x(t) = w_k(t) || ^2
        //AND d_c(t) == min d_k(t)
        NODE *bmu_node = get_BMU(som, normalized);
        const float *p1 = get_coords(bmu_node);
        if (R <= 1.0f) { //ADJUST BMU NODE ONLY
            updatew_online(bmu_node, normalized, Alpha);
        } else { //ADJUST WEIGHT VECTORS OF THE NEIGHBORS TO BMU NODE
            for (int k = 0; k < NNODES; k++) { //ADJUST WEIGHTS OF ALL K NODES IF DOPT <= R
                const float *p2 = get_coords(som->nodes[k]);
                float dist = 0.0f;
                //dist = sqrt((x1-y1)^2 + (x2-y2)^2 + ...)  DISTANCE TO NODE
                for (int p = 0; p < NDIMEN; p++)
                    dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
                dist = sqrt(dist);
                //printf("dist = %f\n", dist);
                if (TOPT == 1 && dist > R)
                    continue;
                //GAUSSIAN NEIGHBORHOOD FUNCTION
                float neighbor_fuct = exp(-(1.0f * dist * dist) / (R * R));
                updatew_online(som->nodes[k], normalized, Alpha * neighbor_fuct);
            }
        }
    }
}


/* ------------------------------------------------------------------------ */
void train_batch2(SOM* som, DMatrix &F, float R) 
                  //VVV_FLOAT_T &numer, VVV_FLOAT_T &denom)
/* ------------------------------------------------------------------------ */
{
    VVV_FLOAT_T numer = VVV_FLOAT_T (NVECS, vector<vector<float> > (NNODES,
                                   vector<float>(NDIMEN, 0.0)));
    VVV_FLOAT_T denom = VVV_FLOAT_T (NVECS, vector<vector<float> > (NNODES,
                                     vector<float>(NDIMEN, 0.0)));
    
    for (int n = 0; n < NVECS; n++) {
        //printf("orig       %f %f %f \n", FEATURE[n][0],FEATURE[n][1],FEATURE[n][2]);
        float *normalized = normalize2(F, n);
        //printf("normalized %f %f %f \n", normalized[0],normalized[1],normalized[2]);
        NODE *bmu_node = get_BMU(som, normalized);
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
    //UPDATE W-DIMENSINAL WEIGHTS FOR EACH NODE
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
        //printf("BEFORE %f %f %f\n", som->nodes[k]->weights[0], som->nodes[k]->weights[1], som->nodes[k]->weights[2]);
        //printf("NEW    %f %f %f\n", new_weights[0], new_weights[1], new_weights[2]);
        //if (R > 1.0f) 
        updatew_batch(som->nodes[k], new_weights);
        //printf("AFTER  %f %f %f\n", som->nodes[k]->weights[0], som->nodes[k]->weights[1], som->nodes[k]->weights[2]);
    }
    free(new_weights);
    numer.clear();
    denom.clear();
}
 
 
 /* ------------------------------------------------------------------------ */
NODE *classify(SOM *som, float *vec)
/* ------------------------------------------------------------------------ */
{       
    NODE *pbmu_node = som->nodes[0];
    float *normalized = normalize(vec);
    float mindist = get_distance(normalized, 0, pbmu_node->weights);
    float dist;
    for (int x = 0; x < NNODES; x++) {
        if ((dist = get_distance(normalized, 0, som->nodes[x]->weights)) < mindist) {
            mindist = dist;
            pbmu_node = som->nodes[x];
        }
    }
    //CAN ADD A FEATURE FOR VOTING AMONG BMUS.
    return pbmu_node;
}

/* ------------------------------------------------------------------------ */
float *normalize(float *vec)
/* ------------------------------------------------------------------------ */
{
    float *m_data = (float *)malloc(SZFLOAT*NDIMEN);
    switch (NORMAL) {
    default:
    case 0: //NONE
        for (int x = 0; x < NDIMEN; x++) {
            m_data[x] = vec[x];
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
            energy += vec[x] * vec[x];
        energy = sqrt(energy);
        for (int x = 0; x < NDIMEN; x++)
            m_data[x] = vec[x] / energy;                
        break;
    }
    return m_data;
}

/* ------------------------------------------------------------------------ */
void updatew_online(NODE *node, float *vec, float Alpha_x_Hck)
/* ------------------------------------------------------------------------ */
{
    //printf("BEFORE %f %f %f\n", node->weights[0], node->weights[1], node->weights[2]);
    //printf("NEW    %f %f %f\n", Alpha_x_Hck * (vec[0] - node->weights[0]), Alpha_x_Hck * (vec[1] - node->weights[1]), Alpha_x_Hck * (vec[2] - node->weights[2]));
    for (int w = 0; w < NDIMEN; w++)
        node->weights[w] += Alpha_x_Hck * (vec[w] - node->weights[w]);
    //printf("AFTER  %f %f %f\n", node->weights[0], node->weights[1], node->weights[2]);
}

/* ------------------------------------------------------------------------ */
void updatew_batch(NODE *node, float *new_w)
/* ------------------------------------------------------------------------ */
{
    for (int w = 0; w < NDIMEN; w++) {
        if (new_w[w] > 0) //????   
            node->weights[w] = new_w[w];
    }
}
 
/* ------------------------------------------------------------------------ */
float *normalize2(DMatrix &F, int n)
/* ------------------------------------------------------------------------ */
{
    float *m_data = (float *)malloc(SZFLOAT*NDIMEN);
    switch (NORMAL) {
    default:
    case 0: //NONE
        for (int x = 0; x < NDIMEN; x++) {
            m_data[x] = F.rows[n][x];
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
            energy += F.rows[n][x] * F.rows[n][x];
        energy = sqrt(energy);
        for (int x = 0; x < NDIMEN; x++)
            m_data[x] = F.rows[n][x] / energy;                
        break;
    }
    return m_data;
}

/* ------------------------------------------------------------------------ */
NODE *get_BMU(SOM *som, float *fvec)
/* ------------------------------------------------------------------------ */
{
    NODE *pbmu_node = som->nodes[0];
    float mindist = get_distance(fvec, 0, pbmu_node->weights);
    float dist;
    for (int x = 0; x < NNODES; x++) {
        if ((dist = get_distance(fvec, 0, som->nodes[x]->weights)) < mindist) {
            mindist = dist;
            pbmu_node = som->nodes[x];
        }
    }
    //CAN ADD A FEATURE FOR VOTING AMONG BMUS.
    return pbmu_node;
}

/* ------------------------------------------------------------------------ */
float get_distance(float *vec, int distance_metric, vector<float> &wvec)
/* ------------------------------------------------------------------------ */
{
    float distance = 0.0f;
    float n1 = 0.0f, n2 = 0.0f;
    switch (distance_metric) {
    default:
    case 0: //EUCLIDIAN
        //if (NDIMEN >= 4) {
            //distance = mse(vec, distance_metric, wvec);
        //} else {
            for (int w = 0; w < NDIMEN; w++)
                distance += (vec[w] - wvec[w]) * (vec[w] - wvec[w]);
        //}
        return sqrt(distance);
    case 1: //SOSD: //SUM OF SQUARED DISTANCES
        //if (m_weights_number >= 4) {
                //distance = mse(vec, m_weights, m_weights_number);
        //} else {
            for (int w = 0; w < NDIMEN; w++)
                distance += (vec[w] - wvec[w]) * (vec[w] - wvec[w]);
        //}
        return distance;
    case 2: //TXCB: //TAXICAB
        for (int w = 0; w < NDIMEN; w++)
            distance += fabs(vec[w] - wvec[w]);
        return distance;
    case 3: //ANGL: //ANGLE BETWEEN VECTORS
        for (int w = 0; w < NDIMEN; w++) {
            distance += vec[w] * wvec[w];
            n1 += vec[w] * vec[w];
            n2 += wvec[w] * wvec[w];
        }
        return acos(distance / (sqrt(n1)*sqrt(n2)));
    //case 4: //MHLN:   //mahalanobis
        //distance = sqrt(m_weights * cov * vec)
        //return distance
    }
    
}

/* ------------------------------------------------------------------------ */
float get_distance2(vector<float> &vec, int distance_metric, vector<float> &wvec)
/* ------------------------------------------------------------------------ */
{
    float distance = 0.0f;
    float n1 = 0.0f, n2 = 0.0f;
    switch (distance_metric) {
    default:
    case 0: //EUCLIDIAN
        for (int w = 0; w < NDIMEN; w++)
            distance += (vec[w] - wvec[w]) * (vec[w] - wvec[w]);
        return sqrt(distance);
    case 1: //SOSD: //SUM OF SQUARED DISTANCES
        //if (m_weights_number >= 4) {
                //distance = mse(vec, m_weights, m_weights_number);
        //} else {
            for (int w = 0; w < NDIMEN; w++)
                distance += (vec[w] - wvec[w]) * (vec[w] - wvec[w]);
        //}
        return distance;
    case 2: //TXCB: //TAXICAB
        for (int w = 0; w < NDIMEN; w++)
            distance += fabs(vec[w] - wvec[w]);
        return distance;
    case 3: //ANGL: //ANGLE BETWEEN VECTORS
        for (int w = 0; w < NDIMEN; w++) {
            distance += vec[w] * wvec[w];
            n1 += vec[w] * vec[w];
            n2 += wvec[w] * wvec[w];
        }
        return acos(distance / (sqrt(n1)*sqrt(n2)));
    //case 4: //MHLN:   //mahalanobis
        //distance = sqrt(m_weights * cov * vec)
        //return distance
    }
}

/* ------------------------------------------------------------------------ */
float *get_wvec(SOM *som, int n)
/* ------------------------------------------------------------------------ */
{
    float *weights = (float *)malloc(SZFLOAT*NDIMEN);
    for (int i = 0; i < NDIMEN; i++)
        weights[i] = som->nodes[n]->weights[i];
    return weights;
}

/* ------------------------------------------------------------------------ */
float *get_coords(NODE *node)
/* ------------------------------------------------------------------------ */
{
    float *coords = (float *)malloc(SZFLOAT);
    for (int i = 0; i < SOM_D; i++)
        coords[i] = node->coords[i];
    return coords;
}

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
    memcpy(name, &path[sl], (dot - sl)*2);
    name[dot-sl] = 0;
}

/* ------------------------------------------------------------------------ */
int save_2D_distance_map(SOM *som, char *fname)
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
                    for (int x = 0; x < D; x++)
                        tmp += pow(*(get_coords(pnode) + x) - *(get_coords(node) + x), 2.0f);
                    tmp = sqrt(tmp);
                    if (tmp <= min_dist) {
                        nodes_number++;
                        dist += get_distance2(node->weights, 0, pnode->weights);
                    }
                }
                dist /= (float)nodes_number;
                fprintf(fp, " %f", dist);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        return 0;
    } else
        return -2;
}

/* ------------------------------------------------------------------------ */
NODE *get_node(SOM *som, int n)
/* ------------------------------------------------------------------------ */
{
    return som->nodes[n];
}
    
///* ------------------------------------------------------------------------ */
//void updatew_batch_index(NODE *node, float new_weight, int k)
///* ------------------------------------------------------------------------ */
//{
    //if (new_weight > 0)
        //node->weights[k] = new_weight;
//}


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
    } else {
        freeMatrix(&matrix);
    }
    return matrix;
}


/* ------------------------------------------------------------------------ */
void freeMatrix (DMatrix *matrix)
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
int validMatrix (DMatrix matrix)
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
            for (j = 0; j < A.n; j++) printf ("%7.3f ", A.rows[i][j]);
            printf ("\n");
        }
    }
}
  
