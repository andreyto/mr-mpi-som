////////////////////////////////////////////////////////////////////////////////
//
//  Parallelizing SOM on MR-MPI
//
//  Author: Seung-Jin Sul
//          (ssul@jcvi.org)
//
//  Revisions
//      v.0.0.9
//          6.1.2010        Start implementing serial SOM.
//          6.2.2010        MPI connected. CMake done!
//          6.3.2010        MR-MPU conencted as static library.
//          6.14.2010       serial online/batch SOM done!
//          6.15.2010       Batch SOM test done! Start mapreducing.
//          6.17.2010       Found the bottleneck in batch mode == get_bmu
//                          improved train_batch is done! ==> train_batch2
//          6.23.2010       bcast and scatter are done! Now all tasks have the initial
//                          weight vectors and the feature vectors are evenly distributed.
//
//      v.1.0.0
//          6.24.2010       Initial version of MRSOM's done!! Promoted to v.1.0.0.
//
//      v.1.1.0
//          6.30.2010       Reimplement without classes.
//
//      v.2.0.0
//          07.01.2010      Incorporated TMatrix struct.
//                          Change the MR part with gather().
//
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



// MPI and MapReduce-MPI
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

using namespace std;

#define MPI_ROOT        0
#define DIMENSION       3
#define NUM_FEATURES    30
#define SOM_X           50
#define SOM_Y           50
#define NUM_SOM_NODES   SOM_X * SOM_Y
#define SOM_DIMEN       2
#define DIST_METRIC     0 //0=EUCL
//#define TRAIN_MODE      0 //0=BATCH, 1=ONLINE
#define TRAIN_OPTION    0 //0=SLOW, 1=FAST
#define NORMALIZE       0 //0=NONE
#define MAX_STRING_LEN  255
#define SZFLOAT         sizeof(float)

//SOM node
typedef struct node {
    vector<float> m_weights;
    vector<float> m_coords;  //2D: (x,y)
} NODE;
typedef vector<NODE *> VEC_NODES_T;

//SOM
typedef struct SOM {
    VEC_NODES_T m_nodes;
} SOM;


typedef struct {
    unsigned int    m, n;  // Rows, cols
    float          *data;  // Data, ordered by row, then by col
    float         **rows;  // Pointers to rows in data
} TMatrix;

typedef vector<vector<vector<float> > > NUMERATOR_PER_VECTOR_T;
typedef vector<vector<vector<float> > > DENOMINATOR_PER_VECTOR_T;

struct GIFTBOX {
    SOM     *som;
    float   R;
    int     new_num_vectors;
    //int     num_som_nodes;
    //int     num_weights_per_node;
    TMatrix *f_vectors;
    TMatrix *w_vectors;
    NUMERATOR_PER_VECTOR_T *numer_vec;
    DENOMINATOR_PER_VECTOR_T *denom_vec;

    int     idx_start;
    int     flag;

    //debug
    int     myid;
    char    *myname;
};

/* ------------------------------------------------------------------------ */
TMatrix createMatrix(const unsigned int rows, const unsigned int cols);
void    freeMatrix(TMatrix *matrix);
int     validMatrix(TMatrix matrix);
TMatrix initMatrix(void);
void    printMatrix(TMatrix A);
void    train_online(SOM *som, TMatrix &f, float R, float learning_rate);
void    train_batch(SOM *som, TMatrix &f, float R);
void    train_batch2(SOM *som, TMatrix &f, float R);
void    MR_train_batch(MAPREDUCE_NS::MapReduce *mr, SOM *som, TMatrix &f, TMatrix &w,
                       int *idx_vector_scattered,
                       float R, int argc, char* argv[], int myid,
                       char *myname, int numprocs);
//NUMERATOR_PER_VECTOR_T &numer_vec,
//DENOMINATOR_PER_VECTOR_T &denom_vec);
float   *normalize(int);
float   *normalize2(TMatrix &f, int n);
NODE    *get_bmu_node(SOM *, float *);
float   get_distance(float *, int, vector<float> &);
float   *get_coords(NODE *);
float   *get_wvec(SOM *, int);
void    updatew_online(NODE *node, float *vec, float learning_rate_x_neighborhood_func);
void    updatew_batch(NODE *, float *);
void    updatew_batch_index(NODE *node, float new_weight, int w);
void    get_file_name(char *path, char *name);
int     save_2D_distance_map(SOM *som, char *fname);
NODE    *get_node(SOM *, int);

////mapper and reducer
void    MR_compute_weight(int itask, MAPREDUCE_NS::KeyValue *kv, void *ptr);
void    MR_accumul_weight(char *key, int keybytes, char *multivalue,
                          int nvalues, int *valuebytes, MAPREDUCE_NS::KeyValue *kv,
                          void *ptr);
void    MR_update_weight(uint64_t itask, char *key, int keybytes, char *value,
                         int valuebytes, MAPREDUCE_NS::KeyValue *kv, void *ptr);

//float sqroot(float m);
/* ------------------------------------------------------------------------ */



/* ------------------------------------------------------------------------ */
int main(int argc, char *argv[])
/* ------------------------------------------------------------------------ */
{
    int length;
    SOM *som;
    double time0, time1;
    //int num_dimensions = DIMENSION;
    //int num_vectors = NUM_FEATURES;
    //int num_nodes = SOM_X*SOM_Y;
    //int som_x = SOM_X;
    //int som_y = SOM_Y;
    //int som_dimen = SOM_DIMEN;

    TMatrix f;
    f = initMatrix();

    //make som//////////////////////////////////////////////////////////
    som = (SOM *)malloc(sizeof(SOM));
    som->m_nodes = VEC_NODES_T(NUM_SOM_NODES);
    for (int x = 0; x < SOM_X * SOM_Y; x++) {
        som->m_nodes[x] = (NODE *)malloc(sizeof(NODE));
    }

    //fill weights//////////////////////////////////////////////////////
    for (int x = 0; x < NUM_SOM_NODES; x++) {
        NODE *node = (NODE *)malloc(sizeof(NODE));
        node->m_weights.resize(DIMENSION);
        node->m_coords.resize(SOM_DIMEN, 0.0);
        for (int i = 0; i < DIMENSION; i++) {
            int w = 0xFFF & rand();
            w -= 0x800;
            node->m_weights[i] = (float)w / 4096.0f;
        }
        som->m_nodes[x] = node;
    }

    TMatrix w;
    w = initMatrix();
    w = createMatrix(NUM_SOM_NODES, DIMENSION);
    if (!validMatrix(w)) {
        printf("FATAL: not valid matrix.\n");
        exit(0);
    }
    //for (int x = 0; x < num_nodes; x++)
    //for (int j = 0; j < num_dimensions; j++)
    //w.rows[x][j] = som->m_nodes[x]->m_weights[j];
    //printMatrix(w);

    //file coords (2D rect)/////////////////////////////////////////////
    for (int x = 0; x < SOM_X; x++) {
        for (int y = 0; y < SOM_Y; y++) {
            som->m_nodes[(x*SOM_Y)+y]->m_coords[0] = y * 1.0f;
            som->m_nodes[(x*SOM_Y)+y]->m_coords[1] = x * 1.0f;
        }
    }

    //create data matrix////////////////////////////////////////////////
    f = createMatrix(NUM_FEATURES, DIMENSION);
    if (!validMatrix(f)) {
        printf("FATAL: not valid matrix.\n");
        exit(0);
    }

    //random features
    //for(int i = 0; i < NUM_FEATURES; i++) {
    //for(int j = 0; j < DIMENSION; j++) {
    ////int w = 0xFFF & rand();
    ////w -= 0x800;
    //f.rows[i][j] = rand() % 100 / 100.0f;
    //}
    //}

    ////read feature data///////////////////////////////////////////////
    FILE *fp = fopen(argv[1],"r");
    for(int i = 0; i < NUM_FEATURES; i++) {
        for(int j = 0; j < DIMENSION; j++) {
            float tmp = 0.0f;
            fscanf(fp, "%f", &tmp);
            f.rows[i][j] = tmp;
        }
    }
    fclose(fp);
    //printMatrix(f);

    //train/////////////////////////////////////////////////////////////
    float R, R0;
    R0 = SOM_X / 2.0f;
    int epochs = atoi(argv[2]);     //# of iterations
    int TRAIN_MODE = atoi(argv[3]); //batch or online
    float N = (float)epochs;
    int x = 0;  //0...N-1
    float nrule, nrule0 = 0.9f;     //learning rate factor

    //MPI///////////////////////////////////////////////////////////////
    int myid, numprocs;
    char myname[MAX_STRING_LEN];
    MPI_Status status;

    int ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "MPI initsialization failed !\n");
        exit(0);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Get_processor_name(myname, &length);

    //struct timeval t1_start;
    //struct timeval t1_end;
    //double t1_time;
    //gettimeofday(&t1_start, NULL);

    //MR-MPI////////////////////////////////////////////////////////////
    MAPREDUCE_NS::MapReduce *mr = new MAPREDUCE_NS::MapReduce(MPI_COMM_WORLD);
    //mr->verbosity = 2;
    //mr->timer = 1;

    //iterations////////////////////////////////////////////////////////
    while (epochs) {
        if (TRAIN_MODE == 0) {
            int chunksize = NUM_FEATURES / numprocs;
            int idx_vector[NUM_FEATURES];
            int idx_vector_scattered[chunksize];

            if (myid == 0) {
                // R to broadcast
                R = R0 * exp(-10.0f * (x * x) / (N * N));
                x++;

                //to scatter feature vectors.
                for (int i = 0; i < NUM_FEATURES; i++)
                    idx_vector[i] = i;

                for (int x = 0; x < NUM_SOM_NODES; x++)
                    for (int j = 0; j < DIMENSION; j++)
                        w.rows[x][j] = som->m_nodes[x]->m_weights[j];
                //printMatrix(w);

                printf("BATCH-  epoch: %d   R: %.2f \n", (epochs-1), R);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Scatter(idx_vector, chunksize, MPI_INT,
                        idx_vector_scattered, chunksize, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&R, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast((void *)w.data, NUM_SOM_NODES*DIMENSION, MPI_FLOAT,
                      0, MPI_COMM_WORLD);

            //fprintf(stderr,"[Node %d]: %s, idx_vector_scattered? %d %d %d  \n", myid, myname, idx_vector_scattered[0], idx_vector_scattered[1], idx_vector_scattered[2]);
            //fprintf(stderr,"[Node %d]: %s, R = %f \n", myid, myname, R);
            //fprintf(stderr,"[Node %d]: %s, main FEATURES %0.2f %0.2f %0.2f  \n", myid, myname, f.rows[0][0], f.rows[0][1], f.rows[0][2]);
            //fprintf(stderr,"[Node %d]: %s, main som %0.2f %0.2f %0.2f  \n", myid, myname, som->m_nodes[0]->m_weights[0], som->m_nodes[0]->m_weights[1], som->m_nodes[0]->m_weights[2]);
            //fprintf(stderr,"[Node %d]: %s, main w.rows %0.2f %0.2f %0.2f  \n", myid, myname, w.rows[0][0], w.rows[0][1], w.rows[0][2]);

            ////train_batch(som, f, R);
            ////train_batch2(som, f, R);

            //NUMERATOR_PER_VECTOR_T numerator_per_vector;
            //numerator_per_vector= NUMERATOR_PER_VECTOR_T (chunksize,
            //vector<vector<float> > (NUM_SOM_NODES,
            //vector<float>(DIMENSION, 0.0)));

            //DENOMINATOR_PER_VECTOR_T denominator_per_vector;
            //denominator_per_vector = DENOMINATOR_PER_VECTOR_T (chunksize,
            //vector<vector<float> > (NUM_SOM_NODES,
            //vector<float>(DIMENSION, 0.0)));

            MR_train_batch(mr, som, f, w, idx_vector_scattered,
                           R, argc, argv, myid, myname, numprocs);
            //numerator_per_vector, denominator_per_vector);

            //numerator_per_vector.clear();
            //denominator_per_vector.clear();
            //if (myid == 0) {
            //int num_keywords2 = mr->map(mr, &MR_update_weight, NULL);
            ////fprintf(stderr,"[Node %d]: %s, mapper2 ends! -- num_keywords = %d! -------\n", myid, myname, num_keywords2);
            //mr->print(-1, 1, 5, 3);
            //}

            //for (int x = 0; x < num_nodes; x++)
            //for (int j = 0; j < num_dimensions; j++)
            //som->m_nodes[x]->m_weights[j] = w.rows[x][j];
            //fprintf(stderr,"[Node %d]: %s, main som %0.2f %0.2f %0.2f  \n", myid, myname, som->m_nodes[0]->m_weights[0], som->m_nodes[0]->m_weights[1], som->m_nodes[0]->m_weights[2]);
            //fprintf(stderr,"[Node %d]: %s, main weights2 %0.2f %0.2f %0.2f  \n", myid, myname, w.rows[0][0], w.rows[0][1], w.rows[0][2]);
            //if (myid == 0) {
            //MPI_Bcast((void *)w.data, num_nodes*num_dimensions, MPI_FLOAT, 0, MPI_COMM_WORLD);
            ////for (int x = 0; x < num_nodes; x++)
            ////for (int j = 0; j < num_dimensions; j++)
            ////som->m_nodes[x]->m_weights[j] = w.rows[x][j];
            //}
        } else if (TRAIN_MODE == 1) { // online
            if (myid == 0) {
                R = R0 * exp(-10.0f * (x * x) / (N * N));
                nrule = nrule0 * exp(-10.0f * (x * x) / (N * N));  //learning rule shrinks over time, for online SOM
                x++;
                train_online(som, f, R, nrule);
                printf("ONLINE-  epoch: %d   R: %.2f \n", (epochs-1), R);
            }
        }


        epochs--;
    }
    //gettimeofday(&t1_end, NULL);
    //t1_time = t1_end.tv_sec - t1_start.tv_sec + (t1_end.tv_usec - t1_start.tv_usec) / 1.e6;
    //fprintf(stderr,"\n**** Processing time: %g seconds **** \n",t1_time);

    freeMatrix(&f);
    MPI_Barrier(MPI_COMM_WORLD);

    ////for (int x = 0; x < SOM_X; x++) {
    ////for (int y = 0; y < SOM_Y; y++) {
    ////for (int j = 0; j < DIMENSION; j++)
    ////printf("%f ", som->m_nodes[(x*SOM_Y)+y]->m_weights[j]);
    ////printf("\n");
    ////}
    ////}
    ////for (int x = 0; x < SOM_X; x++) {
    ////for (int y = 0; y < SOM_Y; y++) {
    ////for (int j = 0; j < SOM_DIMEN; j++)
    ////printf("%f ", som->m_nodes[(x*SOM_Y)+y]->m_coords[j]);
    ////printf("\n");
    ////}
    //}
    //for (int x = 0; x < NUM_FEATURES; x++) {
    //for (int y = 0; y < DIMENSION; y++) {
    //printf("%f ", FEATURE[x][y]);
    //}
    //printf("\n");
    //}

    //save SOM//////////////////////////////////////////////////////////
    if (myid == 0) {
        printf("Saving SOM...\n");
        char som_map[255] = "";
        //get_file_name(argv[1], som_map);
        strcat(som_map, "result.map");
        save_2D_distance_map(som, som_map);
        printf("Done!\n");
    }

    delete mr;
    MPI_Finalize();

    return 0;
}

/*
int main(int argc, char *argv[]) {

    double time0, time1;
    int num_dimensions = DIMENSION;
    int num_vectors = NUM_FEATURES;

    //make som
    SOM *som = (SOM *)malloc(sizeof(SOM));
    som->m_nodes = VEC_NODES_T(SOM_X * SOM_Y);
    for (int x = 0; x < SOM_X*SOM_Y; x++) {
        som->m_nodes[x] = (NODE *)malloc(sizeof(NODE));
    }

    //fill weights
    for (int x = 0; x < SOM_X*SOM_Y; x++) {
        NODE *node = (NODE *)malloc(sizeof(NODE));
        node->m_weights.resize(DIMENSION);
        node->m_coords.resize(SOM_DIMEN, 0.0);
        for (int i = 0; i < DIMENSION; i++) {
            int w = 0xFFF & rand();
            w -= 0x800;
            node->m_weights[i] = (float)w / 4096.0f;
        }
        som->m_nodes[x] = node;
     }

    ////file coords (2D rect)
    for (int x = 0; x < SOM_X; x++) {
        for (int y = 0; y < SOM_Y; y++) {
            som->m_nodes[(x*SOM_Y)+y]->m_coords[0] = y * 1.0f;
            som->m_nodes[(x*SOM_Y)+y]->m_coords[1] = x * 1.0f;
        }
    }
    TMatrix f;
    f = initMatrix();
    f = createMatrix(num_vectors, num_dimensions);
    if (!validMatrix(f)) {
        printf("FATAL: not valid matrix.\n");
        exit(0);
    }

    //random features
    //for(int i = 0; i < num_vectors; i++) {
        //for(int j = 0; j < num_dimensions; j++) {
            ////int w = 0xFFF & rand();
            ////w -= 0x800;
            //f.rows[i][j] = rand() % 100 / 100.0f;
        //}
    //}

    //read feature data
    FILE *fp = fopen(argv[1],"r");
    for(int i = 0; i < num_vectors; i++) {
        for(int j = 0; j < num_dimensions; j++) {
            float tmp;
            fscanf(fp, "%f", &tmp);
            f.rows[i][j] = tmp;
        }
    }
    fclose(fp);
    //printMatrix(f);

    //random features
    //for(int i = 0; i < NUM_FEATURES; i++) {
        //for(int j = 0; j < DIMENSION; j++) {
            ////int w = 0xFFF & rand();
            ////w -= 0x800;
            //FEATURE[i][j] = rand() % 100 / 100.0f;
        //}
    //}


    //train
    float R, R0;
    R0 = SOM_X / 2.0f;
    int epochs = atoi(argv[2]);     //# of iterations
    int TRAIN_MODE = atoi(argv[3]); //batch or online
    float N = (float)epochs;
    int x = 0;  //0...N-1
    float nrule, nrule0 = 0.9f;     //learning rate factor

    //////struct timeval t1_start;
    //////struct timeval t1_end;
    //////double t1_time;
    //////gettimeofday(&t1_start, NULL);

    while (epochs) {
        if (TRAIN_MODE == 0) {

            //train_batch(som, R);
            train_batch2(som, f, R);
            //train_batch_MR(som, FEATURE,
                            //idx_vector_scattered, num_vectors,
                            //R, argc, argv, myid, myname, numprocs);
        }
        else if (TRAIN_MODE == 1) { // online
            R = R0 * exp(-10.0f * (x * x) / (N * N));
            nrule = nrule0 * exp(-10.0f * (x * x) / (N * N));  //learning rule shrinks over time, for online SOM
            x++;
            train_online(som, f, R, nrule);
            printf("ONLINE-  epoch: %d   R: %.2f \n", (epochs-1), R);
        }
        epochs--;
    }
    //////gettimeofday(&t1_end, NULL);
    //////t1_time = t1_end.tv_sec - t1_start.tv_sec + (t1_end.tv_usec - t1_start.tv_usec) / 1.e6;
    //////fprintf(stderr,"\n**** Processing time: %g seconds **** \n",t1_time);



    ////for (int x = 0; x < SOM_X; x++) {
        ////for (int y = 0; y < SOM_Y; y++) {
            ////for (int j = 0; j < DIMENSION; j++)
                ////printf("%f ", som->m_nodes[(x*SOM_Y)+y]->m_weights[j]);
            ////printf("\n");
        ////}
    ////}
    ////for (int x = 0; x < SOM_X; x++) {
        ////for (int y = 0; y < SOM_Y; y++) {
            ////for (int j = 0; j < SOM_DIMEN; j++)
                ////printf("%f ", som->m_nodes[(x*SOM_Y)+y]->m_coords[j]);
            ////printf("\n");
        ////}
    //}
    //for (int x = 0; x < NUM_FEATURES; x++) {
        //for (int y = 0; y < DIMENSION; y++) {
            //printf("%f ", FEATURE[x][y]);
        //}
        //printf("\n");
    //}

    printf("Saving SOM...\n");
    char som_map[255] = "";
    //get_file_name(argv[1], som_map);
    strcat(som_map, "result.map");
    save_2D_distance_map(som, som_map);
    printf("Done!\n");



    return 0;
}
*/

/* ------------------------------------------------------------------------ */
void train_online(SOM *som, TMatrix &f, float R, float learning_rate)
/* ------------------------------------------------------------------------ */
{
    //int num_vectors = NUM_FEATURES;
    //int num_weights = DIMENSION;
    //int num_nodes = SOM_X * SOM_Y;

    for (int n = 0; n < NUM_FEATURES; n++) {
        float *vec_normalized = normalize2(f, n);
        //get best node using d_k (t) = || x(t) = w_k (t) || ^2
        //and d_c (t) == min d_k (t)
        NODE *bmu_node = get_bmu_node(som, vec_normalized);
        float *p1 = get_coords(bmu_node);
        if (R <= 1.0f) { //adjust BMU node only
            updatew_online(bmu_node, vec_normalized, learning_rate);
        } else { //adjust weight vectors of the neighbors to BMU node
            for (int k = 0; k < NUM_SOM_NODES; k++) { //adjust weights of all K nodes if dist <= R
                float *p2 = get_coords(som->m_nodes[k]);
                float dist = 0.0f;
                //dist = sqrt((x1-y1)^2 + (x2-y2)^2 + ...)  distance to node
                for (int p = 0; p < DIMENSION; p++)
                    dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
                dist = sqrt(dist);
                //dist = sqroot(dist);
                if (TRAIN_OPTION == 1 && dist > R)
                    continue;
                ////Gaussian neighborhood function
                float neighborhood_func = exp(-(1.0f * dist * dist) / (R * R));
                updatew_online(som->m_nodes[k], vec_normalized, learning_rate * neighborhood_func);
            }
        }
    }
}

/* ------------------------------------------------------------------------ */
void train_batch(SOM *som, TMatrix &f, float R)
/* ------------------------------------------------------------------------ */
{
    float *numerators = (float *)malloc(DIMENSION * SZFLOAT);
    float *denominators = (float *)malloc(DIMENSION * SZFLOAT);
    float *new_weights = (float *)malloc(DIMENSION * SZFLOAT);

    for (int k = 0; k < NUM_SOM_NODES; k++) {  //adjust weights of all K nodes if dist <= R
        float *p2 = get_coords(som->m_nodes[k]);

        float neighborhood_function=0.0f;
        for (int w = 0; w < DIMENSION; w++) {
            numerators[w] = 0.0f;
            denominators[w] = 0.0f;
            new_weights[w] = 0.0f;
        }

        for (int n = 0; n < NUM_FEATURES; n++) {
            //Compute distances and select winning node (BMU)
            float *vec_normalized = normalize2(f, n);

            //Get best node using d_k (t) = || x(t) = w_k (t_0) || ^2
            //and d_c (t) == min d_k (t)
            //First BMU is selected. No voting.
//bottleneck///////////////////////////////////////////////
            NODE *bmu_node = get_bmu_node(som, vec_normalized);
//bottleneck///////////////////////////////////////////////

            const float *p1 = get_coords(bmu_node);
            float dist = 0.0f;

            //dist = sqrt((x1-y1)^2 + (x2-y2)^2 + ...)  distance to node
            for (int p = 0; p < DIMENSION; p++)
                dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
            dist = sqrt(dist);

            if (TRAIN_OPTION == 1 && dist > R)
                continue;

            //Gaussian neighborhood function
            neighborhood_function = exp(-(1.0f * dist * dist) / (R * R));

            //new weight, w_k(t_f) = (SUM(h_ck (t') * x(t')) / (SUM()h_ck(t'))
            for (int w = 0; w < DIMENSION; w++) {
                numerators[w] += 1.0f * neighborhood_function * vec_normalized[w];
                denominators[w] += neighborhood_function;
            }
            for (int w = 0; w < DIMENSION; w++)
                if (denominators[w] != 0)
                    new_weights[w] = numerators[w] / denominators[w];
        } //for_vectors

        //Update weights
        updatew_batch(som->m_nodes[k], new_weights);
    } //for_nodes

    free(numerators);
    free(denominators);
    free(new_weights);
}

/* ------------------------------------------------------------------------ */
void train_batch2(SOM* som, TMatrix &f, float R)
/* ------------------------------------------------------------------------ */
{
    //int num_vectors = NUM_FEATURES;
    //int num_weights = DIMENSION;
    //int num_nodes = SOM_X * SOM_Y;

    typedef vector<vector<vector<float> > > numerator_per_vector_T;
    numerator_per_vector_T numerator_per_vector;
    numerator_per_vector = numerator_per_vector_T (NUM_FEATURES,
                           vector<vector<float> > (NUM_SOM_NODES,
                                   vector<float>(DIMENSION, 0.0)));

    typedef vector<vector<vector<float> > > denominator_per_vector_T;
    denominator_per_vector_T denominator_per_vector;
    denominator_per_vector = denominator_per_vector_T (NUM_FEATURES,
                             vector<vector<float> > (NUM_SOM_NODES,
                                     vector<float>(DIMENSION, 0.0)));

    for (int n = 0; n < NUM_FEATURES; n++) {

        //printf("orig       %f %f %f \n", FEATURE[n][0],FEATURE[n][1],FEATURE[n][2]);
        float *vec_normalized = normalize2(f, n);
        //printf("normalized %f %f %f \n", vec_normalized[0],vec_normalized[1],vec_normalized[2]);

        NODE *bmu_node = get_bmu_node(som, vec_normalized);
        //cout << bmu_node->m_coords[0] << " " << bmu_node->m_coords[1] << endl;

        float *p1 = get_coords(bmu_node);
        //printf("coord 1 %f %f \n", p1[0], p1[1]);


        for (int k = 0; k < NUM_SOM_NODES; k++) {
            float *p2 = get_coords(som->m_nodes[k]);
            //printf("coord 2 %f %f \n", p2[0], p2[1]);
            float dist = 0.0f;
            for (int p = 0; p < DIMENSION; p++)
                dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
            dist = sqrt(dist);

            float neighborhood_function = exp(-(1.0f * dist * dist) / (R * R));

            for (int w = 0; w < DIMENSION; w++) {
                numerator_per_vector[n][k][w] += 1.0f * neighborhood_function * vec_normalized[w];
                denominator_per_vector[n][k][w] += neighborhood_function;
            }
        }
    }

    //update w-dimensinal weights for each node
    float *new_w_vectors = (float *)malloc(DIMENSION * SZFLOAT);
    for (int i = 0; i < DIMENSION; i++)
        new_w_vectors[i] = 0.0f;

    for (int k = 0; k < NUM_SOM_NODES; k++) {
        for (int w = 0; w < DIMENSION; w++) {
            float temp_numer = 0.0f;
            float temp_demon = 0.0f;
            for (int n = 0; n < NUM_FEATURES; n++) {
                temp_numer += numerator_per_vector[n][k][w];
                temp_demon += denominator_per_vector[n][k][w];
                //printf("temp_numer, temp_demon = %f %f\n", temp_numer, temp_demon);
            }

            if (temp_demon != 0)
                new_w_vectors[w] = temp_numer / temp_demon;
            else
                new_w_vectors[w] = 0.0f;
        }
        //printf("%f %f %f\n", som->m_nodes[k]->m_weights[0], som->m_nodes[k]->m_weights[1], som->m_nodes[k]->m_weights[2]);
        //printf("%f %f %f\n", new_w_vectors[0], new_w_vectors[1], new_w_vectors[2]);
        updatew_batch(som->m_nodes[k], new_w_vectors);
        //printf("%f %f %f\n", som->m_nodes[k]->m_weights[0], som->m_nodes[k]->m_weights[1], som->m_nodes[k]->m_weights[2]);
    }
    free(new_w_vectors);
    numerator_per_vector.clear();
    denominator_per_vector.clear();
}

/* ------------------------------------------------------------------------ */
void MR_train_batch(MAPREDUCE_NS::MapReduce *mr, SOM *som, TMatrix &f,
                    TMatrix &w, int *idx_vector_scattered,
                    float R, int argc, char* argv[], int myid,
                    char *myname, int numprocs)
//NUMERATOR_PER_VECTOR_T &numer_vec,
//DENOMINATOR_PER_VECTOR_T &denom_vec)
/* ------------------------------------------------------------------------ */
{
    //int num_som_nodes = SOM_X*SOM_Y;
    //int num_weights_per_node = DIMENSION;
    //int num_vectors = numvec;
    int new_num_vectors = NUM_FEATURES / numprocs; //chunnk size

    GIFTBOX gfbox;
    gfbox.som = som;
    //gfbox.m_nodes = &som->m_nodes;
    gfbox.myid = myid;
    gfbox.myname = myname;
    //gfbox.numprocs = numprocs;
    gfbox.new_num_vectors = new_num_vectors;
    //gfbox.num_som_nodes = NUM_SOM_NODES;
    //gfbox.num_weights_per_node = DIMENSION;
    gfbox.f_vectors = &f;
    gfbox.w_vectors = &w;
    //gfbox.numer_vec = numer_vec;
    //gfbox.denom_vec = denom_vec;
    //gfbox.w_vectors = w_vectors;
    //gfbox.new_weight_vectors = new_weight_vectors;
    gfbox.R = R;
    gfbox.idx_start = idx_vector_scattered[0];


    //double tstart = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);


    //fprintf(stderr,"[Node %d]: %s train_batch_MR FEATURES %0.2f %0.2f %0.2f **** \n", myid, myname, f.rows[0][0],f.rows[0][1],f.rows[0][2]);
    int num_keywords = mr->map(numprocs, &MR_compute_weight, &gfbox);
    //fprintf(stderr,"[Node %d]: %s, mapper1 ends! -- num_keywords = %d! -------\n", myid, myname, num_keywords);
    //mr->print(-1, 1, 5, 3);

    int total_num_KM = mr->collate(NULL);     //aggregates a KeyValue object across processors and converts it into a KeyMultiValue object.
    //int total_num_KM = mr->clone();         //converts a KeyValue object directly into a KeyMultiValue object.
    //fprintf(stderr,"[Node %d]: %s ------ COLLATE ends! total_num_KM = %d -------\n", myid, myname, total_num_KM);
    //mr->print(-1, 1, 5, 3);

    //MPI_Barrier(MPI_COMM_WORLD);

    int num_unique = mr->reduce(&MR_accumul_weight, (void *)NULL);
    ////int num_unique = mr->reduce(&accumul_weight, &gfbox);
    //fprintf(stderr,"[Node %d]: %s ------ reducer ends! total_num_KM = %d -------\n", myid, myname, num_unique);
    //mr->print(-1, 1, 5, 3);

    MPI_Barrier(MPI_COMM_WORLD);

    //gfbox.flag = 0;
    //int num_keywords2 = mr->map(mr, &MR_update_weight, &gfbox);
    //fprintf(stderr,"[Node %d]: %s, mapper2 ends! -- num_keywords = %d! -------\n", myid, myname, num_keywords2);
    ////mr->print(-1, 1, 5, 3);

    mr->gather(1);
    //mr->print(-1, 1, 5, 3);

    int num_keywords2 = mr->map(mr, &MR_update_weight, &gfbox);
    //fprintf(stderr,"[Node %d]: %s, mapper2 ends! -- num_keywords = %d! -------\n", myid, myname, num_keywords2);
    mr->print(-1, 1, 5, 3);

    //double tstop = MPI_Wtime();

    //delete mr;
}

/* ------------------------------------------------------------------------ */
void updatew_online(NODE *node, float *vec, float learning_rate_x_neighborhood_func)
/* ------------------------------------------------------------------------ */
{
    for (int w = 0; w < DIMENSION; w++)
        node->m_weights[w] += learning_rate_x_neighborhood_func * (vec[w] - node->m_weights[w]);
}

/* ------------------------------------------------------------------------ */
void updatew_batch(NODE *node, float *new_w)
/* ------------------------------------------------------------------------ */
{
    for (int w = 0; w < DIMENSION; w++) {
        if (new_w[w] > 0) //????
            node->m_weights[w] = new_w[w];
    }
}

///* ------------------------------------------------------------------------ */
//float *normalize(int n)
///* ------------------------------------------------------------------------ */
//{
//float *m_data = (float *)malloc(SZFLOAT*DIMENSION);

//switch (NORMALIZE) {
//default:
//case 0:
//for (int x = 0; x < DIMENSION; x++) {
//m_data[x] = FEATURE[n][x];
////printf("%f %f\n", FEATURE[n][x], m_data[x]);
//}
//break;
//}

//return m_data;
//}

/* ------------------------------------------------------------------------ */
float *normalize2(TMatrix &f, int n)
/* ------------------------------------------------------------------------ */
{
    float *m_data = (float *)malloc(SZFLOAT*DIMENSION);
    //printf("%d %d\n", f.m, f.n);

    switch (NORMALIZE) {
    default:
    case 0:
        for (int x = 0; x < DIMENSION; x++) {
            m_data[x] = f.rows[n][x];
            //printf("%f %f\n", f.rows[n][x], m_data[x]);
        }
        break;
    }

    return m_data;
}

/* ------------------------------------------------------------------------ */
NODE *get_bmu_node(SOM *som, float *fvec)
/* ------------------------------------------------------------------------ */
{
    NODE *pbmu_node = som->m_nodes[0];
    float mindist = get_distance(fvec, 0, pbmu_node->m_weights);
    float dist;

    for (int x = 0; x < NUM_SOM_NODES; x++) {
        if ((dist = get_distance(fvec, 0, som->m_nodes[x]->m_weights)) < mindist) {
            mindist = dist;
            pbmu_node = som->m_nodes[x];
        }
    }

    //Can add a feature for voting among BMUs.

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
    case 0:   //euclidian
        for (int w = 0; w < DIMENSION; w++)
            distance += (vec[w] - wvec[w]) * (vec[w] - wvec[w]);

        return sqrt(distance);
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
    case 0:   //euclidian
        for (int w = 0; w < DIMENSION; w++)
            distance += (vec[w] - wvec[w]) * (vec[w] - wvec[w]);

        return sqrt(distance);
    }
}

/* ------------------------------------------------------------------------ */
float *get_wvec(SOM *som, int n)
/* ------------------------------------------------------------------------ */
{
    float *weights = (float *)malloc(SZFLOAT*DIMENSION);
    for (int i = 0; i < DIMENSION; i++)
        weights[i] = som->m_nodes[n]->m_weights[i];

    return weights;
}

/* ------------------------------------------------------------------------ */
float *get_coords(NODE *node)
/* ------------------------------------------------------------------------ */
{
    float *coords = (float *)malloc(SZFLOAT);
    for (int i = 0; i < SOM_DIMEN; i++)
        coords[i] = node->m_coords[i];

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
                for (int m = 0; m < NUM_SOM_NODES; m++) {
                    NODE *node = get_node(som, m);
                    if (node == pnode)
                        continue;
                    float tmp = 0.0;
                    for (int x = 0; x < D; x++)
                        tmp += pow(*(get_coords(pnode) + x) - *(get_coords(node) + x), 2.0f);
                    tmp = sqrt(tmp);
                    if (tmp <= min_dist) {
                        nodes_number++;
                        dist += get_distance2(node->m_weights, 0, pnode->m_weights);
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
    return som->m_nodes[n];
}

/* ------------------------------------------------------------------------ */
void MR_compute_weight(int itask, MAPREDUCE_NS::KeyValue *kv, void *ptr)
/* ------------------------------------------------------------------------ */
{
    GIFTBOX *gb = (GIFTBOX *) ptr;
    TMatrix f = *(gb->f_vectors);

    //NUMERATOR_PER_VECTOR_T *numerator_per_vector = gb->numer_vec;
    //DENOMINATOR_PER_VECTOR_T *denominator_per_vector = gb->denom_vec;
    NUMERATOR_PER_VECTOR_T numerator_per_vector;
    numerator_per_vector= NUMERATOR_PER_VECTOR_T (gb->new_num_vectors,
                          vector<vector<float> > (NUM_SOM_NODES,
                                  vector<float>(DIMENSION, 0.0)));

    DENOMINATOR_PER_VECTOR_T denominator_per_vector;
    denominator_per_vector = DENOMINATOR_PER_VECTOR_T (gb->new_num_vectors,
                             vector<vector<float> > (NUM_SOM_NODES,
                                     vector<float>(DIMENSION, 0.0)));

    //fprintf(stderr,"[Node %d]: %s new_num_vectors %d \n", gb->myid, gb->myname, gb->new_num_vectors);
    for (int n = 0; n < gb->new_num_vectors; n++) {

        float *vec_normalized = normalize2(f, gb->idx_start+n); //n-th feature vector in the scatered list.
        //fprintf(stderr,"[Node %d]: %s MR_compute_weight FEATURES %0.2f %0.2f %0.2f **** \n", gb->myid, gb->myname, f.rows[0][0],f.rows[0][1],f.rows[0][2]);

//bottleneck////////////////////////////////////////////////////////////
        //get the best matching unit
        NODE *bmu_node = get_bmu_node(gb->som, vec_normalized);
        //NODE *bmu_node = get_bmu_node2(gb->w_vectors, vec_normalized);
//bottleneck////////////////////////////////////////////////////////////

        //get the coords for the BMU
        float *p1 = get_coords(bmu_node);

        for (int k = 0; k < NUM_SOM_NODES; k++) {
            NODE *tp = gb->som->m_nodes[k];
            float *p2 = get_coords(tp);
            //float *p2 = get_coords((*gb->m_nodes)[k]);

            //fprintf(stderr,"[Node %d]: %s %0.2f %0.2f %0.2f **** \n", gb->myid, gb->myname, p2[0], p2[1], p2[2]);
            float dist = 0.0f;
            for (int p = 0; p < DIMENSION; p++)
                dist += (p1[p] - p2[p]) * (p1[p] - p2[p]);
            dist = sqrt(dist);
            //fprintf(stderr,"[Node %d]: %s %0.2f **** \n", gb->myid, gb->myname, dist);

            float neighborhood_function=0.0f;
            neighborhood_function = exp(-(1.0f * dist * dist) / (gb->R * gb->R));

            for (int w = 0; w < DIMENSION; w++) {
                //(*numerator_per_vector)[n][k][w] += 1.0f * neighborhood_function * vec_normalized[w];
                //(*denominator_per_vector)[n][k][w] += neighborhood_function; // = neighborhood_function * 3
                numerator_per_vector[n][k][w] += 1.0f * neighborhood_function * vec_normalized[w];
                denominator_per_vector[n][k][w] += neighborhood_function; // = neighborhood_function * 3
                //fprintf(stderr,"[Node %d]: %s, %d %d, %0.2f, %0.2f **** \n", gb->myid, gb->myname, n, w, numerator_per_vector[n][k][w], denominator_per_vector[n][k][w]);
            }
        }
    }

    //for (int n = 0; n < gb->new_num_vectors; n++)
    //for (int k = 0; k < gb->num_som_nodes; k++)
    //printf("%f %f %f\n", numerator_per_vector[n][k][0], numerator_per_vector[n][k][1], numerator_per_vector[n][k][2]);
    //printf("\n");
    //for (int n = 0; n < gb->new_num_vectors; n++)
    //for (int k = 0; k < gb->num_som_nodes; k++)
    //printf("%f %f %f\n", numerator_per_vector[n][k][0], numerator_per_vector[n][k][1], numerator_per_vector[n][k][2]);


    //update w-dimensinal weights for each node
    //float *sum_numer = (float *)malloc(SZFLOAT * gb->num_weights_per_node);
    //float *sum_demon = (float *)malloc(SZFLOAT * gb->num_weights_per_node);
    float sum_numer[DIMENSION];
    float sum_demon[DIMENSION];
    for (int k = 0; k < NUM_SOM_NODES; k++) {
        for (int w = 0; w < DIMENSION; w++) {
            float temp_numer = 0.0f;
            float temp_demon = 0.0f;
            for (int n = 0; n < gb->new_num_vectors; n++) {
                temp_numer += numerator_per_vector[n][k][w];
                temp_demon += denominator_per_vector[n][k][w];
            }
            sum_numer[w] = temp_numer; //local sum vector for k-th node
            sum_demon[w] = temp_demon; //local sum vector for k-th node
        }

        for (int w = 0; w < DIMENSION; w++) {
            //char weightnum[MAX_STRING_LEN];
            //sprintf(weightnum, "N %d %d", k, w);
            ////cout << weightnum << " " << sum_numer[w] << endl;

            ////m_nodes[k]->updatew_batch(new_weights);
            //char byte_key[strlen(weightnum)+1], byte_value[SZFLOAT], byte_value2[SZFLOAT];
            ////float f_key, f_value;
            //memcpy(byte_key, &weightnum, strlen(weightnum)+1);
            //memcpy(byte_value, &sum_numer[w], SZFLOAT);
            //kv->add(byte_key, strlen(weightnum)+1, byte_value, SZFLOAT);

            //sprintf(weightnum, "D %d %d", k, w);
            ////cout << weightnum << " " << sum_demon[w] << endl;
            //memcpy(byte_key, &weightnum, strlen(weightnum)+1);
            //memcpy(byte_value2, &sum_demon[w], SZFLOAT);
            //kv->add(byte_key, strlen(weightnum)+1, byte_value2, SZFLOAT);

            char weightnum[MAX_STRING_LEN];
            sprintf(weightnum, "%d %d", k, w);

            /////////////////////////////////////
            //should think about key byte align!
            /////////////////////////////////////
            int sizeoffloat = SZFLOAT;
            char byte_key[strlen(weightnum)+1];
            char byte_value[SZFLOAT];
            char byte_value2[SZFLOAT];
            char byte_value_cancat[SZFLOAT*2];
            //char byte_value_cancat2[sizeof(int)*2];
            memcpy(byte_key, &weightnum, strlen(weightnum)+1);
            memcpy(byte_value, &sum_numer[w], SZFLOAT);
            memcpy(byte_value2, &sum_demon[w], SZFLOAT);
            for (int i = 0; i < (int)SZFLOAT; i++) {
                byte_value_cancat[i] = byte_value[i];
                byte_value_cancat[i+SZFLOAT] = byte_value2[i];
            }//total 4*2 = 8bytes for two floats.

            ////DEBUG
            //printf("orig floats = %g, %g\n", sum_numer[w], sum_demon[w]);
            //char *c1 = (char *)malloc(SZFLOAT);
            //char *c2 = (char *)malloc(SZFLOAT);
            //for (int i = 0; i < (int)SZFLOAT; i++) {
            //c1[i] = byte_value_cancat[i];
            //c2[i] = byte_value_cancat[i+SZFLOAT];
            //}
            //printf("parsed floats = %g, %g\n", *(float *)c1, *(float *)c2);

            kv->add(byte_key, strlen(weightnum)+1, byte_value_cancat, SZFLOAT*2);

            //try encoded key ==> fail!
            //char byte_value3[sizeof(int)];
            //char byte_value4[sizeof(int)];
            //memcpy(byte_value3, &k, sizeof(int));
            //memcpy(byte_value4, &w, sizeof(int));
            //for (int i = 0; i < (int)sizeof(int); i++) {
            //byte_value_cancat2[i] = byte_value3[i];
            //byte_value_cancat2[i+sizeof(int)] = byte_value4[i];
            //}
            //kv->add(byte_value_cancat2, sizeof(int)*2, byte_value_cancat, SZFLOAT*2);
        }
    }
    //free(sum_numer);
    //free(sum_demon);
    numerator_per_vector.clear();
    denominator_per_vector.clear();
    //fprintf(stderr,"[Node %d]: %s end mapper **** \n", gb->myid, gb->myname);
}


/* ------------------------------------------------------------------------ */
void MR_accumul_weight(char *key, int keybytes, char *multivalue,
                       int nvalues, int *valuebytes, MAPREDUCE_NS::KeyValue *kv,
                       void *ptr)
/* ------------------------------------------------------------------------ */
{
    GIFTBOX *gb = (GIFTBOX *) ptr;

    // a SZFLOAT*2-byte structure for two floats => two float values
    vector<float> vec_f_numer(nvalues, 0.0);
    vector<float> vec_f_denom(nvalues, 0.0);
    for (int j = 0; j < nvalues; j++) {
        char *c1 = (char *)malloc(SZFLOAT);
        char *c2 = (char *)malloc(SZFLOAT);
        for (int i = 0; i < (int)SZFLOAT; i++) {
            c1[i] = multivalue[i];
            c2[i] = multivalue[i+SZFLOAT];
        }
        //printf("parsed floats = %g, %g\n", *(float *)c1, *(float *)c2);
        vec_f_numer[j] = *(float *)c1;
        vec_f_denom[j] = *(float *)c2;
        //printf("%g ",*(float *) multivalue);
        multivalue += SZFLOAT*2;
    }
    //printf("summed %s, %f, %f, %f, %f\n", key, vec_f_numer[0],vec_f_numer[0],vec_f_denom[0],vec_f_denom[0]);
    //fprintf(stderr,"[Node %d]: %s reduce %s, %f, %f \n", gb->myid, gb->myname, key, vec_f_numer[0],vec_f_numer[0],vec_f_denom[0],vec_f_denom[0]);
    //new weights for node K and dimension D
    //cout << accumulate(vec_f_numer.begin(), vec_f_numer.end(), 0.0f) << " ";
    //cout << accumulate(vec_f_denom.begin(), vec_f_denom.end(), 0.0f) << endl;
    float new_weight = accumulate(vec_f_numer.begin(), vec_f_numer.end(), 0.0f ) /
                       accumulate(vec_f_denom.begin(), vec_f_denom.end(), 0.0f );

    //m_nodes[K]->updatew_batch_index(new_weight, W);
    //(*gb->m_nodes)[K]->updatew_batch_index(new_weight, W);
    //gb->new_w_vectors[K][W] = new_weight;
    char byte_value[SZFLOAT];
    memcpy(byte_value, &new_weight, SZFLOAT);

    kv->add(key, keybytes, byte_value, SZFLOAT);
    //(*gb->m_nodes)[K]->updatew_batch_index(new_weight, W);
    //gb->weight_vectors[K][W] = new_weight;

    vec_f_numer.clear();
    vec_f_denom.clear();
    //fprintf(stderr,"[Node %d]: %s, %s, %d, %d \n", gb->myid, gb->myname, key, keybytes, nvalues);

    //kv->add(key, keybytes, (char *) &nvalues, sizeof(int));
}

/* ------------------------------------------------------------------------ */
void MR_update_weight(uint64_t itask, char *key, int keybytes, char *value,
                      int valuebytes, MAPREDUCE_NS::KeyValue *kv, void *ptr)
/* ------------------------------------------------------------------------ */
{
    GIFTBOX *gb = (GIFTBOX *) ptr;
    //char *whitespace = " \t\n\f\r\0";
    char *whitespace = " ";
    char *key_tokens = strtok(key, whitespace);
    int K = atoi(key_tokens);
    key_tokens = strtok(NULL, whitespace);
    int W = atoi(key_tokens);
    float new_weight = *(float *)value;
    ////if (gb->flag) {
    //if (new_weight > 0) {
    NODE *tp = gb->som->m_nodes[K];
    updatew_batch_index(tp, new_weight, W);
    //}
    //TMatrix w = *(gb->w_vectors);
    //w.rows[K][W] = new_weight;
    ////}
    ////else
    ////kv->add(key, keybytes, value, valuebytes);
    //printf("%d %d %f\n", K, W, new_weight);
}

/* ------------------------------------------------------------------------ */
void updatew_batch_index(NODE *node, float new_weight, int w)
/* ------------------------------------------------------------------------ */
{
    if (new_weight > 0)
        node->m_weights[w] = new_weight;
}

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

/* ------------------------------------------------------------------------ */
TMatrix createMatrix(const unsigned int rows, const unsigned int cols)
/* ------------------------------------------------------------------------ */
{
    TMatrix           matrix;
    unsigned long int m, n;
    unsigned int      i;

    m = rows;
    n = cols;
    matrix.m    = rows;
    matrix.n    = cols;
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
void freeMatrix (TMatrix *matrix)
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
int validMatrix (TMatrix matrix)
/* ------------------------------------------------------------------------ */
{
    if ((matrix.data == NULL) || (matrix.rows == NULL) ||
            (matrix.m == 0) || (matrix.n == 0))
        return 0;
    else return 1;
}


/* ------------------------------------------------------------------------ */
TMatrix initMatrix()
/* ------------------------------------------------------------------------ */
{
    TMatrix matrix;

    matrix.m = 0;
    matrix.n = 0;
    matrix.data = NULL;
    matrix.rows = NULL;

    return matrix;
}


/* ------------------------------------------------------------------------ */
void printMatrix(TMatrix A)
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
