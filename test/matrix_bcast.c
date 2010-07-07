/* ------------------------------------------------------------------------ */
/* -- Parallel Matrix Operations - Matrix Multiplication                 -- */
/* --                                                                    -- */
/* -- E. van den Berg                                         02/10/2001 -- */
/* ------------------------------------------------------------------------ */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define TAG_MATRIX_PARTITION    0x4560

// Type definitions
typedef struct
{  unsigned int   m, n;  // Rows, cols
   double        *data;  // Data, ordered by row, then by col
   double       **rows;  // Pointers to rows in data
} TMatrix;

// Global variables
int processor_rank  = 0;
int processor_count = 1;

// Function declaractions
TMatrix createMatrix     (const unsigned int rows, const unsigned int cols);
void    freeMatrix       (TMatrix *matrix);
int     validMatrix      (TMatrix matrix);
TMatrix initMatrix       (void);
TMatrix matrixMultiply   (TMatrix A, TMatrix B);
void    doMatrixMultiply (TMatrix A, TMatrix B, TMatrix C);
void    printMatrix      (TMatrix A);


/* ------------------------------------------------------------------------ */
int main (int argc, char *argv[])
/* ------------------------------------------------------------------------ */
{  MPI_Status   status;
   TMatrix      A,B,C,D;
   unsigned int m, n, i, j, offset;
   double       time0, time1;

   A = initMatrix(); B = initMatrix(); C = initMatrix(); D = initMatrix();

   // Always use MPI_Init first
   if (MPI_Init(&argc, &argv) == MPI_SUCCESS) do
   {  // Determine which number is assigned to this processor
      MPI_Comm_size (MPI_COMM_WORLD, &processor_count);
      MPI_Comm_rank (MPI_COMM_WORLD, &processor_rank );

      if (processor_rank == 0)
      {  
          //printf ("Please enter matrix dimension n : "); scanf("%u", &n);
         
         n = 30;
         // Record starting time
         time0 = MPI_Wtime();

         // Allocate memory for matrices
         A = createMatrix(n, n);
     B = createMatrix(n, n);
     C = createMatrix(n, n);
     
     // Initialize matrices
     for (i = 0; i < n ; i++)
     {  for (j = 0; j < n; j++)
        {  A.rows[i][j] = (int)(3.0 * (rand() / (RAND_MAX + 1.0)));
               B.rows[i][j] = (int)(3.0 * (rand() / (RAND_MAX + 1.0)));
        }
     }
     if (!validMatrix(A) || !validMatrix(B) || !validMatrix(C)) n = 0;
         
         // Broadcast (send) size of matrix
         MPI_Bcast((void *)&n, 1, MPI_INT, 0, MPI_COMM_WORLD); if (n == 0) break;
         m = n / processor_count;
     
         // Broadcast (send) B matrix
         MPI_Bcast((void *)B.data, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

         // Send each process it's own part of A
     offset = n - m * (processor_count - 1);
     for (i = 1; i < processor_count; i++)
     {  MPI_Send((void *)A.rows[offset], m*n, MPI_DOUBLE,
                 i, TAG_MATRIX_PARTITION, MPI_COMM_WORLD);
        offset += m;
     }
     
     // Multiply own part of matrix A with B into already existing matrix C
     A.m = n - m * (processor_count - 1);
     doMatrixMultiply(A,B,C);
     A.m = n;
     
     // Receive part of C matrix from each process
     offset = n - m * (processor_count - 1);
     for (i = 1; i < processor_count; i++)
     {  MPI_Recv((void *)C.rows[offset], m*n, MPI_DOUBLE,
                 i, TAG_MATRIX_PARTITION, MPI_COMM_WORLD, &status);
        offset += m;
     }
     
     printMatrix(C);
     // Record finish time
     time1 = MPI_Wtime();

         // Print time statistics
     printf ("Total time using [%2d] processors : [%f] seconds\n", processor_count, time1 - time0);

      }
      else
      {
         // Broadcast (receive) size of matrix
         MPI_Bcast((void *)&n, 1, MPI_INT, 0, MPI_COMM_WORLD); if (n == 0) break;

         // Allocate memory for matrices
         m = n / processor_count;
     A = createMatrix(m, n);
     B = createMatrix(n ,n);
     
         // Broadcast (receive) B matrix
         MPI_Bcast((void *)B.data, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     MPI_Recv((void *)A.data, m*n, MPI_DOUBLE, 0, TAG_MATRIX_PARTITION, MPI_COMM_WORLD, &status);
     
     // Multiply local matrices
     C = matrixMultiply(A,B);
     
     // Send back result
     MPI_Send((void *)C.data, m*n, MPI_DOUBLE, 0, TAG_MATRIX_PARTITION, MPI_COMM_WORLD);
      }
   } while (0); // Enable break statement to be used above

   // Free matrix data
   freeMatrix(&A); freeMatrix(&B); freeMatrix(&C);

   // Wait for everyone to stop   
   MPI_Barrier(MPI_COMM_WORLD);

   // Always use MPI_Finalize as the last instruction of the program
   MPI_Finalize();

   return 0;
}


/* ------------------------------------------------------------------------ */
TMatrix createMatrix(const unsigned int rows, const unsigned int cols)
/* ------------------------------------------------------------------------ */
{  TMatrix           matrix;
   unsigned long int m, n;
   unsigned int      i;

   m = rows; n = cols;
   matrix.m    = rows;
   matrix.n    = cols;
   matrix.data = (double *) malloc(sizeof(double) * m * n);
   matrix.rows = (double **) malloc(sizeof(double *) * m);
   if (validMatrix(matrix))
   {  matrix.m = rows; 
      matrix.n = cols;
      for (i = 0; i < rows; i++)
      {  matrix.rows[i] = matrix.data + (i * cols);
      }
   }
   else
   {  freeMatrix(&matrix);
   }

   return matrix;
}


/* ------------------------------------------------------------------------ */
void freeMatrix (TMatrix *matrix)
/* ------------------------------------------------------------------------ */
{  if (matrix == NULL) return;

   if (matrix -> data) { free(matrix -> data); matrix -> data = NULL; }
   if (matrix -> rows) { free(matrix -> rows); matrix -> rows = NULL; }
   matrix -> m = 0;
   matrix -> n = 0;
}


/* ------------------------------------------------------------------------ */
int validMatrix (TMatrix matrix)
/* ------------------------------------------------------------------------ */
{  if ((matrix.data == NULL) || (matrix.rows == NULL) ||
       (matrix.m == 0) || (matrix.n == 0))
        return 0;
   else return 1;
}


/* ------------------------------------------------------------------------ */
TMatrix initMatrix()
/* ------------------------------------------------------------------------ */
{  TMatrix matrix;
 
   matrix.m = 0;
   matrix.n = 0;
   matrix.data = NULL;
   matrix.rows = NULL;

   return matrix;
}


/* ------------------------------------------------------------------------ */
TMatrix matrixMultiply(TMatrix A, TMatrix B)
/* ------------------------------------------------------------------------ */
{  TMatrix C;

   C = initMatrix();

   if (validMatrix(A) && validMatrix(B) && (A.n == B.m))
   {  C = createMatrix(A.m, B.n);
      if (validMatrix(C))
      {  doMatrixMultiply(A, B, C);
      }
   }

   return C;
}


/* ------------------------------------------------------------------------ */
void doMatrixMultiply(TMatrix A, TMatrix B, TMatrix C)
/* ------------------------------------------------------------------------ */
{  unsigned int i, j, k;
   double sum;

   for (i = 0; i < A.m; i++) // Rows
   {  for (j = 0; j < B.n; j++) // Cols
      {  sum = 0;
         for (k = 0; k < A.n; k++) sum += A.rows[i][k] * B.rows[k][j];
         C.rows[i][j] = sum;
      }
   }
}


/* ------------------------------------------------------------------------ */
void printMatrix(TMatrix A)
/* ------------------------------------------------------------------------ */
{  unsigned int i, j;

   if (validMatrix(A))
   {  for (i = 0; i < A.m; i++)
      {  for (j = 0; j < A.n; j++) printf ("%7.3f ", A.rows[i][j]);
     printf ("\n");
      }
   }
}
