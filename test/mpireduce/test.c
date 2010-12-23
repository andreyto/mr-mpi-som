#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int  argc, char* argv[])
{

    int mprocs, myrank, ista, iend,i;
    double sum,tmp;
    int imsg[4];
    double a[9];

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&mprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    
    ista = myrank *3 ;
    iend = ista +3;
    for (i=ista; i< iend; i++)
        a[i]=i+1;

    sum =0.0;
    for (i=ista; i< iend; i++)
        sum=sum + a[i];

    MPI_Reduce(********************************);

    sum = tmp;

    if( myrank == 0)
        printf("sum= %f\n", sum);

    MPI_Finalize();
    return 0;
}
