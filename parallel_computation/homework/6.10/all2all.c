#include <stdio.h>
#include "mpi.h"
#include "stdlib.h"

int main(int argc, char **argv)
{
    int p, id;
    int block_size;
    double elapsed_time;
    int *send_buff;
    int *recv_buff;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 2)
    {
        if (id == 0)
            printf("Command line: %s <m>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }
    block_size = atoi(argv[1]);

    send_buff = (int *)malloc(block_size * p * sizeof(int));
    recv_buff = (int *)malloc(block_size * p * sizeof(int));

    for (int i = 0; i < p; i++)
    {
        for (int j = 0; j < block_size; j++)
        {
            send_buff[i * block_size + j] = id * block_size + j;
        }
    }
    elapsed_time = -MPI_Wtime();
    MPI_Alltoall(send_buff, block_size, MPI_INT, recv_buff, block_size, MPI_INT, MPI_COMM_WORLD);
    elapsed_time += MPI_Wtime();
    if (id == 0)
    {
        /*
        for (int m = 0; m < block_size * p; m++)
        {
            printf("%d,", recv_buff[m]);
        }
        */
        printf("Total elapsed time: %10.6f\n",
               elapsed_time);
    }

    MPI_Finalize();
    return 0;
}
