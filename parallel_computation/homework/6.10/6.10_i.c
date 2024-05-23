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

    MPI_Request *request = (MPI_Request *)malloc(p * 2 * sizeof(MPI_Request));
    send_buff = (int *)malloc(block_size * p * sizeof(int));
    recv_buff = (int *)malloc(block_size * p * sizeof(int));
    if (request == NULL || send_buff == NULL || recv_buff == NULL)
    {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }

    for (int j = 0; j < block_size; j++)
    {
        send_buff[id * block_size + j] = id * block_size + j;
    }

    elapsed_time = -MPI_Wtime();
    int cnt = 0;
    for (int j = 0; j < p; j++)
    {
        MPI_Irecv(&recv_buff[j * block_size], block_size, MPI_INT, j, 0, MPI_COMM_WORLD, &request[cnt]);
        ++cnt;
    }

    for (int j = 0; j < p; j++)
    {
        MPI_Isend(&send_buff[id * block_size], block_size, MPI_INT, j, 0, MPI_COMM_WORLD, &request[cnt]);
        ++cnt;
    }
    MPI_Waitall(cnt, request, MPI_STATUS_IGNORE);

    elapsed_time += MPI_Wtime();
    if (id == 0)
    {
        /*
                for (int m = 0; m < block_size * p; m++)
                {
                    printf("%d,", recv_buff[m]);
                }
                printf("%d", p * block_size);
        */
        printf("Total elapsed time: %10.6f\n",
               elapsed_time);
    }

    MPI_Finalize();
    return 0;
}
