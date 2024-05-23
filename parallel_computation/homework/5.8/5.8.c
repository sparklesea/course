#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define BLOCK_LOW(id, p, n) ((id) * (n) / (p) / 2)
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id) + 1, p, n) - 1)
#define BLOCK_SIZE(id, p, n) (BLOCK_LOW((id) + 1, p, n) - BLOCK_LOW((id), p, n))
int main(int argc, char *argv[])
{
    int count;           /* local prime count */
    double elapsed_time; /* parallel execution time */
    int first;           /* index of first multiple */
    int global_count;    /* global prime count */
    int high_value;      /* highest value on this proc */
    int i;               /* */
    int id;              /* process id number */
    int index;           /* index of current prime */
    int low_value;       /* lowest value on this proc */
    char *marked;        /* portion of 2, ..., 'n' */
    int n;               /* sieving from 2, ..., 'n' */
    int p;               /* number of processes */
    int proc0_size;      /* size of proc 0's subarray */
    int prime;           /* current prime */
    int size;            /* elements in marked string */

    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 2)
    {
        if (id == 0)
            printf("Command line: %s <m>\n", argv[0]);
        MPI_Finalize();
        exit(1);
    }

    n = atoi(argv[1]);

    low_value = 3 + BLOCK_LOW(id, p, n - 1) * 2;
    high_value = 3 + BLOCK_HIGH(id, p, n - 1) * 2;
    size = BLOCK_SIZE(id, p, n - 1);

    int sqrt_n = (int)sqrt((double)n);

    proc0_size = (n - 1) / p;
    if ((2 + proc0_size) < (int)sqrt((double)n))
    {
        if (id == 0)
            printf("Too many processes\n");
        MPI_Finalize();
        exit(1);
    }
    // 消除广播，自己计算素数
    char *own_prime = (char *)malloc((sqrt_n + 1) * sizeof(char));
    if (own_prime == NULL)
    {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }
    for (i = 0; i <= sqrt_n; i++)
        own_prime[i] = 0;
    // 筛选出prime
    int j;
    for (i = 2; i <= sqrt_n; i += 2) // 先标记偶数
        own_prime[i] = 1;
    for (i = 3; i <= sqrt_n; i += 2)
    {
        if (own_prime[i] == 0)
        {
            for (j = 2 * i; j <= sqrt_n; j += i)
                own_prime[j] = 1;
        }
    }

    marked = (char *)malloc(size * sizeof(char));
    if (marked == NULL)
    {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    }

    for (i = 0; i < size; i++)
        marked[i] = 0;

    int block_length = 1024 * 1024; // 最外层的循环，每次对一个块内的数字进行筛选，增加cache利用率
    int block_low, block_high;
    int first_index; // 块内第一个素数index
    for (i = 0; i < size; i += block_length)
    {
        if(i==0){
            block_low = low_value;
        }else{
            block_low += block_length * 2;
        }
        block_high = MIN(high_value, low_value + block_length * 2);
        for (prime = 3; prime <= sqrt_n; prime++)
        {
            if (own_prime[prime] == 1)
                continue;
            if (prime * prime > block_low)
            {
                first = prime * prime;
            }
            else
            {
                if (!(block_low % prime))
                {
                    first = block_low;
                }
                else
                {
                    first = prime - (block_low % prime) +
                            block_low;
                }
            }
            if ((first + prime) & 1 == 1)
                first += prime;

            first_index = (first - 3) / 2 - BLOCK_LOW(id, p, n - 1);
            for (i = first; i <= high_value; i += 2 * prime)
            {
                marked[first_index] = 1;
                first_index += prime;
            }
        }
    }

    count = 0;
    for (i = 0; i < size; i++)
        if (!marked[i])
            count++;

    MPI_Reduce(&count, &global_count, 1, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    elapsed_time += MPI_Wtime();
    if (id == 0)
    {
        global_count += 1;
        printf("%d primes are less than or equal to %d\n",
               global_count, n);
        printf("Total elapsed time: %10.6f\n",
               elapsed_time);
    }

    MPI_Finalize();

    return 0;
}
