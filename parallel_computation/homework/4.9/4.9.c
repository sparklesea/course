#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define n 1000000
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define BLOCK_LOW(id, p, n) ((id) * (n) / (p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id) + 1, p, n) - 1)
#define BLOCK_SIZE(id, p, n) (BLOCK_LOW((id) + 1, p, n) - BLOCK_LOW((id), p, n))

int main(int argc, char *argv[])
{
    double elapsed_time; // 程序计算时间
    int i;
    int id;
    int index;        // 目前素数index
    int low_value;    // 块内最小值
    int high_value;   // 块内最大值
    char *marked;     // 标记数组，素数为0
    int p;            // 并行程序数
    int average_size; // 分块后每块平均大小
    int prime;
    int size;      // 块大小
    int intra_gap; // 最大块内间隔
    int max_gap;   // 最大间隔
    int recv_buffer;
    int count;
    int global_count;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    low_value = 2 + BLOCK_LOW(id, p, n - 1);
    high_value = 2 + BLOCK_HIGH(id, p, n - 1);
    size = BLOCK_SIZE(id, p, n - 1);

    average_size = (n - 1) / p;
    if ((2 + average_size) < (int)sqrt((double)n))
    {
        if (id == 0)
            printf("Too many processes\n");
        MPI_Finalize();
        exit(1);
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

    if (id == 0)
        index = 0;

    prime = 2;
    int first;
    do // 标记素数位置
    {
        if (prime * prime > low_value)
        {
            first = prime * prime - low_value;
        }
        else
        {
            if (!(low_value % prime))
                first = 0;
            else
                first = prime - (low_value % prime);
        } // 寻找第一个可能不是素数的位置

        for (i = first; i < size; i += prime)
            marked[i] = 1;

        if (id == 0) // 0号程序负责改变prime
        {
            while (marked[++index])
                ;
            prime = index + 2;
        }
        MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD); // 将prime广播给其他程序
    } while (prime * prime <= n);

    intra_gap = 0;
    int lasti = 0;
    int first_prime = 0;
    bool flag = false; // 块内有无素数
    count = 0;
    for (i = 0; i < size; i++)
        if (!marked[i]) // 找到一个素数
        {
            if (!flag)
            { // 第一个素数
                first_prime = lasti = i;
                flag = true;
            }
            else
            {
                intra_gap = MAX(intra_gap, i - lasti); // 更新gap
                lasti = i;
            }
            count++;
        }

    MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (id != 0 && flag) // 如果不是第一个程序且块内有素数，给相邻的前一个程序发送前面的间隔
    {
        MPI_Send(&first_prime, 1, MPI_INT, id - 1, 0, MPI_COMM_WORLD);
    }
    if (id != p - 1) // 如果不是最后一个程序，接受相邻的后一个程序的
    {
        MPI_Recv(&recv_buffer, 1, MPI_INT, id + 1, 0, MPI_COMM_WORLD, &status);
    }
    if (!flag) // 如果块内没有素数，给相邻的前一个程序发送size+接受的间隔
    {
        first_prime = size + recv_buffer;
        MPI_Send(&first_prime, 1, MPI_INT, id - 1, 0, MPI_COMM_WORLD);
    }
    if (id != p - 1) // 考虑块间间隔
    {
        intra_gap = MAX(intra_gap, size - lasti + recv_buffer);
    }

    MPI_Reduce(&intra_gap, &max_gap, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD); // 求总的最大间隔

    elapsed_time += MPI_Wtime();
    if (id == 0)
    {
        printf("%d primes are less than or equal to 1000000\n",
               global_count);
        printf("Total elapsed time: %10.6f\n",
               elapsed_time);
        printf("Max gap: %d\n",
               max_gap);
    }
    MPI_Finalize();
    return 0;
}
