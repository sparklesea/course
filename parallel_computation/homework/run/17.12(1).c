#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int main(int argc, char *argv[])
{
    omp_set_num_threads(atoi(argv[1]));

    int m = 1000;
    int n = atoi(argv[2]);
    int(*a)[n] = (int(*)[n])malloc(sizeof(int) * m * n);
    int *b = (int *)malloc(sizeof(int) * n);
    int *c = (int *)malloc(sizeof(int) * m);
    for (int i = 0; i < n; i++)
    {
        b[i] = 1;
        for (int j = 0; j < m; j++){
            a[j][i] = 1;
        }
    }

    double start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        c[i] = 0;
        #pragma omp parallel for reduction(+:c[i])
        for (int j = 0; j < n; j++){
            c[i] += a[i][j] * b[j];
        }
    }

    double finish = omp_get_wtime();
    double for_time = finish - start;

    for (int i = 0; i < m;i++){
        if (c[i] != n)
        {
            printf("Fail!\n");
            return 0;
        }
    }
    printf("Processing time:%7.10f\n", for_time);
}
