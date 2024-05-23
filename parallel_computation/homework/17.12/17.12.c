#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int main(int argc, char *argv[])
{
    int count;
    int i, j;
    int t;
    int tid;
    int m, n;
    double start, finish, for_time;
    omp_set_num_threads(atoi(argv[2]));

    m = 100;
    n = atoi(argv[1]);
    int(*a)[n] = (int(*)[n])malloc(sizeof(int) * m * n);
    int *b = (int *)malloc(sizeof(int) * n);
    int *c = (int *)malloc(sizeof(int) * m);
    for (i = 0; i < n; i++)
    {
        b[i] = 1;
        for (j = 0; j < m; j++)
            a[j][i] = 1;
    }

    start = omp_get_wtime();
#pragma omp parallel for private(j) schedule(static)
        for (i = 0; i < m; i++)
        {
            c[i] = 0;
#pragma omp parallel for reduction(+:c[i])
            for (j = 0; j < n;j++){
                c[i] += a[i][j] * b[j];
            }
        }


    finish = omp_get_wtime();

    for_time = finish - start;
    int flag = 1;
    for (i = 0; i < m;i++){
        if (c[i] != n)
        {
            flag = 0;
            break;
        }
    }
    if(flag==0){
        printf("false\n");
    }
    else printf("processing time:%7.10f\n", for_time);
}
