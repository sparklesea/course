#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int main(int argc, char *argv[])
{
    int count;
    int i;
    int local_count;
    int samples;
    unsigned short xi[3];
    int t;
    int tid;
    double x, y;
    double start, finish, for_time;

    samples = atoi(argv[1]);
    omp_set_num_threads(atoi(argv[2]));

    start = omp_get_wtime();
    count = 0;
#pragma omp parallel private(xi, t) reduction(+:count)
{
    xi[2] = tid = omp_get_thread_num();
    t = omp_get_num_threads();

#pragma omp parallel for private(i,x,y) schedule(dynamic, samples/t/10) 
    for (i = tid*samples/t; i < (tid+1)*samples/t; i += 1)
    {
        x = erand48(xi);
        y = erand48(xi);
        if (x * x + y * y <= 1.0)
            count++;
    }
}

finish = omp_get_wtime();
for_time = finish - start;
printf("Estimate of pi:%7.10f\n", 4.0 * count / samples);
printf("processing time:%7.10f\n", for_time);
}
