#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int main(int argc, char *argv[])
{
    int samples = atoi(argv[1]);
    omp_set_num_threads(atoi(argv[2]));

    unsigned short xi[3];
    xi[0] = 1;
    xi[1] = 2;
    double start = omp_get_wtime();
    int count = 0;
    #pragma omp parallel for private(xi)
    for (int i = 0; i < samples; i++)
    {
        xi[2] = omp_get_thread_num();
        double x = erand48(xi);
        double y = erand48(xi);
        #pragma omp critical
        if (x * x + y * y <= 1.0){
            count++;
        }
    }
    
    double finish = omp_get_wtime();
    double for_time = finish - start;
    printf("Estimate of pi:%7.10f\n", 4.0 * count / samples);
    printf("processing time:%7.10f\n", for_time);
}
