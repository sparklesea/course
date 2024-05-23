#include <stdio.h>
#include <stdlib.h>
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

    samples = atoi(argv[1]);
    omp_set_num_threads(atoi(argv[2]));
    count = 0;
#pragma omp parallel private(xi, t, i, x, y, local_count) {
    local_count = 0;
    xi[0] = atoi(argv[3]);
    xi[1] = atoi(argv[4]);
    xi[2] = tid = omp_get_thread_num();
    t = opm_get_num_threads();

    for (i = tid; i < samples; i += t)
    {
        x = erand48(xi);
        y = erand48(xi);
        if (x * x + y * y <= 1.0)
            local_count++;
    }
#pragma omp critical
    count += local_count;
}
printf("Estimate of pi:%7.10f\n", 4.0 * count / samples);

}
