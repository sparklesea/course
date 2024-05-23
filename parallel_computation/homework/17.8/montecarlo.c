#include <stdlib.h>
#include <stdio.h>

int main(int argc,char** argv){
    int count;
    unsigned short xi[3];
    int i;
    int samples;
    double x, y;
    samples = atoi(argv[1]);
    xi[0] = atoi(argv[2]);
    xi[1] = atoi(argv[3]);
    xi[2] = atoi(argv[4]);
    count = 0;
    for (i = 0; i < samples;i++){
        x = erand48(xi);
        y = erand48(xi);
        if(x*x+y*y<=1.0)
            count++;
    }
    printf("Estimate of pi: %7.10f\n", 4.0 * count / samples);
}
