#include "norm.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define N 100000

double now()
{
    struct timeval t;
    double f_t;
    gettimeofday(&t, NULL);
    f_t = t.tv_usec;
    f_t = f_t / 1000.0;
    f_t += t.tv_sec;
    return f_t;
}

int main(int argc, char** argv)

{
    double tm, seq_tm, vect_tm, thrd_tm, thrdvect_tm;
    float du;
    printf("---- Norm Parallel/Vectorized Computing ----\n");
    float* V = malloc(N * sizeof(float));

    // initialize the rand vector
    // trying to sometimes put negative values
    srand((unsigned int)time(NULL));
    for (long i = 0; i < N; i++) {
        V[i] = (float)rand() / (float)RAND_MAX;
        V[i] *= (rand() > RAND_MAX / 2) ? 1 : -1;
    }

    // sequential norm
    tm = now();
    du = norm(V, N);
    seq_tm = now() - tm;
    printf("Sequential norm val:\t\t%lf\n", du);
    printf("Sequential norm time (E-03 s):\t%f\n", seq_tm);

    printf("--------------------------------------------\n");

    // vectorial norm
    tm = now();
    du = vect_norm(V, N);
    vect_tm = now() - tm;
    printf("Vectorial norm val:\t\t%lf\n", du);
    printf("Vectorial norm time (E-03 s):\t%f\n", vect_tm);
    printf("Vectorial norm acceleration:\t%f\n", seq_tm / vect_tm);

    printf("--------------------------------------------\n");
}
