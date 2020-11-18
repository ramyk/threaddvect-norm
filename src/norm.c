#include "norm.h"
#include <immintrin.h>
#include <math.h>
#include <pthread.h>

#ifndef NB_THREADS
#define NB_THREADS 6
#endif

float norm(float* U, int n)
{
    float du = 0;
    for (int i = 0; i < n; i++)
        du += sqrt(fabs(U[i]));
    return du;
}

// avx utilities
//
__m256 _mm256_abs_ps(__m256 vect)
{
    __m256 signbit = _mm256_set1_ps(-0.0f);
    return _mm256_andnot_ps(signbit, vect);
}

float vect_norm(float* U, int n)
{
    // get the partial U norm vector
    // vector for partial U norm accumulation
    __m256 partnorm = _mm256_set1_ps(0.0f);
    __m256 placeholder; // vectorized placeholder
    for (int i = 0; i < n / 8; i++) {
        // use of loadu intrinsic is motivated
        // by the fact that at this function's
        // level, we ignore the alignment of U
        placeholder = _mm256_sqrt_ps(
            _mm256_abs_ps(
                _mm256_loadu_ps(U + 8 * i)));
        partnorm = _mm256_add_ps(partnorm, placeholder);
    }

    // get result in float array
    float partnorm_f[8] __attribute__((aligned(32)));
    _mm256_store_ps(partnorm_f, partnorm);

    // sum the partial float
    // array into final sum
    float du = 0;
    for (int i = 0; i < 8; i++)
        du += partnorm_f[i];
    return du;
}

// threading utilities
//
float result[NB_THREADS];
pthread_mutex_t result_lock;
float thrd_result = 0;
pthread_t thread_ptr[NB_THREADS];
struct thread_data {
    unsigned int id;
    float* U;
    long start;
    long end;
};
struct thread_data th_array[NB_THREADS];

// per-thread scalar code
void* thread_scalarnorm(void* threadargs)
{
    struct thread_data* thrdlocal_data;
    thrdlocal_data = (struct thread_data*)threadargs;
    // shared variables correspondances
    unsigned int id = thrdlocal_data->id;
    float* U = thrdlocal_data->U;
    long start = thrdlocal_data->start;
    long end = thrdlocal_data->end;

    //body of the thread
    float rslt = 0;
    for (long i = start; i < end; i++)
        rslt += sqrt(fabs(U[i]));

    // return thread result to the
    // global threads results array
    result[id] = rslt;

    pthread_exit(NULL);
    return 0;
}

// per-thread vectorial code
void* thread_vectnorm(void* threadargs)
{
    struct thread_data* thrdlocal_data;
    thrdlocal_data = (struct thread_data*)threadargs;
    unsigned int id = thrdlocal_data->id;
    float* U = thrdlocal_data->U;
    long start = thrdlocal_data->start;
    long end = thrdlocal_data->end;

    __m256 partnorm = _mm256_set1_ps(0.0f);
    __m256 placeholder;
    for (long i = start; i < end; i += 8) {
        placeholder = _mm256_sqrt_ps(
            _mm256_abs_ps(
                _mm256_loadu_ps(U + i)));
        partnorm = _mm256_add_ps(partnorm, placeholder);
    }

    float partnorm_f[8] __attribute__((aligned(32)));
    _mm256_store_ps(partnorm_f, partnorm);

    float du = 0;
    for (int i = 0; i < 8; i++)
        du += partnorm_f[i];

    result[id] = du;
    pthread_exit(NULL);
    return 0;
}

float normPar(float* U, int n, int nb_threads, int mode)
{
    float sum = 0;
    int i = 0;
    for (i = 0; i < nb_threads; i++) {
        th_array[i].id = i;
        th_array[i].U = U;
        th_array[i].start = i * n / nb_threads;
        th_array[i].end = (i + 1) * n / nb_threads;
        // create thread i
        pthread_create(thread_ptr + i, NULL,
            (mode == 0) ? thread_scalarnorm : thread_vectnorm,
            th_array + i);
    }
    // wait for every thrad to finish
    for (i = 0; i < nb_threads; i++) {
        pthread_join(thread_ptr[i], NULL);
        sum += result[i];
    }
    return sum;
}
