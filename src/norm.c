#include "norm.h"
#include <immintrin.h>
#include <math.h>
#include <stdio.h>

float norm(float* U, int n)
{
    float du = 0;
    for (int i = 0; i < n; i++)
        du += sqrt(fabs(U[i]));
    return du;
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

// avx utilities
__m256 _mm256_abs_ps(__m256 vect)
{
    __m256 signbit = _mm256_set1_ps(-0.0f);
    return _mm256_andnot_ps(signbit, vect);
}
