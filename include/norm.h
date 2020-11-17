#ifndef __NORM_H_
#define __NORM_H_

#include <immintrin.h>

float norm(float*, int);
float vect_norm(float*, int);

// avx utilities
__m256 _mm256_abs_ps(__m256);

#endif // __NORM_H_
