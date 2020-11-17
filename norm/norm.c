#include "norm.h"
#include <math.h>

float norm(float* U, int n)
{
    int du = 0;
    for (int i = 0; i < n; i++)
        du += sqrt(fabs(U[i]));
    return du;
}
