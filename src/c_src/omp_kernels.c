#include <immintrin.h>
#include <omp.h>
#include <stddef.h>

#define DO_NOT_OPTIMIZE(ptr) asm volatile("" : : "g"(ptr) : "memory")

// STREAM TRIAD Multi-Threaded via OpenMP
void c_stream_triad_omp(double* restrict a, const double* restrict b,
                        const double* restrict c, double q, size_t n,
                        size_t internal_its, int num_threads) {
    omp_set_num_threads(num_threads);

    for (size_t j = 0; j < internal_its; j++) {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i] + q * c[i];
        }
        DO_NOT_OPTIMIZE(a);
    }
}

// Peak FMA FP64 Multi-Threaded via OpenMP
double c_fma_peak_omp(size_t iterations_per_thread, int num_threads) {
    omp_set_num_threads(num_threads);
    double total_sum = 0.0;

#pragma omp parallel reduction(+ : total_sum)
    {
        __m256d v0 = _mm256_set1_pd(1.0000001);
        __m256d v1 = _mm256_set1_pd(1.0000002);
        __m256d v2 = _mm256_set1_pd(1.0000003);
        __m256d v3 = _mm256_set1_pd(1.0000004);
        __m256d v4 = _mm256_set1_pd(1.0000005);
        __m256d v5 = _mm256_set1_pd(1.0000006);
        __m256d v6 = _mm256_set1_pd(1.0000007);
        __m256d v7 = _mm256_set1_pd(1.0000008);
        __m256d v8 = _mm256_set1_pd(1.0000009);
        __m256d v9 = _mm256_set1_pd(1.0000010);
        __m256d v10 = _mm256_set1_pd(1.0000011);
        __m256d v11 = _mm256_set1_pd(1.0000012);
        __m256d mult = _mm256_set1_pd(1.0000001);

        for (size_t i = 0; i < iterations_per_thread; i++) {
            v0 = _mm256_fmadd_pd(v0, mult, mult);
            v1 = _mm256_fmadd_pd(v1, mult, mult);
            v2 = _mm256_fmadd_pd(v2, mult, mult);
            v3 = _mm256_fmadd_pd(v3, mult, mult);
            v4 = _mm256_fmadd_pd(v4, mult, mult);
            v5 = _mm256_fmadd_pd(v5, mult, mult);
            v6 = _mm256_fmadd_pd(v6, mult, mult);
            v7 = _mm256_fmadd_pd(v7, mult, mult);
            v8 = _mm256_fmadd_pd(v8, mult, mult);
            v9 = _mm256_fmadd_pd(v9, mult, mult);
            v10 = _mm256_fmadd_pd(v10, mult, mult);
            v11 = _mm256_fmadd_pd(v11, mult, mult);
        }

        v0 = _mm256_add_pd(v0, v1);
        v2 = _mm256_add_pd(v2, v3);
        v4 = _mm256_add_pd(v4, v5);
        v6 = _mm256_add_pd(v6, v7);
        v8 = _mm256_add_pd(v8, v9);
        v10 = _mm256_add_pd(v10, v11);

        v0 = _mm256_add_pd(v0, v2);
        v4 = _mm256_add_pd(v4, v6);
        v8 = _mm256_add_pd(v8, v10);

        v0 = _mm256_add_pd(v0, v4);
        v0 = _mm256_add_pd(v0, v8);

        double res[4];
        _mm256_storeu_pd(res, v0);
        total_sum += res[0];
    }

    return total_sum;
}