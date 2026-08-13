#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#define DO_NOT_OPTIMIZE(ptr) asm volatile("" : : "g"(ptr) : "memory")

// 1. BENCHMARK DE PREDICTION DE BRANCHEMENT (Sauts forcés)
uint64_t c_branch_test(const uint8_t* cond_array, size_t n, size_t internal_its) {
    uint64_t val = 0;
    for (size_t j = 0; j < internal_its; j++) {
        for (size_t i = 0; i < n; i++) {
            if (cond_array[i]) {
                val += 3;
                asm volatile(""); // Empêche l'utilisation de CMOV et force un vrai saut (jcc)
            } else {
                val += 7;
                asm volatile(""); // Empêche l'optimisation branchless
            }
        }
        DO_NOT_OPTIMIZE(&val);
    }
    return val;
}

// 2. BENCHMARK INT64 ALU PEAK (AVX2 Vectorized Add)
uint64_t c_int_alu_peak(size_t iterations) {
    __m256i v0 = _mm256_set1_epi64x(1);
    __m256i v1 = _mm256_set1_epi64x(2);
    __m256i v2 = _mm256_set1_epi64x(3);
    __m256i v3 = _mm256_set1_epi64x(4);
    __m256i v4 = _mm256_set1_epi64x(5);
    __m256i v5 = _mm256_set1_epi64x(6);
    __m256i v6 = _mm256_set1_epi64x(7);
    __m256i v7 = _mm256_set1_epi64x(8);

    __m256i add_val = _mm256_set1_epi64x(1);

    for (size_t i = 0; i < iterations; i++) {
        v0 = _mm256_add_epi64(v0, add_val);
        v1 = _mm256_add_epi64(v1, add_val);
        v2 = _mm256_add_epi64(v2, add_val);
        v3 = _mm256_add_epi64(v3, add_val);
        v4 = _mm256_add_epi64(v4, add_val);
        v5 = _mm256_add_epi64(v5, add_val);
        v6 = _mm256_add_epi64(v6, add_val);
        v7 = _mm256_add_epi64(v7, add_val);
    }

    v0 = _mm256_add_epi64(v0, v1);
    v2 = _mm256_add_epi64(v2, v3);
    v4 = _mm256_add_epi64(v4, v5);
    v6 = _mm256_add_epi64(v6, v7);

    v0 = _mm256_add_epi64(v0, v2);
    v4 = _mm256_add_epi64(v4, v6);

    v0 = _mm256_add_epi64(v0, v4);

    uint64_t res[4];
    _mm256_storeu_si256((__m256i*)res, v0);
    return res[0];
}