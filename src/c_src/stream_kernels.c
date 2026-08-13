#include <stddef.h>

// Barrière mémoire : dit au compilateur que le tableau 'a' a été "consommé"
#define DO_NOT_OPTIMIZE(ptr) asm volatile("" : : "g"(ptr) : "memory")

// COPY : A = B
void c_stream_copy(double* restrict a, const double* restrict b, size_t n, size_t internal_its) {
    for (size_t j = 0; j < internal_its; j++) {
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i];
        }
        DO_NOT_OPTIMIZE(a); // Empêche GCC d'éliminer les itérations de j
    }
}

void c_stream_scale(double* restrict a, const double* restrict b, double q, size_t n, size_t internal_its) {
    for (size_t j = 0; j < internal_its; j++) {
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i] * q;
        }
        DO_NOT_OPTIMIZE(a); // Empêche GCC d'éliminer les itérations de j
    }
}

void c_stream_add(double* restrict a, const double* restrict b, const double* restrict c, size_t n, size_t internal_its) {
    for (size_t j = 0; j < internal_its; j++) {
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i] + c[i];
        }
        DO_NOT_OPTIMIZE(a); // Empêche GCC d'éliminer les itérations de j
    }
}



// TRIAD : A = B + q * C
void c_stream_triad(double* restrict a, const double* restrict b, const double* restrict c, double q, size_t n, size_t internal_its) {
    for (size_t j = 0; j < internal_its; j++) {
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i] + q * c[i];
        }
        DO_NOT_OPTIMIZE(a); // Empêche GCC d'éliminer les itérations de j
    }
}