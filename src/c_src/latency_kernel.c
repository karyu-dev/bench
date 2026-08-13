#include <stddef.h>
#include <stdint.h>

#define DO_NOT_OPTIMIZE(ptr) asm volatile("" : : "g"(ptr) : "memory")

// Parcourt une chaîne de pointeurs/indices de taille N pendant 'iterations' pas
uint64_t c_pointer_chasing(const uint64_t* array, size_t iterations) {
    uint64_t p = 0;
    for (size_t i = 0; i < iterations; i++) {
        p = array[p];
    }
    DO_NOT_OPTIMIZE(p);
    return p;
}