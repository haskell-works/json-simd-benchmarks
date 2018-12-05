#include <stdio.h>
#include <immintrin.h>
#include <mmintrin.h>

void print256_num(__m256i var);

void print128_num(__m128i var);

void fprint256_num(FILE *file, __m256i var);

void fprint128_num(FILE *file, __m128i var);

void print_bits_64(uint64_t v);
