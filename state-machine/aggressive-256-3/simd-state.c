#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "simd.h"

extern __m256i simd_transition_phi_saturated_256[256];

void sm_process_chunk(
    uint8_t *in_buffer,
    size_t in_length,
    uint32_t *inout_state,
    uint32_t *out_phi_buffer) {  
  __m256i s = _mm256_set_epi64x(0, *inout_state, 0, *inout_state);

  size_t in_length_part = in_length / 8;

  __m256i c01 = _mm256_set1_epi32(in_length_part);
  __m256i c02 = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  __m256i c03 = _mm256_mul_epi32(c01, c02);
  __m256i c04 = _mm256_set1_epi32(0xff);

  for (size_t i = 0; i < in_length_part; i += 4) {
    __m256i x0 = _mm256_i32gather_epi32(in_buffer + i, c03, 1);

    __m256i x00 = _mm256_and_si256(x0, c04);
    __m256i x01 = _mm256_i32gather_epi32(simd_transition_phi_saturated_256, x00, 1);
    s = _mm256_shuffle_epi8(x01, s);

    __m256i x10 = _mm256_and_si256(_mm256_srli_epi32(x0, 8), c04);
    __m256i x11 = _mm256_i32gather_epi32(simd_transition_phi_saturated_256, x10, 1);
    s = _mm256_shuffle_epi8(x11, s);

    __m256i x20 = _mm256_and_si256(_mm256_srli_epi32(x0, 16), c04);
    __m256i x21 = _mm256_i32gather_epi32(simd_transition_phi_saturated_256, x20, 1);
    s = _mm256_shuffle_epi8(x21, s);

    __m256i x30 = _mm256_and_si256(_mm256_srli_epi32(x0, 24), c04);
    __m256i x31 = _mm256_i32gather_epi32(simd_transition_phi_saturated_256, x30, 1);
    s = _mm256_shuffle_epi8(x31, s);
  }

  *inout_state = _mm256_extract_epi64(s, 0);
}
