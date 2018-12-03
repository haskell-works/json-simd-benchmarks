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

  size_t in_length_part = in_length / 4;

  uint8_t *buf0 = in_buffer;
  uint8_t *buf1 = in_buffer + in_length_part * 1;
  uint8_t *buf2 = in_buffer + in_length_part * 2;
  uint8_t *buf3 = in_buffer + in_length_part * 3;

  for (size_t i = 0; i < in_length_part; i += 1) {
    __m256i pt0 = simd_transition_phi_saturated_256[buf0[i]] & _mm256_set_epi32(0x00000000, 0x00000000, 0x00000000, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0xffffffff);
    __m256i pt1 = simd_transition_phi_saturated_256[buf1[i]] & _mm256_set_epi32(0x00000000, 0x00000000, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0xffffffff, 0x00000000);
    __m256i pt2 = simd_transition_phi_saturated_256[buf2[i]] & _mm256_set_epi32(0x00000000, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0xffffffff, 0x00000000, 0x00000000);
    __m256i pt3 = simd_transition_phi_saturated_256[buf3[i]] & _mm256_set_epi32(0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0xffffffff, 0x00000000, 0x00000000, 0x00000000);

    __m256i pt = pt0 | pt1 | pt2 | pt3;

    s = _mm256_shuffle_epi8(pt, s);
  }

  *inout_state = _mm256_extract_epi64(s, 0);
}
