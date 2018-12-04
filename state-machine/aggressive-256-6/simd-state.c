#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "simd.h"

extern uint32_t simd_phi_table_32[];
extern uint32_t simd_transition_table_32[];

void sm_process_chunk(
    uint8_t *in_buffer,
    size_t in_length,
    uint32_t *inout_state,
    uint32_t *out_phi_buffer) {  
  __m256i s = _mm256_set_epi64x(0, 0x03020100, 0, 0x03020100);

  size_t part_length = in_length;

  for (size_t i = 0; i < part_length; i += 1) {
    __m256i p = _mm256_set_epi64x(0, 0, 0, simd_phi_table_32       [in_buffer[i]]);
    __m256i t = _mm256_set_epi64x(0, 0, 0, simd_transition_table_32[in_buffer[i]]);
    out_phi_buffer[i] = _mm256_extract_epi32(_mm256_shuffle_epi8(p, s), 0);
    s = _mm256_shuffle_epi8(t, s);
  }

  *inout_state = _mm256_extract_epi64(s, 0);
}
