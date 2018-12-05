#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "simd.h"

extern __m128i simd_transition_128[256];
extern __m128i simd_phi_128       [256];

void sm_process_chunk(
    uint8_t *in_buffer,
    size_t in_length,
    uint32_t *inout_state,
    uint32_t *out_phi_buffer) {  
  __m128i s = _mm_set_epi64x(0, *inout_state);

  for (size_t i = 0; i < in_length; i += 1) {
    __m128i p = _mm_shuffle_epi8(simd_phi_128[in_buffer[i]], s);
    out_phi_buffer[i] = _mm_extract_epi32(p, 0);
    s = _mm_shuffle_epi8(simd_transition_128[in_buffer[i]], s);
  }

  *inout_state = _mm_extract_epi64(s, 0);
}
