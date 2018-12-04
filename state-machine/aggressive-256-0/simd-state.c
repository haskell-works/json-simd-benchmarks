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

  for (size_t i = 0; i < in_length; i += 1) {
    s = _mm256_shuffle_epi8(simd_transition_phi_saturated_256[in_buffer[i]], s);
  }

  *inout_state = _mm256_extract_epi64(s, 0);
}
