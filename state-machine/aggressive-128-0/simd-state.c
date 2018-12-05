#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "debug.h"

#include "simd.h"

extern uint32_t simd_transition_table_32[256];
extern uint32_t simd_phi_table_32       [256];

void sm_process_chunk(
    uint8_t *in_buffer,
    size_t in_length,
    uint32_t *inout_state,
    uint32_t *out_phi_buffer) {  
  __m128i s = _mm_set_epi64x(0, *inout_state);

  for (size_t i = 0; i < in_length; i += 1) {
    uint8_t w = in_buffer[i];
    __m128i p = _mm_shuffle_epi8(_mm_set1_epi32(simd_phi_table_32[w]), s);
    out_phi_buffer[i] = _mm_extract_epi32(p, 0);
    s = _mm_shuffle_epi8(_mm_set1_epi32(simd_transition_table_32[w]), s);

    uint32_t s32 = _mm_extract_epi32(s, 0) & 0xff;

    printf("%02zu: ", i); printf("%c ", w); printf("%d ", s32); printf("\n");

    if (i > 10)
      exit(1);
  }

  *inout_state = _mm_extract_epi64(s, 0);
}
