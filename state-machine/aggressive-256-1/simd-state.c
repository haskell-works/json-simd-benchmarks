#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "simd.h"


void sm_process_chunk(
    uint8_t *in_buffer,
    size_t in_length,
    uint32_t *inout_state,
    uint32_t *out_phi_buffer) {  
  __m256i s = _mm256_set_epi64x(0, *inout_state, 0, *inout_state);

  size_t in_length_part = in_length / 8;

  uint8_t *buf0 = in_buffer;
  uint8_t *buf1 = in_buffer + in_length_part * 1;
  uint8_t *buf2 = in_buffer + in_length_part * 2;
  uint8_t *buf3 = in_buffer + in_length_part * 3;
  uint8_t *buf4 = in_buffer + in_length_part * 4;
  uint8_t *buf5 = in_buffer + in_length_part * 5;
  uint8_t *buf6 = in_buffer + in_length_part * 6;
  uint8_t *buf7 = in_buffer + in_length_part * 7;

  

  // uint32_t *phi0 = out_phi_buffer;
  // uint32_t *phi1 = out_phi_buffer + in_length_part * 1;
  // uint32_t *phi2 = out_phi_buffer + in_length_part * 2;
  // uint32_t *phi3 = out_phi_buffer + in_length_part * 3;
  __m256i offsets = _mm256_set_epi64x(in_length_part * 4, in_length_part * 2, in_length_part * 1, in_length_part * 0);

  __m256i s0 = _mm256_set_epi64x(0x0f0e0d0c0b0a0908, 0x00706050403020100, 0x0f0e0d0c0b0a0908, 0x00706050403020100);

  for (size_t i = 0; i < in_length_part; i += 1) {
    s = _mm256_shuffle_epi8(_mm256_i32gather_epi32(in_buffer + i, offsets, 1), s);
  }

  *inout_state = _mm256_extract_epi64(s, 0);
}
