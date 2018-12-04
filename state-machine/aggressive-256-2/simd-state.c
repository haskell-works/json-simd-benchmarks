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
    uint8_t *inout_state,
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

  // __m256i offsets = _mm256_set_epi64x(in_length_part * 4, in_length_part * 2, in_length_part * 1, in_length_part * 0);

  for (size_t i = 0; i < in_length_part; i += 1) {
    __m256i pt0 = simd_transition_phi_saturated_256[buf0[i]] & _mm256_set_epi32(0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xffffffff);
    __m256i pt1 = simd_transition_phi_saturated_256[buf1[i]] & _mm256_set_epi32(0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xffffffff, 0x00000000);
    __m256i pt2 = simd_transition_phi_saturated_256[buf2[i]] & _mm256_set_epi32(0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xffffffff, 0x00000000, 0x00000000);
    __m256i pt3 = simd_transition_phi_saturated_256[buf3[i]] & _mm256_set_epi32(0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xffffffff, 0x00000000, 0x00000000, 0x00000000);
    __m256i pt4 = simd_transition_phi_saturated_256[buf4[i]] & _mm256_set_epi32(0x00000000, 0x00000000, 0x00000000, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000);
    __m256i pt5 = simd_transition_phi_saturated_256[buf5[i]] & _mm256_set_epi32(0x00000000, 0x00000000, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000);
    __m256i pt6 = simd_transition_phi_saturated_256[buf6[i]] & _mm256_set_epi32(0x00000000, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000);
    __m256i pt7 = simd_transition_phi_saturated_256[buf7[i]] & _mm256_set_epi32(0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000);

    __m256i pt = pt0 | pt1 | pt2 | pt3 | pt4 | pt5 | pt6 | pt7;

    s = _mm256_shuffle_epi8(pt, s);
  }

  __m256i t = _mm256_set_epi32(
    0x0c0c0c0c, 0x08080808, 0x04040404, 0x00000000,
    0x0c0c0c0c, 0x08080808, 0x04040404, 0x00000000);

  s = _mm256_sub_epi32(s, t);
  
  __m256i io_s = _mm256_set1_epi32(*inout_state);

                                                                uint8_t s0 = *inout_state;
  io_s = _mm256_shuffle_epi8(                    s     , io_s); uint8_t s1 = _mm256_extract_epi8(io_s, 0);
  io_s = _mm256_shuffle_epi8(_mm256_bsrli_epi128(s, 32), io_s); uint8_t s2 = _mm256_extract_epi8(io_s, 0);
  io_s = _mm256_shuffle_epi8(_mm256_bsrli_epi128(s, 64), io_s); uint8_t s3 = _mm256_extract_epi8(io_s, 0);
  io_s = _mm256_shuffle_epi8(_mm256_bsrli_epi128(s, 96), io_s); uint8_t sn = _mm256_extract_epi8(io_s, 0);



  *inout_state = _mm256_extract_epi64(io_s, 0);
}
