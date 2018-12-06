#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "simd.h"

extern __m128i simd_transition_table_32[256];

void sm_process_chunk(
    uint8_t *in_buffer,
    size_t in_length,
    uint32_t *inout_state,
    uint32_t *out_phi_buffer) {  
  __m128i s = _mm_set_epi64x(0, *inout_state);

  for (size_t i = 0; i < in_length; i += 1) {
    __m128i tp = simd_transition_table_32[in_buffer[i]];
    __m128i p = _mm_shuffle_epi8(tp, s);
    out_phi_buffer[i] = _mm_extract_epi32(p, 0);
    s = _mm_shuffle_epi8(tp, s);
  }

  *inout_state = _mm_extract_epi64(s, 0);
}

void make_ib_bp_chunks(
    uint8_t state,
    uint32_t *in_phis,
    size_t phi_length,
    uint8_t *out_ibs,
    uint8_t *out_ops,
    uint8_t *out_cls) {
  __m128i ib_offset = _mm_set_epi64x(0, 5 + state * 8);
  __m128i op_offset = _mm_set_epi64x(0, 6 + state * 8);
  __m128i cl_offset = _mm_set_epi64x(0, 7 + state * 8);

  for (size_t i = 0; i < phi_length; i += 8) {
    __m256i v_8 = *(__m256i *)&in_phis[i];
    __m256i v_ib_8 = _mm256_sll_epi64(v_8, ib_offset);
    __m256i v_op_8 = _mm256_sll_epi64(v_8, op_offset);
    __m256i v_cl_8 = _mm256_sll_epi64(v_8, cl_offset);
    uint8_t all_ibs = (uint8_t)_pext_u32(_mm256_movemask_epi8(v_ib_8), 0x11111111);
    uint8_t all_ops = (uint8_t)_pext_u32(_mm256_movemask_epi8(v_op_8), 0x11111111);
    uint8_t all_cls = (uint8_t)_pext_u32(_mm256_movemask_epi8(v_cl_8), 0x11111111);

    size_t j = i / 8;
    out_ibs[j] = all_ibs;
    out_ops[j] = all_ops;
    out_cls[j] = all_cls;
  }
}
