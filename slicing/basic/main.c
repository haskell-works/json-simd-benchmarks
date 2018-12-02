#include "simd.h"

#include <stdio.h>
#include <string.h>
#include <immintrin.h>

int main(
    int argc,
    char **argv) {
  if (argc != 4) {
    fprintf(stderr, "./a.out <input-file> <output-ib-file> <output-bp-file>\n");
    exit(1);
  }

  char *in_filename     = argv[1];
  char *ib_out_filename = argv[2];
  char *bp_out_filename = argv[3];

  FILE *in = fopen(in_filename, "r");

  if (!in) {
    fprintf(stderr, "Failed to open input file %s\n", in_filename);
    exit(1);
  }

  FILE *ib_out = fopen(ib_out_filename, "w");
  
  if (!ib_out) {
    fprintf(stderr, "Failed to open ib output file %s\n", ib_out_filename);
    exit(1);
  }

  FILE *bp_out = fopen(bp_out_filename, "w");
  
  if (!bp_out) {
    fprintf(stderr, "Failed to open bp output file %s\n", bp_out_filename);
    exit(1);
  }

  uint8_t buffer[W8_BUFFER_SIZE];

  uint8_t *bits_of_d = malloc(W32_BUFFER_SIZE); memset(bits_of_d, 0, W32_BUFFER_SIZE);
  uint8_t *bits_of_a = malloc(W32_BUFFER_SIZE); memset(bits_of_a, 0, W32_BUFFER_SIZE);
  uint8_t *bits_of_z = malloc(W32_BUFFER_SIZE); memset(bits_of_z, 0, W32_BUFFER_SIZE);
  uint8_t *bits_of_b = malloc(W32_BUFFER_SIZE); memset(bits_of_b, 0, W32_BUFFER_SIZE);
  uint8_t *bits_of_e = malloc(W32_BUFFER_SIZE); memset(bits_of_e, 0, W32_BUFFER_SIZE);
  uint8_t *bits_of_q = malloc(W32_BUFFER_SIZE); memset(bits_of_q, 0, W32_BUFFER_SIZE);
  uint8_t *bits_of_w = malloc(W32_BUFFER_SIZE); memset(bits_of_q, 0, W32_BUFFER_SIZE);

  uint8_t result_ib[W8_BUFFER_SIZE / 8];
  uint8_t result_a [W8_BUFFER_SIZE / 8];
  uint8_t result_z [W8_BUFFER_SIZE / 8];
  bp_state_t bp_state;

  uint64_t accum = 0;

  size_t last_trailing_ones = 0;
  size_t total_bytes_read   = 0;
  uint64_t quote_mask_carry = 0;
  size_t quote_odds_carry   = 0;
  size_t quote_evens_carry  = 1;

  uint8_t out_bp_buffer[W32_BUFFER_SIZE * 2];

  while (1) {
    size_t bytes_read = fread(buffer, 1, W8_BUFFER_SIZE, in);

    total_bytes_read += bytes_read;

    if (bytes_read < W8_BUFFER_SIZE) {
      if (ferror(in)) {
        fprintf(stderr, "Error reading file\n");
        exit(1);
      }

      if (bytes_read == 0) {
        if (feof(in)) {
          break;
        }
      }

      size_t next_alignment = ((bytes_read + 63) / 64) * 64;

      memset(buffer + bytes_read, 0, next_alignment - bytes_read);

      bytes_read = next_alignment;
    }

    accum += process_chunk(buffer, bytes_read,
      bits_of_d,
      bits_of_a,
      bits_of_z,
      bits_of_q,
      bits_of_b,
      bits_of_e,
      bits_of_w,
      &last_trailing_ones,
      &quote_odds_carry,
      &quote_evens_carry,
      &quote_mask_carry,
      result_ib,
      result_a,
      result_z);

    size_t ib_bytes = (bytes_read + 7) / 8;

    fwrite(result_ib, 1, ib_bytes, ib_out);

    size_t out_bp_bytes = write_bp_chunk(
      result_ib,
      result_a,
      result_z,
      ib_bytes,
      &bp_state,
      out_bp_buffer);

    fwrite(out_bp_buffer, out_bp_bytes, sizeof(uint64_t), bp_out);

    fflush(ib_out);
    fflush(bp_out);
  }

  write_bp_chunk_final(&bp_state, out_bp_buffer);

  fwrite(out_bp_buffer, 2, sizeof(uint64_t), bp_out);

  fclose(in);
  fclose(ib_out);

  return 0;
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
