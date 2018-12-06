#include "simd.h"

#include <stdio.h>
#include <string.h>
#include <immintrin.h>

int main(
    int argc,
    char **argv) {
  if (argc != 6) {
    fprintf(stderr, "./a.out <input-file> <output-ib-file> <output-bp-file> <output-op-file> <output-cl-file>\n");
    exit(1);
  }

  char *in_filename     = argv[1];
  char *ib_out_filename = argv[2];
  char *bp_out_filename = argv[3];
  char *op_out_filename = argv[4];
  char *cl_out_filename = argv[5];

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

  FILE *op_out = fopen(op_out_filename, "w");

  if (!op_out) {
    fprintf(stderr, "Failed to open op output file %s\n", op_out_filename);
    exit(1);
  }

  FILE *cl_out = fopen(cl_out_filename, "w");

  if (!cl_out) {
    fprintf(stderr, "Failed to open cl output file %s\n", cl_out_filename);
    exit(1);
  }

  uint8_t buffer[W8_BUFFER_SIZE];
  uint32_t phi_buffer[W8_BUFFER_SIZE];

  uint8_t ibs_buffer[W8_BUFFER_SIZE];
  uint8_t ops_buffer[W8_BUFFER_SIZE];
  uint8_t cls_buffer[W8_BUFFER_SIZE];

  // uint32_t result_ib[W8_BUFFER_SIZE];
  // uint32_t result_a [W8_BUFFER_SIZE];
  // uint32_t result_z [W8_BUFFER_SIZE];
  // uint64_t accum = 0;
  
  uint64_t remaining_bp_bits = 0;
  size_t remaining_bp_bits_len = 0;

  uint8_t out_bp_buffer[W32_BUFFER_SIZE * 2];

  size_t total_bytes_read = 0;
  uint32_t state = 0x03020100;

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

    uint32_t chunk_state = state;

    sm_process_chunk(buffer, bytes_read,
      &state,
      phi_buffer);

    make_ib_bp_chunks(chunk_state, phi_buffer, bytes_read,
      ibs_buffer,
      ops_buffer,
      cls_buffer);

    size_t idx_bytes = (bytes_read + 7) / 8;

    fwrite(ibs_buffer, 1, idx_bytes, ib_out);

    size_t out_bp_bytes = write_bp_chunk(
      ops_buffer,
      cls_buffer,
      idx_bytes,
      &remaining_bp_bits,
      &remaining_bp_bits_len,
      out_bp_buffer);

    fwrite(out_bp_buffer, out_bp_bytes, sizeof(uint64_t), bp_out);

    fflush(ib_out);
    fflush(bp_out);
  }

  // write_bp_chunk_final(&bp_state, out_bp_buffer);

  fprintf(stderr, "Final state %u\n", state);

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

  uint32_t ib_offset = 5 + state * 8;
  uint32_t op_offset = 6 + state * 8;
  uint32_t cl_offset = 7 + state * 8;

  // printf("ib_offset: %lu\n", ib_offset);
  // printf("op_offset: %lu\n", op_offset);
  // printf("cl_offset: %lu\n", cl_offset);
  
  for (size_t i = 0; i < phi_length; i += 8) {
    __m256i v_8 = *(__m256i *)&in_phis[i];
    __m256i v_ib_8 = _mm256_slli_epi64(v_8, ib_offset);
    __m256i v_op_8 = _mm256_slli_epi64(v_8, op_offset);
    __m256i v_cl_8 = _mm256_slli_epi64(v_8, cl_offset);
    uint8_t all_ibs = (uint8_t)_pext_u32(_mm256_movemask_epi8(v_ib_8), 0x11111111);
    uint8_t all_ops = (uint8_t)_pext_u32(_mm256_movemask_epi8(v_op_8), 0x11111111);
    uint8_t all_cls = (uint8_t)_pext_u32(_mm256_movemask_epi8(v_cl_8), 0x11111111);
    // printf("in_phis[i]: "); print_bits_32(in_phis[i]); printf(" "); print_bits_8(all_ibs); printf("\n");

    size_t j = i / 8;
    out_ibs[j] = all_ibs;
    out_ops[j] = all_ops;
    out_cls[j] = all_cls;

    // if (i > 20) {
    //   exit(1);
    // }
  }
}

// This code is not right for this kind of index
size_t
write_bits(
    uint64_t bits,
    size_t bits_len,
    uint64_t *remaining_bits,
    size_t *remaning_bits_len,
    uint64_t *out_buffer);

size_t write_bp_chunk(
    uint8_t *result_op,
    uint8_t *result_cl,
    size_t ib_bytes,
    uint64_t *remaining_bits,
    size_t *remaning_bits_len,
    uint8_t *out_buffer) {
  uint64_t *w64_result_op = (uint64_t *)result_op;
  uint64_t *w64_result_cl = (uint64_t *)result_cl;
  uint64_t *w64_work_bp   = (uint64_t *)out_buffer;

  uint64_t  w64_len           = ib_bytes / 8;
  size_t    w64s_ready        = 0;

  for (size_t i = 0; i < w64_len; ++i) {
    // printf("ib: "); print_bits_64(w64_result_ib[i]); printf("\n");
    // printf("op: "); print_bits_64(w64_result_op[i]); printf("\n");
    // printf("cl: "); print_bits_64(w64_result_cl[i]); printf("\n");

    // uint64_t w64_ib = w64_result_ib[i];
    uint64_t w64_op = w64_result_op[i];
    uint64_t w64_cl = w64_result_cl[i];

    uint64_t w64_op_lo = w64_op;
    uint64_t w64_op_hi = w64_op >> 32;

    uint64_t w64_cl_lo = w64_cl;
    uint64_t w64_cl_hi = w64_cl >> 32;

    uint64_t op_lo = _pdep_u64(w64_op_lo, 0x5555555555555555);
    uint64_t cl_lo = _pdep_u64(w64_cl_lo, 0xaaaaaaaaaaaaaaaa);
    uint64_t ib_lo = op_lo | cl_lo;

    uint64_t op_hi = _pdep_u64(w64_op_hi, 0x5555555555555555);
    uint64_t cl_hi = _pdep_u64(w64_cl_hi, 0xaaaaaaaaaaaaaaaa);
    uint64_t ib_hi = op_hi | cl_hi;

    size_t pc_ib_lo = __builtin_popcountll(ib_lo);
    size_t pc_ib_hi = __builtin_popcountll(ib_hi);

    uint64_t ext_lo = _pext_u64(op_lo, ib_lo);
    uint64_t ext_hi = _pext_u64(op_hi, ib_hi);

    w64s_ready += write_bits(ext_lo, pc_ib_lo, remaining_bits, remaning_bits_len, w64_work_bp + w64s_ready);
    w64s_ready += write_bits(ext_hi, pc_ib_hi, remaining_bits, remaning_bits_len, w64_work_bp + w64s_ready);
  }

  return w64s_ready;
}

size_t
write_bits(
    uint64_t bits,
    size_t bits_len,
    uint64_t *remaining_bits,
    size_t *remaining_bits_len,
    uint64_t *out_buffer) {
  *remaining_bits |= (bits << *remaining_bits_len);

  if (*remaining_bits_len + bits_len >= 64) {
    // Write full word
    *out_buffer = *remaining_bits;

    // Set up for next iteration
    *remaining_bits = bits >> (64 - *remaining_bits_len);

    *remaining_bits_len = *remaining_bits_len + bits_len - 64;

    return 1;
  } else {
    *remaining_bits_len += bits_len;

    return 0;
  }
}
