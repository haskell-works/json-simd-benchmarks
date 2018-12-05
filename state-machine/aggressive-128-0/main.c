#include "simd.h"

#include <stdio.h>
#include <string.h>
#include <immintrin.h>

typedef struct bp_state {
  uint64_t  remainder_bits_d;
  uint64_t  remainder_bits_a;
  uint64_t  remainder_bits_z;
  size_t    remainder_len;
} bp_state_t;

size_t write_bp_chunk(
    uint8_t *result_ib,
    uint8_t *result_a,
    uint8_t *result_z,
    size_t ib_bytes,
    bp_state_t *bp_state,
    uint8_t *out_buffer);

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

  uint32_t result_ib[W8_BUFFER_SIZE];
  uint32_t result_a [W8_BUFFER_SIZE];
  uint32_t result_z [W8_BUFFER_SIZE];
  uint64_t accum = 0;
  bp_state_t bp_state = {0, 0, 0, 0};

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
      ibs_buffer,
      ops_buffer,
      cls_buffer,
      idx_bytes,
      &bp_state,
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

size_t write_bp_chunk(
    uint8_t *result_ib,
    uint8_t *result_a,
    uint8_t *result_z,
    size_t ib_bytes,
    bp_state_t *bp_state,
    uint8_t *out_buffer) {
  uint64_t *w64_result_ib = (uint64_t *)result_ib;
  uint64_t *w64_result_a  = (uint64_t *)result_a;
  uint64_t *w64_result_z  = (uint64_t *)result_z;
  uint64_t *w64_work_bp   = (uint64_t *)out_buffer;

  uint64_t  w64_len           = ib_bytes / 8;
  uint64_t  remainder_bits_d  = (*bp_state).remainder_bits_d;
  uint64_t  remainder_bits_a  = (*bp_state).remainder_bits_a;
  uint64_t  remainder_bits_z  = (*bp_state).remainder_bits_z;
  size_t    remainder_len     = (*bp_state).remainder_len;
  size_t    w64s_ready        = 0;

  for (size_t i = 0; i < w64_len; ++i) {
    printf("ib: "); print_bits_64(w64_result_ib[i]); printf("\n");
    printf("op: "); print_bits_64(w64_result_a [i]); printf("\n");
    printf("cl: "); print_bits_64(w64_result_z [i]); printf("\n");

    uint64_t w64_ib = w64_result_ib[i];
    uint64_t w64_a  = w64_result_a[i];
    uint64_t w64_z  = w64_result_z[i];

    size_t pc_ib = __builtin_popcountll(w64_ib);

    uint64_t ext_d = _pext_u64(~(w64_a | w64_z) , w64_ib);
    uint64_t ext_a = _pext_u64(w64_a            , w64_ib);
    uint64_t ext_z = _pext_u64(w64_z            , w64_ib);

    remainder_bits_d |= (ext_d << remainder_len);
    remainder_bits_a |= (ext_a << remainder_len);
    remainder_bits_z |= (ext_z << remainder_len);

    if (remainder_len + pc_ib >= 64) {
      // Write full word
      w64_work_bp[w64s_ready] =
        _pdep_u64(remainder_bits_a, 0x5555555555555555) |
        _pdep_u64(remainder_bits_a, 0xaaaaaaaaaaaaaaaa) |
        _pdep_u64(remainder_bits_d, 0xaaaaaaaaaaaaaaaa);

      printf("remainder_bits_d: "); print_bits_64(remainder_bits_d); printf("\n");
      printf("remainder_bits_a: "); print_bits_64(remainder_bits_a); printf("\n");
      printf("remainder_bits_z: "); print_bits_64(remainder_bits_z); printf("\n");

      printf("%d: ", i); print_bits_64(w64_work_bp[w64s_ready]); printf(" <===\n");
 
      w64s_ready += 1;

      remainder_bits_a = remainder_bits_a >> 32;
      remainder_bits_z = remainder_bits_z >> 32;
      remainder_bits_d = remainder_bits_d >> 32;

      w64_work_bp[w64s_ready] =
        _pdep_u64(remainder_bits_a, 0x5555555555555555) |
        _pdep_u64(remainder_bits_a, 0xaaaaaaaaaaaaaaaa) |
        _pdep_u64(remainder_bits_d, 0xaaaaaaaaaaaaaaaa);

      printf("remainder_bits_d: "); print_bits_64(remainder_bits_d); printf("\n");
      printf("remainder_bits_a: "); print_bits_64(remainder_bits_a); printf("\n");
      printf("remainder_bits_z: "); print_bits_64(remainder_bits_z); printf("\n");

      printf("%d: ", i); print_bits_64(w64_work_bp[w64s_ready]); printf(" <===\n");
 
      w64s_ready += 1;

      // Set up for next iteration
      remainder_bits_d = ext_d >> (64 - remainder_len);
      remainder_bits_a = ext_a >> (64 - remainder_len);
      remainder_bits_z = ext_z >> (64 - remainder_len);

      remainder_len = remainder_len + pc_ib - 64;

      exit(1);
    } else {
      remainder_len += pc_ib;
    }
  }


  (*bp_state).remainder_bits_d  = remainder_bits_d;
  (*bp_state).remainder_bits_a  = remainder_bits_a;
  (*bp_state).remainder_bits_z  = remainder_bits_z;
  (*bp_state).remainder_len     = remainder_len;

  return w64s_ready;
}
