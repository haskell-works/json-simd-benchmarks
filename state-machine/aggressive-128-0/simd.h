#include <immintrin.h>
#include <mmintrin.h>
#include <stdint.h>
#include <stdio.h>

#define W8_BUFFER_SIZE    (1024 * 32)
#define W32_BUFFER_SIZE   (W8_BUFFER_SIZE / 4)
#define W64_BUFFER_SIZE   (W8_BUFFER_SIZE / 8)

size_t
sm_write_bp_chunk(
    uint8_t *result_op,
    uint8_t *result_cl,
    size_t ib_bytes,
    uint64_t *remaining_bp_bits,
    size_t *remaning_bp_bits_len,
    uint8_t *out_buffer);

size_t
sm_write_bp_chunk_final(
    uint64_t remaining_bits,
    size_t remaining_bits_len,
    uint64_t *out_buffer);

void
sm_process_chunk(
    uint8_t *in_buffer,
    size_t in_length,
    uint32_t *inout_state,
    uint32_t *out_phi_buffer);

void
sm_make_ib_bp_chunks(
    uint8_t state,
    uint32_t *in_phis,
    size_t phi_length,
    uint8_t *out_ibs,
    uint8_t *out_ops,
    uint8_t *out_cls);

void
sm_make_ib_op_cl_chunks(
    uint8_t state,
    uint32_t *in_phis,
    size_t phi_length,
    uint8_t *out_ibs,
    uint8_t *out_ops,
    uint8_t *out_cls);
