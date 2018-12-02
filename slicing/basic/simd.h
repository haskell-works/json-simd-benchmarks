#include <immintrin.h>
#include <mmintrin.h>
#include <stdint.h>
#include <stdio.h>

#define W8_BUFFER_SIZE    (1024 * 32)
#define W32_BUFFER_SIZE   (W8_BUFFER_SIZE / 4)
#define W64_BUFFER_SIZE   (W8_BUFFER_SIZE / 8)

extern __m256i transition_phi_table_wide[256];
extern __m128i transition_phi_table[256];

typedef struct vec32_8i {
  uint8_t w00;
  uint8_t w01;
  uint8_t w02;
  uint8_t w03;
  uint8_t w04;
  uint8_t w05;
  uint8_t w06;
  uint8_t w07;
  uint8_t w08;
  uint8_t w09;
  uint8_t w10;
  uint8_t w11;
  uint8_t w12;
  uint8_t w13;
  uint8_t w14;
  uint8_t w15;
  uint8_t w16;
  uint8_t w17;
  uint8_t w18;
  uint8_t w19;
  uint8_t w20;
  uint8_t w21;
  uint8_t w22;
  uint8_t w23;
  uint8_t w24;
  uint8_t w25;
  uint8_t w26;
  uint8_t w27;
  uint8_t w28;
  uint8_t w29;
  uint8_t w30;
  uint8_t w31;
} vec32_8i_t;

typedef struct vec16_16i {
  uint16_t w00;
  uint16_t w01;
  uint16_t w02;
  uint16_t w03;
  uint16_t w04;
  uint16_t w05;
  uint16_t w06;
  uint16_t w07;
  uint16_t w08;
  uint16_t w09;
  uint16_t w10;
  uint16_t w11;
  uint16_t w12;
  uint16_t w13;
  uint16_t w14;
  uint16_t w15;
} vec16_16i_t;

typedef struct vec8_32i {
  uint32_t w0;
  uint32_t w1;
  uint32_t w2;
  uint32_t w3;
  uint32_t w4;
  uint32_t w5;
  uint32_t w6;
  uint32_t w7;
} vec8_32i_t;

typedef struct vec4_64i {
  uint64_t w0;
  uint64_t w1;
  uint64_t w2;
  uint64_t w3;
} vec4_64i_t;

typedef union vm256i {
  vec8_32i_t w8s;
  vec8_32i_t w16s;
  vec8_32i_t w32s;
  vec4_64i_t w64s;
  __m256i m;
} vm256i_t;

typedef struct vec16_8i {
  uint8_t w00;
  uint8_t w01;
  uint8_t w02;
  uint8_t w03;
  uint8_t w04;
  uint8_t w05;
  uint8_t w06;
  uint8_t w07;
  uint8_t w08;
  uint8_t w09;
  uint8_t w10;
  uint8_t w11;
  uint8_t w12;
  uint8_t w13;
  uint8_t w14;
  uint8_t w15;
} vec16_8i_t;

typedef struct vec8_16i {
  uint16_t w00;
  uint16_t w01;
  uint16_t w02;
  uint16_t w03;
  uint16_t w04;
  uint16_t w05;
  uint16_t w06;
  uint16_t w07;
} vec8_16i_t;

typedef struct vec4_32i {
  uint32_t w0;
  uint32_t w1;
  uint32_t w2;
  uint32_t w3;
} vec4_32i_t;

typedef struct vec2_64i {
  uint64_t w0;
  uint64_t w1;
} vec2_64i_t;

typedef union vm128i {
  vec8_32i_t w8s;
  vec8_32i_t w16s;
  vec8_32i_t w32s;
  vec4_64i_t w64s;
  __m128i m;
} vm128i_t;

typedef struct bp_state {
  uint64_t  remainder_bits_d;
  uint64_t  remainder_bits_a;
  uint64_t  remainder_bits_z;
  size_t    remainder_len;
} bp_state_t;

extern uint32_t transition_table_simd[];

extern uint32_t phi_table_simd[];

void print256_num(__m256i var);

void fprint256_num(FILE *file, __m256i var);

void print128_num(__m128i var);

void fprint128_num(FILE *file, __m128i var);

void print_bits_64(uint64_t v);

typedef struct bp_state bp_state_t;

int main_spliced(
    int argc,
    char **argv);

uint64_t process_chunk(
    uint8_t *in_buffer,
    size_t in_length,
    uint8_t *work_bits_of_d,       // Working buffer of minimum length ((in_length + 63) / 64)
    uint8_t *work_bits_of_a,       // Working buffer of minimum length ((in_length + 63) / 64)
    uint8_t *work_bits_of_z,       // Working buffer of minimum length ((in_length + 63) / 64)
    uint8_t *work_bits_of_q,       // Working buffer of minimum length ((in_length + 63) / 64)
    uint8_t *work_bits_of_b,       // Working buffer of minimum length ((in_length + 63) / 64)
    uint8_t *work_bits_of_e,       // Working buffer of minimum length ((in_length + 63) / 64)
    uint8_t *work_bits_of_s,       // Working buffer of minimum length ((in_length + 63) / 64)
    size_t *last_trailing_ones,
    size_t *quote_odds_carry,
    size_t *quote_evens_carry,
    uint64_t *quote_mask_carry,
    uint8_t *result_ibs,
    uint8_t *result_a,
    uint8_t *result_z);

void init_bp_state(
    bp_state_t *bp_state);

size_t write_bp_chunk(
    uint8_t *result_ib,
    uint8_t *result_a,
    uint8_t *result_z,
    size_t ib_bytes,
    bp_state_t *bp_state,
    uint8_t *out_buffer);

size_t write_bp_chunk_final(
    bp_state_t *bp_state,
    uint8_t *out_buffer);

// ---

int sm_main(
    int argc,
    char **argv);

void sm_process_chunk(
    uint8_t *in_buffer,
    size_t in_length,
    uint32_t *inout_state,
    uint32_t *out_phi_buffer);

void make_ib_bp_chunks(
    uint8_t state,
    uint32_t *in_phis,
    size_t phi_length,
    uint8_t *out_ibs,
    uint8_t *out_ops,
    uint8_t *out_cls);
