#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "simd.h"

extern uint32_t phi_table[][256];
extern uint32_t transition_table[][256];

void sm_process_chunk(
    uint8_t *in_buffer,
    size_t in_length,
    uint8_t *inout_state,
    uint8_t *out_phi_buffer) {  
  uint8_t s = *inout_state;

  for (size_t i = 0; i < in_length; ++i) {
    out_phi_buffer[i] = phi_table       [s][in_buffer[i]];
    s                 = transition_table[s][in_buffer[i]];
  }

  *inout_state = s;
}
