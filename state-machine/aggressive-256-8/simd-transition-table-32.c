#include <stdint.h>
#include <immintrin.h>

uint32_t simd_transition_table_32[] =
{ 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010001, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x03010103, 0x00010100, 0x03010103, 0x03010103, 0x00010100
  , 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103
  , 0x03010103, 0x03010103, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103
  , 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103
  , 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103
  , 0x03010103, 0x03010103, 0x03010103, 0x00010100, 0x00010200, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103
  , 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103
  , 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103, 0x03010103
  , 0x03010103, 0x03010103, 0x03010103, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  , 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100, 0x00010100
  };
