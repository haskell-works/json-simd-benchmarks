#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>

uint64_t rand64() {
  uint64_t a0 = (uint64_t)(rand() & 0xff);
  uint64_t a1 = (uint64_t)(rand() & 0xff);
  uint64_t a2 = (uint64_t)(rand() & 0xff);
  uint64_t a3 = (uint64_t)(rand() & 0xff);
  uint64_t a4 = (uint64_t)(rand() & 0xff);
  uint64_t a5 = (uint64_t)(rand() & 0xff);
  uint64_t a6 = (uint64_t)(rand() & 0xff);
  uint64_t a7 = (uint64_t)(rand() & 0xff);

  return
    (a0 << 56) |
    (a1 << 48) |
    (a2 << 40) |
    (a3 << 32) |
    (a4 << 24) |
    (a5 << 16) |
    (a6 <<  8) |
    (a7 <<  0);
}

uint64_t sub8(uint64_t a, uint64_t b) {
  if (a >= b) {
    return a - b;
  } else {
    return 0;
  }
}

uint64_t h8 = 0x8080808080808080;

uint64_t psubb(uint64_t a, uint64_t b) {
  return ((a | h8) - (b & ~h8)) ^ ((a ^ ~b) & h8);
}

uint64_t q(uint64_t a, uint64_t b, uint64_t d) {
  return ((d & ((~a) | b)) | (~a & b));
}

uint64_t psubusb(uint64_t a, uint64_t b) {
  uint64_t d = psubb(a, b);
  uint64_t s = psubb(0, (~q(a, b, d) & h8) >> 7);
  uint64_t r = d & s;
  return r;
}

int test_psubb(uint64_t a, uint64_t b) {
  uint64_t c = psubb(a, b);

  __m64 xa = (__m64)a;
  __m64 xb = (__m64)b;

  __m64 xe = _m_psubb(xa, xb);

  uint64_t e = (uint64_t)xe;

  if (c != e) {
    printf("Assertion failed!\n");
    printf("              psubusb              _m_psubusb\n");
    printf("   "       "0x%016llx      "       "0x%016llx\n", a, a);
    printf(" - "       "0x%016llx    - "       "0x%016llx\n", b, b);
    printf("---------------------  ----------------------\n");
    printf("   "       "0x%016llx      "       "0x%016llx\n", c, e);

    return 0;
  } else {
    return 1;
  }
}


int test_psubusb(uint64_t a, uint64_t b) {
  uint64_t c = psubusb(a, b);

  __m64 xa = (__m64)a;
  __m64 xb = (__m64)b;

  __m64 xe = _m_psubusb(xa, xb);

  uint64_t e = (uint64_t)xe;

  if (c != e) {
    printf("Assertion failed!\n");
    printf("              psubusb              _m_psubusb\n");
    printf("   "       "0x%016llx      "       "0x%016llx\n", a, a);
    printf(" - "       "0x%016llx    - "       "0x%016llx\n", b, b);
    printf("---------------------  ----------------------\n");
    printf("   "       "0x%016llx      "       "0x%016llx\n", c, e);
    printf("\n");

    uint64_t d = psubb(a, b);
    uint64_t s = psubb(0, (~q(a, b, d) & h8) >> 7);
    uint64_t r = d & s;

    printf("a: 0x%016llx\n", a);
    printf("b: 0x%016llx\n", b);
    printf("d: 0x%016llx\n", d);
    printf("e: 0x%016llx\n", q(a, b, d) & h8);
    printf("s: 0x%016llx\n", s);
    printf("r: 0x%016llx\n", r);

    return 0;
  } else {
    return 1;
  }
}


int main(
    int argc,
    char **argv) {
  time_t t;

  srand((unsigned) time(&t));

  for (int i = 0; i < 1000; ++i) {
    uint64_t a = rand64();
    uint64_t b = rand64();

    if (!test_psubb(a, b)) {
      printf("Failure\n");
      break;
    }
  }

  for (int i = 0; i < 1000; ++i) {
    uint64_t a = rand64();
    uint64_t b = rand64();

    if (!test_psubusb(a, b)) {
      printf("Failure\n");
      break;
    }
  }

  // printf("     "                                "a0  "      "a0  "      "a1  "      "a1\n");
  // printf("     "                                "b0  "      "b1  "      "b0  "      "b1\n");
  // printf("d0 ""   %d""   %d""   %d""   %d""\n", q(0, 0, 0) & 1, q(0, 1, 0) & 1, q(1, 0, 0) & 1, q(1, 1, 0) & 1);
  // printf("d1 ""   %d""   %d""   %d""   %d""\n", q(0, 0, 1) & 1, q(0, 1, 1) & 1, q(1, 0, 1) & 1, q(1, 1, 1) & 1);

  return 0;
}
