#include "simd.h"

uint8_t escape_mask[2][256] =
  { { 0xff, 0xfd, 0xfb, 0xff, 0xf7, 0xf5, 0xff, 0xf7, 0xef, 0xed, 0xeb, 0xef, 0xff, 0xfd, 0xef, 0xff
    , 0xdf, 0xdd, 0xdb, 0xdf, 0xd7, 0xd5, 0xdf, 0xd7, 0xff, 0xfd, 0xfb, 0xff, 0xdf, 0xdd, 0xff, 0xdf
    , 0xbf, 0xbd, 0xbb, 0xbf, 0xb7, 0xb5, 0xbf, 0xb7, 0xaf, 0xad, 0xab, 0xaf, 0xbf, 0xbd, 0xaf, 0xbf
    , 0xff, 0xfd, 0xfb, 0xff, 0xf7, 0xf5, 0xff, 0xf7, 0xbf, 0xbd, 0xbb, 0xbf, 0xff, 0xfd, 0xbf, 0xff
    , 0x7f, 0x7d, 0x7b, 0x7f, 0x77, 0x75, 0x7f, 0x77, 0x6f, 0x6d, 0x6b, 0x6f, 0x7f, 0x7d, 0x6f, 0x7f
    , 0x5f, 0x5d, 0x5b, 0x5f, 0x57, 0x55, 0x5f, 0x57, 0x7f, 0x7d, 0x7b, 0x7f, 0x5f, 0x5d, 0x7f, 0x5f
    , 0xff, 0xfd, 0xfb, 0xff, 0xf7, 0xf5, 0xff, 0xf7, 0xef, 0xed, 0xeb, 0xef, 0xff, 0xfd, 0xef, 0xff
    , 0x7f, 0x7d, 0x7b, 0x7f, 0x77, 0x75, 0x7f, 0x77, 0xff, 0xfd, 0xfb, 0xff, 0x7f, 0x7d, 0xff, 0x7f
    , 0xff, 0xfd, 0xfb, 0xff, 0xf7, 0xf5, 0xff, 0xf7, 0xef, 0xed, 0xeb, 0xef, 0xff, 0xfd, 0xef, 0xff
    , 0xdf, 0xdd, 0xdb, 0xdf, 0xd7, 0xd5, 0xdf, 0xd7, 0xff, 0xfd, 0xfb, 0xff, 0xdf, 0xdd, 0xff, 0xdf
    , 0xbf, 0xbd, 0xbb, 0xbf, 0xb7, 0xb5, 0xbf, 0xb7, 0xaf, 0xad, 0xab, 0xaf, 0xbf, 0xbd, 0xaf, 0xbf
    , 0xff, 0xfd, 0xfb, 0xff, 0xf7, 0xf5, 0xff, 0xf7, 0xbf, 0xbd, 0xbb, 0xbf, 0xff, 0xfd, 0xbf, 0xff
    , 0xff, 0xfd, 0xfb, 0xff, 0xf7, 0xf5, 0xff, 0xf7, 0xef, 0xed, 0xeb, 0xef, 0xff, 0xfd, 0xef, 0xff
    , 0xdf, 0xdd, 0xdb, 0xdf, 0xd7, 0xd5, 0xdf, 0xd7, 0xff, 0xfd, 0xfb, 0xff, 0xdf, 0xdd, 0xff, 0xdf
    , 0xff, 0xfd, 0xfb, 0xff, 0xf7, 0xf5, 0xff, 0xf7, 0xef, 0xed, 0xeb, 0xef, 0xff, 0xfd, 0xef, 0xff
    , 0xff, 0xfd, 0xfb, 0xff, 0xf7, 0xf5, 0xff, 0xf7, 0xff, 0xfd, 0xfb, 0xff, 0xff, 0xfd, 0xff, 0xff
    }
  , { 0xfe, 0xff, 0xfa, 0xfb, 0xf6, 0xf7, 0xfe, 0xff, 0xee, 0xef, 0xea, 0xeb, 0xfe, 0xff, 0xee, 0xef
    , 0xde, 0xdf, 0xda, 0xdb, 0xd6, 0xd7, 0xde, 0xdf, 0xfe, 0xff, 0xfa, 0xfb, 0xde, 0xdf, 0xfe, 0xff
    , 0xbe, 0xbf, 0xba, 0xbb, 0xb6, 0xb7, 0xbe, 0xbf, 0xae, 0xaf, 0xaa, 0xab, 0xbe, 0xbf, 0xae, 0xaf
    , 0xfe, 0xff, 0xfa, 0xfb, 0xf6, 0xf7, 0xfe, 0xff, 0xbe, 0xbf, 0xba, 0xbb, 0xfe, 0xff, 0xbe, 0xbf
    , 0x7e, 0x7f, 0x7a, 0x7b, 0x76, 0x77, 0x7e, 0x7f, 0x6e, 0x6f, 0x6a, 0x6b, 0x7e, 0x7f, 0x6e, 0x6f
    , 0x5e, 0x5f, 0x5a, 0x5b, 0x56, 0x57, 0x5e, 0x5f, 0x7e, 0x7f, 0x7a, 0x7b, 0x5e, 0x5f, 0x7e, 0x7f
    , 0xfe, 0xff, 0xfa, 0xfb, 0xf6, 0xf7, 0xfe, 0xff, 0xee, 0xef, 0xea, 0xeb, 0xfe, 0xff, 0xee, 0xef
    , 0x7e, 0x7f, 0x7a, 0x7b, 0x76, 0x77, 0x7e, 0x7f, 0xfe, 0xff, 0xfa, 0xfb, 0x7e, 0x7f, 0xfe, 0xff
    , 0xfe, 0xff, 0xfa, 0xfb, 0xf6, 0xf7, 0xfe, 0xff, 0xee, 0xef, 0xea, 0xeb, 0xfe, 0xff, 0xee, 0xef
    , 0xde, 0xdf, 0xda, 0xdb, 0xd6, 0xd7, 0xde, 0xdf, 0xfe, 0xff, 0xfa, 0xfb, 0xde, 0xdf, 0xfe, 0xff
    , 0xbe, 0xbf, 0xba, 0xbb, 0xb6, 0xb7, 0xbe, 0xbf, 0xae, 0xaf, 0xaa, 0xab, 0xbe, 0xbf, 0xae, 0xaf
    , 0xfe, 0xff, 0xfa, 0xfb, 0xf6, 0xf7, 0xfe, 0xff, 0xbe, 0xbf, 0xba, 0xbb, 0xfe, 0xff, 0xbe, 0xbf
    , 0xfe, 0xff, 0xfa, 0xfb, 0xf6, 0xf7, 0xfe, 0xff, 0xee, 0xef, 0xea, 0xeb, 0xfe, 0xff, 0xee, 0xef
    , 0xde, 0xdf, 0xda, 0xdb, 0xd6, 0xd7, 0xde, 0xdf, 0xfe, 0xff, 0xfa, 0xfb, 0xde, 0xdf, 0xfe, 0xff
    , 0xfe, 0xff, 0xfa, 0xfb, 0xf6, 0xf7, 0xfe, 0xff, 0xee, 0xef, 0xea, 0xeb, 0xfe, 0xff, 0xee, 0xef
    , 0xfe, 0xff, 0xfa, 0xfb, 0xf6, 0xf7, 0xfe, 0xff, 0xfe, 0xff, 0xfa, 0xfb, 0xfe, 0xff, 0xfe, 0xff
    }
  };
