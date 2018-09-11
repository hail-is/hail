#define __STDC_FORMAT_MACROS // see: http://stackoverflow.com/questions/8132399/how-to-printf-uint64-t and https://sourceware.org/bugzilla/show_bug.cgi?id=15366
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include "ibs.h"

using namespace simdpp;

#define expect(name, x) if (!(x)) { ++failures; printf(name ": expected " #x " to be true, but was false\n\n"); } else { ++successes; }
#define expect_equal(name, fmt, x, y) if ((x) != (y)) { ++failures; printf(name ": expected " #x " to equal " #y ", but actually got " fmt " and " fmt"\n\n", x, y); } else { ++successes; }

uint64_t resultIndex(uint64_t* result, int nSamples, int si, int sj, int ibs) {
  return result[si*nSamples*3 + sj*3 + ibs];
}

int main(int argc, char** argv) {
  if (argc != 1) {
    printf("Expected zero arguments.\n");
    return -1;
  }

  uint64_t failures = 0;
  uint64_t successes = 0;

  // ibs256 tests
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector allNA1 = make_uint(0xAAAAAAAAAAAAAAAA);
    uint64vector allNA2 = make_uint(0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, allNA1, allNA2);

    expect_equal("allNA", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("allNA", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("allNA", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector allHomRef1 = make_uint(0x0000000000000000);
    uint64vector allHomRef2 = make_uint(0x0000000000000000);
    ibs256_with_na(result, allHomRef1, allHomRef2);

    expect_equal("allHomRef", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("allHomRef", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("allHomRef", "%" PRIu64, result[2], ((uint64_t)(sizeof(uint64vector)*8/2)));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector allHet1 = make_uint(0x5555555555555555);
    uint64vector allHet2 = make_uint(0x5555555555555555);
    ibs256_with_na(result, allHet1, allHet2);

    expect_equal("allHet", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("allHet", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("allHet", "%" PRIu64, result[2], ((uint64_t)(sizeof(uint64vector)*8/2)));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector allHomAlt1 = make_uint(0xFFFFFFFFFFFFFFFF);
    uint64vector allHomAlt2 = make_uint(0xFFFFFFFFFFFFFFFF);
    ibs256_with_na(result, allHomAlt1, allHomAlt2);

    expect_equal("allHomAlt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("allHomAlt", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("allHomAlt", "%" PRIu64, result[2], ((uint64_t)(sizeof(uint64vector)*8/2)));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector allHomAlt = make_uint(0xFFFFFFFFFFFFFFFF);
    uint64vector allHomRef = make_uint(0x0000000000000000);
    ibs256_with_na(result, allHomAlt, allHomRef);

    expect_equal("homAlt v homRef", "%" PRIu64, result[0], ((uint64_t)(sizeof(uint64vector)*8/2)));
    expect_equal("homAlt v homRef", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("homAlt v homRef", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector allHet = make_uint(0x5555555555555555);
    uint64vector allHomRef = make_uint(0x0000000000000000);
    ibs256_with_na(result, allHet, allHomRef);

    expect_equal("het v homRef", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("het v homRef", "%" PRIu64, result[1], ((uint64_t)(sizeof(uint64vector)*8/2)));
    expect_equal("het v homRef", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector allHet = make_uint(0x5555555555555555);
    uint64vector allHomAlt = make_uint(0xFFFFFFFFFFFFFFFF);
    ibs256_with_na(result, allHet, allHomAlt);

    expect_equal("het v homAlt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("het v homAlt", "%" PRIu64, result[1], ((uint64_t)(sizeof(uint64vector)*8/2)));
    expect_equal("het v homAlt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector allHet = make_uint(0x5555555555555555);
    uint64vector allHomAltOneNA = make_uint(0xBFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    ibs256_with_na(result, allHet, allHomAltOneNA);

    expect_equal("het v homAltOneNA", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("het v homAltOneNA", "%" PRIu64, result[1], ((uint64_t)(sizeof(uint64vector)*8/2 - 1)));
    expect_equal("het v homAltOneNA", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector allHet = make_uint(0x5555555555555555);
    uint64vector allHomAltTwoNA = make_uint(0xAFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    ibs256_with_na(result, allHet, allHomAltTwoNA);

    expect_equal("het v homAltTwoNA", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("het v homAltTwoNA", "%" PRIu64, result[1], ((uint64_t)(sizeof(uint64vector)*8/2 - 2)));
    expect_equal("het v homAltTwoNA", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector ref_het_alt_het = make_uint(0x1DAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    uint64vector het_ref_het_alt = make_uint(0x47AAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0x1DAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    uint64vector het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0x47AAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    uint64vector het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    uint64vector het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  #if UINT64_VECTOR_SIZE > 2
  // these tests have meaningful data higher than 128 bits
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x1DAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    uint64vector het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x47AAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x1DAAAAAAAAAAAAAA);
    uint64vector het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x47AAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA1D);
    uint64vector het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA47);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64vector ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA);
    uint64vector het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  #endif

  // ibsVec tests
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64_t ref_het_alt_het[8] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    uint64_t het_ref_het_alt[8] =
      { 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    ibsVec2(result, 8, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ibsVec2 ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ibsVec2 ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)8));
    expect_equal("ibsVec2 ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64_t ref_het_alt_het[8] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA };
    uint64_t het_ref_het_alt[8] =
      { 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47,
        0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA };
    ibsVec2(result, 8, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ibsVec2 7 ref_het_alt_het v 7 het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ibsVec2 7 ref_het_alt_het v 7 het_ref_het_alt", "%" PRIu64, result[1], 7 * ((uint64_t)4));
    expect_equal("ibsVec2 7 ref_het_alt_het v 7 het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64_t ref_het_alt_het[8] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA };
    uint64_t ref_het_alt_het2[8] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA };
    ibsVec2(result, 8, ref_het_alt_het, ref_het_alt_het2);

    expect_equal("ibsVec2 ref_het_alt_het v self", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ibsVec2 ref_het_alt_het v self", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("ibsVec2 ref_het_alt_het v self", "%" PRIu64, result[2], 4*((uint64_t)7));
  }
  // ibsMat (note that the number of samples must be both greater than and a
  // multiple of CACHE_SIZE_IN_MATRIX_ROWS)
  //
  // the number of genotypes in each row is 256 (8 32-genotype packs), the
  // Makefile overrides NUMBER_OF_GENOTYPES_PER_ROW when building the tests
  {
    uint64_t result[16*16*3] = { 0 };
    uint64_t all_ref_het_alt_het[16*8] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA,

        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
      };
    uint64_t all_ref_het_alt_het2[16*8] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA,

        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
      };
    ibsMat(result, 16, 8, all_ref_het_alt_het, all_ref_het_alt_het2);

    expect_equal("ibsMat identical 0 0, ibs0", "%" PRIu64, resultIndex(result, 16, 0, 0, 0), ((uint64_t)0));
    expect_equal("ibsMat identical 0 0, ibs1", "%" PRIu64, resultIndex(result, 16, 0, 0, 1), ((uint64_t)0));
    expect_equal("ibsMat identical 0 0, ibs2", "%" PRIu64, resultIndex(result, 16, 0, 0, 2), 4*((uint64_t)7));
    expect_equal("ibsMat identical 0 1, ibs0", "%" PRIu64, resultIndex(result, 16, 0, 1, 0), ((uint64_t)0));
    expect_equal("ibsMat identical 0 1, ibs1", "%" PRIu64, resultIndex(result, 16, 0, 1, 1), ((uint64_t)0));
    expect_equal("ibsMat identical 0 1, ibs2", "%" PRIu64, resultIndex(result, 16, 0, 1, 2), 4*((uint64_t)7));
    expect_equal("ibsMat identical 1 1, ibs0", "%" PRIu64, resultIndex(result, 16, 1, 1, 0), ((uint64_t)0));
    expect_equal("ibsMat identical 1 1, ibs1", "%" PRIu64, resultIndex(result, 16, 1, 1, 1), ((uint64_t)0));
    expect_equal("ibsMat identical 1 1, ibs2", "%" PRIu64, resultIndex(result, 16, 1, 1, 2), 4*((uint64_t)7));
  }
  {
    uint64_t result[16*16*3] = { 0 };
    uint64_t one_ibs1_rest_ibs2[16*8] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAA1F, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA,

        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
      };
    uint64_t one_ibs1_rest_ibs2_2[16*8] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAA1F, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA,

        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
      };
    ibsMat(result, 16, 8, one_ibs1_rest_ibs2, one_ibs1_rest_ibs2_2);

    expect_equal("ibsMat one-ibs1 0 0, ibs0", "%" PRIu64, resultIndex(result, 16, 0, 0, 0), ((uint64_t)0));
    expect_equal("ibsMat one-ibs1 0 0, ibs1", "%" PRIu64, resultIndex(result, 16, 0, 0, 1), ((uint64_t)0));
    expect_equal("ibsMat one-ibs1 0 0, ibs2", "%" PRIu64, resultIndex(result, 16, 0, 0, 2), 4*((uint64_t)7));
    expect_equal("ibsMat one-ibs1 0 1, ibs0", "%" PRIu64, resultIndex(result, 16, 0, 1, 0), ((uint64_t)0));
    expect_equal("ibsMat one-ibs1 0 1, ibs1", "%" PRIu64, resultIndex(result, 16, 0, 1, 1), ((uint64_t)1));
    expect_equal("ibsMat one-ibs1 0 1, ibs2", "%" PRIu64, resultIndex(result, 16, 0, 1, 2), 3+4*((uint64_t)6));
    expect_equal("ibsMat one-ibs1 1 1, ibs0", "%" PRIu64, resultIndex(result, 16, 1, 1, 0), ((uint64_t)0));
    expect_equal("ibsMat one-ibs1 1 1, ibs1", "%" PRIu64, resultIndex(result, 16, 1, 1, 1), ((uint64_t)0));
    expect_equal("ibsMat one-ibs1 1 1, ibs2", "%" PRIu64, resultIndex(result, 16, 1, 1, 2), 4*((uint64_t)7));
  }

  if (failures != 0) {
    printf("%" PRIu64 " test(s) failed.\n", failures);
    return -1;
  } else {
    printf("%" PRIu64 " test(s) succeeded.\n", successes);
    return 0;
  }
}
