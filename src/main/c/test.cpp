#include <stdint.h>
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
    uint64v allNA1 = make_uint(0xAAAAAAAAAAAAAAAA);
    uint64v allNA2 = make_uint(0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, allNA1, allNA2);

    expect_equal("allNA", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("allNA", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("allNA", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHomRef1 = make_uint(0x0000000000000000);
    uint64v allHomRef2 = make_uint(0x0000000000000000);
    ibs256_with_na(result, allHomRef1, allHomRef2);

    expect_equal("allHomRef", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("allHomRef", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("allHomRef", "%" PRIu64, result[2], ((uint64_t)128));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHet1 = make_uint(0x5555555555555555);
    uint64v allHet2 = make_uint(0x5555555555555555);
    ibs256_with_na(result, allHet1, allHet2);

    expect_equal("allHet", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("allHet", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("allHet", "%" PRIu64, result[2], ((uint64_t)128));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHomAlt1 = make_uint(0xFFFFFFFFFFFFFFFF);
    uint64v allHomAlt2 = make_uint(0xFFFFFFFFFFFFFFFF);
    ibs256_with_na(result, allHomAlt1, allHomAlt2);

    expect_equal("allHomAlt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("allHomAlt", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("allHomAlt", "%" PRIu64, result[2], ((uint64_t)128));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHomAlt = make_uint(0xFFFFFFFFFFFFFFFF);
    uint64v allHomRef = make_uint(0x0000000000000000);
    ibs256_with_na(result, allHomAlt, allHomRef);

    expect_equal("homAlt v homRef", "%" PRIu64, result[0], ((uint64_t)128));
    expect_equal("homAlt v homRef", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("homAlt v homRef", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHet = make_uint(0x5555555555555555);
    uint64v allHomRef = make_uint(0x0000000000000000);
    ibs256_with_na(result, allHet, allHomRef);

    expect_equal("het v homRef", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("het v homRef", "%" PRIu64, result[1], ((uint64_t)128));
    expect_equal("het v homRef", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHet = make_uint(0x5555555555555555);
    uint64v allHomAlt = make_uint(0xFFFFFFFFFFFFFFFF);
    ibs256_with_na(result, allHet, allHomAlt);

    expect_equal("het v homAlt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("het v homAlt", "%" PRIu64, result[1], ((uint64_t)128));
    expect_equal("het v homAlt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHet = make_uint(0x5555555555555555);
    uint64v allHomAltOneNA = make_uint(0xBFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    ibs256_with_na(result, allHet, allHomAltOneNA);

    expect_equal("het v homAltOneNA", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("het v homAltOneNA", "%" PRIu64, result[1], ((uint64_t)127));
    expect_equal("het v homAltOneNA", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v allHet = make_uint(0x5555555555555555);
    uint64v allHomAltTwoNA = make_uint(0xAFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    ibs256_with_na(result, allHet, allHomAltTwoNA);

    expect_equal("het v homAltTwoNA", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("het v homAltTwoNA", "%" PRIu64, result[1], ((uint64_t)126));
    expect_equal("het v homAltTwoNA", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0x1DAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    uint64v het_ref_het_alt = make_uint(0x47AAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0x1DAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    uint64v het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0x47AAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x1DAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    uint64v het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x47AAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x1DAAAAAAAAAAAAAA);
    uint64v het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x47AAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA1D);
    uint64v het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA47);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA);
    uint64v het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    uint64v het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64v ref_het_alt_het = make_uint(0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    uint64v het_ref_het_alt = make_uint(0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
    ibs256_with_na(result, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)4));
    expect_equal("ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }

  // ibsVec tests
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64_t ref_het_alt_het[7] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    uint64_t het_ref_het_alt[7] =
      { 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
        0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA };
    ibsVec2(result, 7, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ibsVec2 ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ibsVec2 ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[1], ((uint64_t)8));
    expect_equal("ibsVec2 ref_het_alt_het v het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64_t ref_het_alt_het[7] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D };
    uint64_t het_ref_het_alt[7] =
      { 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47,
        0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47, 0xAAAAAAAAAAAAAA47 };
    ibsVec2(result, 7, ref_het_alt_het, het_ref_het_alt);

    expect_equal("ibsVec2 7 ref_het_alt_het v 7 het_ref_het_alt", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ibsVec2 7 ref_het_alt_het v 7 het_ref_het_alt", "%" PRIu64, result[1], 7 * ((uint64_t)4));
    expect_equal("ibsVec2 7 ref_het_alt_het v 7 het_ref_het_alt", "%" PRIu64, result[2], ((uint64_t)0));
  }
  {
    uint64_t result[3] = { 0, 0, 0 };
    uint64_t ref_het_alt_het[7] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D };
    ibsVec2(result, 7, ref_het_alt_het, ref_het_alt_het);

    expect_equal("ibsVec2 ref_het_alt_het v self", "%" PRIu64, result[0], ((uint64_t)0));
    expect_equal("ibsVec2 ref_het_alt_het v self", "%" PRIu64, result[1], ((uint64_t)0));
    expect_equal("ibsVec2 ref_het_alt_het v self", "%" PRIu64, result[2], 4*((uint64_t)7));
  }
  // ibsMat
  {
    uint64_t result[2*2*3] = { 0, 0, 0,  0, 0, 0,
                               0, 0, 0,  0, 0, 0 };
    uint64_t all_ref_het_alt_het[2*7] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D };
    ibsMat(result, 2, 7, all_ref_het_alt_het, all_ref_het_alt_het);

    expect_equal("ibsMat identical 0 0, ibs0", "%" PRIu64, resultIndex(result, 2, 0, 0, 0), ((uint64_t)0));
    expect_equal("ibsMat identical 0 0, ibs1", "%" PRIu64, resultIndex(result, 2, 0, 0, 1), ((uint64_t)0));
    expect_equal("ibsMat identical 0 0, ibs2", "%" PRIu64, resultIndex(result, 2, 0, 0, 2), 4*((uint64_t)7));
    expect_equal("ibsMat identical 0 1, ibs0", "%" PRIu64, resultIndex(result, 2, 0, 1, 0), ((uint64_t)0));
    expect_equal("ibsMat identical 0 1, ibs1", "%" PRIu64, resultIndex(result, 2, 0, 1, 1), ((uint64_t)0));
    expect_equal("ibsMat identical 0 1, ibs2", "%" PRIu64, resultIndex(result, 2, 0, 1, 2), 4*((uint64_t)7));
    expect_equal("ibsMat identical 1 1, ibs0", "%" PRIu64, resultIndex(result, 2, 1, 1, 0), ((uint64_t)0));
    expect_equal("ibsMat identical 1 1, ibs1", "%" PRIu64, resultIndex(result, 2, 1, 1, 1), ((uint64_t)0));
    expect_equal("ibsMat identical 1 1, ibs2", "%" PRIu64, resultIndex(result, 2, 1, 1, 2), 4*((uint64_t)7));
  }
  {
    uint64_t result[2*2*3] = { 0, 0, 0,  0, 0, 0,
                               0, 0, 0,  0, 0, 0 };
    uint64_t one_ibs1_rest_ibs2[2*7] =
      { 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D,
        0xAAAAAAAAAAAAAA1F, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D, 0xAAAAAAAAAAAAAA1D };
    ibsMat(result, 2, 7, one_ibs1_rest_ibs2, one_ibs1_rest_ibs2);

    expect_equal("ibsMat one-ibs1 0 0, ibs0", "%" PRIu64, resultIndex(result, 2, 0, 0, 0), ((uint64_t)0));
    expect_equal("ibsMat one-ibs1 0 0, ibs1", "%" PRIu64, resultIndex(result, 2, 0, 0, 1), ((uint64_t)0));
    expect_equal("ibsMat one-ibs1 0 0, ibs2", "%" PRIu64, resultIndex(result, 2, 0, 0, 2), 4*((uint64_t)7));
    expect_equal("ibsMat one-ibs1 0 1, ibs0", "%" PRIu64, resultIndex(result, 2, 0, 1, 0), ((uint64_t)0));
    expect_equal("ibsMat one-ibs1 0 1, ibs1", "%" PRIu64, resultIndex(result, 2, 0, 1, 1), ((uint64_t)1));
    expect_equal("ibsMat one-ibs1 0 1, ibs2", "%" PRIu64, resultIndex(result, 2, 0, 1, 2), 3+4*((uint64_t)6));
    expect_equal("ibsMat one-ibs1 1 1, ibs0", "%" PRIu64, resultIndex(result, 2, 1, 1, 0), ((uint64_t)0));
    expect_equal("ibsMat one-ibs1 1 1, ibs1", "%" PRIu64, resultIndex(result, 2, 1, 1, 1), ((uint64_t)0));
    expect_equal("ibsMat one-ibs1 1 1, ibs2", "%" PRIu64, resultIndex(result, 2, 1, 1, 2), 4*((uint64_t)7));
  }

  if (failures != 0) {
    printf("%" PRIu64 " test(s) failed.\n", failures);
  } else {
    printf("%" PRIu64 " test(s) succeeded.\n", successes);
  }

  return 0;
}
