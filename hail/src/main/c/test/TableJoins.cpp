#include "catch.hpp"
#include "hail/table/PartitionContext.h"
#include "hail/table/TableRead.h"
#include "hail/table/TableWrite.h"
#include "hail/table/TableJoins.h"
#include "hail/table/Linearizers.h"
#include "TableEmit.h"
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

namespace hail {

struct JoinOnFirstChar {
  int compare(char const * left, char const * right) {
    if (*reinterpret_cast<int const *>(left) == 0) { return -1; }
    if (*reinterpret_cast<int const *>(right) == 0) { return 1; }

    return *(left + sizeof(int)) - *(right + sizeof(int));
  }

  char const * operator()(NativeStatus * st, Region * region, char const * left, char const * right) {
    if (right == nullptr) { return left; }
    int rlen = *reinterpret_cast<int const *>(right);
    if (rlen == 0) { return left; }
    int llen = *reinterpret_cast<int const *>(left);

    auto off = region->allocate(sizeof(int) + llen + rlen);
    *reinterpret_cast<int *>(off) = llen + rlen - 1;
    memcpy(off + sizeof(int), left + sizeof(int), llen);
    memcpy(off + sizeof(int) + llen, right + sizeof(int) + 1, rlen - 1);
    return off;
  }

};

TEST_CASE("TableLeftJoinRightDistinct on first character of str") {
  std::vector<std::string> t1 { "bar1", "baz12", "foo123", "goo1234" };
  std::vector<std::string> t2 { "bar12345", "foo123456", "qux1234567" };

  PartitionContext ctx;

  TestStringDecoder && dec1 {t1};
  TestStringDecoder && dec2 {t2};

  using RightReaderStream = LinearizedPullStream<TableNativeRead<NestedLinearizerEndpoint, TestStringDecoder>>;
  using Writer = TableNativeWrite<TestStringEncoder>;
  using Join = TableLeftJoinRightDistinct<Writer, RightReaderStream, JoinOnFirstChar>;
  using LeftReader = TableNativeRead<Join, TestStringDecoder>;

  SECTION("1") {
    std::vector<std::string> expected { "bar1ar12345", "baz12ar12345", "foo123oo123456", "goo1234" };

    RightReaderStream r_stream { dec2, &ctx };
    LeftReader reader { dec1, &r_stream, &ctx };
    while (reader.advance()) { reader.consume(); }

    CHECK(reader.end()->rows_ == expected);
    // the last row in `right` is unmatched so will stay in Join object until its destruction.
    CHECK(ctx.pool_.num_free_regions() == ctx.pool_.num_regions() - 1);
  }

  SECTION("2") {
    std::vector<std::string> expected { "bar12345ar1", "foo123456oo123", "qux1234567" };

    RightReaderStream r_stream { dec1, &ctx };
    LeftReader reader { dec2, &r_stream, &ctx };
    while (reader.advance()) { reader.consume(); }

    CHECK(reader.end()->rows_ == expected);
    // the last row in `right` is unmatched so will stay in Join object until its destruction.
    CHECK(ctx.pool_.num_free_regions() == ctx.pool_.num_regions());

  }
}

}
