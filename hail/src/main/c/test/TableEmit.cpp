#include "catch.hpp"
#include "hail/table/PartitionContext.h"
#include "hail/table/TableRead.h"
#include "hail/table/TableWrite.h"
#include "hail/table/TableMapRows.h"
#include "hail/table/TableFilterRows.h"
#include "hail/table/TableExplodeRows.h"
#include "TableEmit.h"
#include <cstring>
#include <string>
#include <vector>
#include <numeric>

namespace hail {

TEST_CASE("PartitionContext globals set correctly") {
  PartitionContext ctx { nullptr, "hello" };
  CHECK(strcmp(ctx.globals_, "hello") == 0);
}

TEST_CASE("PartitionContext regions reference counted correctly") {
  PartitionContext ctx;
  REQUIRE(ctx.pool_.num_regions() == 0);
  REQUIRE(ctx.pool_.num_free_regions() == 0);

  SECTION("can get + release region from context pool") {
    auto region = ctx.pool_.get_region();
    REQUIRE(ctx.pool_.num_regions() == 1);
    REQUIRE(ctx.pool_.num_free_regions() == 0);
    region = nullptr;
    REQUIRE(ctx.pool_.num_regions() == 1);
    REQUIRE(ctx.pool_.num_free_regions() == 1);
  }
}

TEST_CASE("TestDecoder/Encoder works") {
  PartitionContext ctx;
  std::vector<std::string> str_rows { "assafgh", "SDfghf" };

  TestStringDecoder dec { str_rows };
  TestStringEncoder enc;

  auto region = ctx.pool_.get_region();
  std::vector<char const *> rows;
  while (dec.decode_byte()) {
    rows.push_back(dec.decode_row(region.get()));
  }

  for (char const * row : rows) {
    enc.encode_byte(1);
    enc.encode_row(row);
  }

  CHECK(enc.rows_ == str_rows);
}

TEST_CASE("TableRead<TableWrite> works") {
  std::string globals = "hello";
  std::vector<std::string> str_rows { "assafghsg", "SDfghfasdfsadf", "foo22" };
  TestStringImplementation tester {str_rows, globals};

  char * gbuf = (char *) malloc(9);
  *reinterpret_cast<int *>(gbuf) = 5;
  strcpy(gbuf + sizeof(int), globals.c_str());
  PartitionContext ctx { nullptr, gbuf };

  using Writer = TableNativeWrite<TestStringEncoder>;
  SECTION("TableRead<TableWrite> works") {
    using Reader = TableNativeRead<Writer, TestStringDecoder>;
    auto transformed = str_rows;

    Reader reader { TestStringDecoder(str_rows), &ctx };
    while (reader.advance()) { reader.consume(); }
    CHECK(reader.end()->rows_ == transformed);
  }

  SECTION("TableRead<TableMapRows<TableWrite>> works") {
    using Reader = TableNativeRead<TableMapRows<Writer, AppendString>, TestStringDecoder>;
    auto transformed = tester.map([](std::string globals, std::string str) -> std::string { return str + globals; });

    Reader reader { TestStringDecoder(str_rows), &ctx };
    while (reader.advance()) { reader.consume(); }
    CHECK(reader.end()->rows_ == transformed);
  }

  SECTION("TableRead<TableFilterRows<TableWrite>> works") {
    using Reader = TableNativeRead<TableFilterRows<Writer, FilterString>, TestStringDecoder>;
    auto transformed = tester.filter([](std::string globals, std::string str) -> bool { return str.size() == globals.size(); });

    Reader reader { TestStringDecoder(str_rows), &ctx };
    while (reader.advance()) { reader.consume(); }
    CHECK(reader.end()->rows_ == transformed);
  }

  SECTION("TableRead<TableExplodeRows<TableWrite>> works") {
    using Reader = TableNativeRead<TableExplodeRows<Writer, ExplodeToChars>, TestStringDecoder>;
    auto lenf = [](std::string str) -> int { return str.size(); };
    auto get_i_f = [](std::string str, int i) -> std::string { return str.substr(i, 1); };
    auto transformed = tester.explode(lenf, get_i_f);

    Reader reader { TestStringDecoder(str_rows), &ctx };
    while (reader.advance()) { reader.consume(); }
    CHECK(reader.end()->rows_ == transformed);
  }
}

}
