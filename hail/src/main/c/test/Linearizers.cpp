#include "catch.hpp"
#include "hail/table/PartitionContext.h"
#include "hail/table/TableRead.h"
#include "hail/table/TableWrite.h"
#include "hail/table/TableMapRows.h"
#include "hail/table/TableFilterRows.h"
#include "hail/table/TableExplodeRows.h"
#include "hail/table/Linearizers.h"
#include "TableEmit.h"
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

namespace hail {

TEST_CASE("Linearized Map/Filter/Explode works") {
  std::string globals = "hello";
  std::vector<std::string> str_rows { "assafghsg", "SDfghfasdfsadf", "foo22" };

  char * gbuf = (char *) malloc(9);
  *reinterpret_cast<int *>(gbuf) = 5;
  strcpy(gbuf + sizeof(int), globals.c_str());
  PartitionContext ctx { nullptr, gbuf };

  TestStringImplementation tester {str_rows, globals};
  TestStringEncoder encoder;

  SECTION("TableRead<TableWrite> works") {
    auto transformed = str_rows;
    SECTION("Nested") {
      using Reader = TableNativeRead<NestedLinearizerEndpoint, TestStringDecoder>;

      LinearizedPullStream<Reader> reader { Reader(TestStringDecoder(str_rows), &ctx) };
      for (char const * row : reader) {
        encoder.encode_byte(1);
        encoder.encode_row(row);
      }
      encoder.encode_byte(0);
      CHECK(encoder.rows_ == transformed);
    }

    SECTION("Unnested") {
      using Reader = TableNativeRead<UnnestedLinearizerEndpoint, TestStringDecoder>;

      LinearizedPullStream<Reader> reader { Reader(TestStringDecoder(str_rows), &ctx) };
      for (char const * row : reader) {
        encoder.encode_byte(1);
        encoder.encode_row(row);
      }
      encoder.encode_byte(0);
      CHECK(encoder.rows_ == transformed);
    }
  }

  SECTION("Linearized TableRead<TableMapRows> works") {
    auto transformed = tester.map([](std::string globals, std::string str) -> std::string { return str + globals; });
    SECTION("Nested") {
      using Reader = TableNativeRead<TableMapRows<NestedLinearizerEndpoint, AppendString>, TestStringDecoder>;

      LinearizedPullStream<Reader> reader { Reader(TestStringDecoder(str_rows), &ctx) };
      for (char const * row : reader) {
        encoder.encode_byte(1);
        encoder.encode_row(row);
      }
      encoder.encode_byte(0);
      CHECK(encoder.rows_ == transformed);
    }

    SECTION("Unnested") {
      using Reader = TableNativeRead<TableMapRows<UnnestedLinearizerEndpoint, AppendString>, TestStringDecoder>;

      LinearizedPullStream<Reader> reader { Reader(TestStringDecoder(str_rows), &ctx) };
      for (char const * row : reader) {
        encoder.encode_byte(1);
        encoder.encode_row(row);
      }
      encoder.encode_byte(0);
      CHECK(encoder.rows_ == transformed);
    }
  }

  SECTION("TableRead<TableFilterRows<TableWrite>> works") {
    auto transformed = tester.filter([](std::string globals, std::string str) -> bool { return str.size() == globals.size(); });

    SECTION("Nested") {
      using Reader = TableNativeRead<TableFilterRows<NestedLinearizerEndpoint, FilterString>, TestStringDecoder>;
      LinearizedPullStream<Reader> reader { Reader(TestStringDecoder(str_rows), &ctx) };
      for (char const * row : reader) {
        encoder.encode_byte(1);
        encoder.encode_row(row);
      }
      encoder.encode_byte(0);
      CHECK(encoder.rows_ == transformed);
    }

    SECTION("Unnested") {
      using Reader = TableNativeRead<TableFilterRows<UnnestedLinearizerEndpoint, FilterString>, TestStringDecoder>;
      LinearizedPullStream<Reader> reader { Reader(TestStringDecoder(str_rows), &ctx) };
      for (char const * row : reader) {
        encoder.encode_byte(1);
        encoder.encode_row(row);
      }
      encoder.encode_byte(0);
      CHECK(encoder.rows_ == transformed);
    }
  }

  SECTION("TableRead<TableExplodeRows<TableWrite>> works") {
    using Reader = TableNativeRead<TableExplodeRows<NestedLinearizerEndpoint, ExplodeToChars>, TestStringDecoder>;
    auto lenf = [](std::string str) -> int { return str.size(); };
    auto get_i_f = [](std::string str, int i) -> std::string { return str.substr(i, 1); };
    auto transformed = tester.explode(lenf, get_i_f);

    LinearizedPullStream<Reader> reader { Reader(TestStringDecoder(str_rows), &ctx) };
    for (char const * row : reader) {
      encoder.encode_byte(1);
      encoder.encode_row(row);
    }
    encoder.encode_byte(0);
    CHECK(encoder.rows_ == transformed);
  }
}

}
