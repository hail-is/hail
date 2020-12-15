#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include "lsm.h"
#include <map>

TEST_CASE("Simple LSM test", "") {
LSM m;
m.put(10, 7);
m.put(63, 222);
m.put(10, 5);
REQUIRE( m.get(10) == 5 );
REQUIRE( m.get(63) == 222 );
}
//creating to files, calling merge, and then checking that the resulting file is the one you want.
TEST_CASE("Merge 2 files", "") {
  LSM m;
  Level l;

  std::map<int32_t, maybe_value> m_older;
  m_older.insert_or_assign(10, 7);
  m_older.insert_or_assign(75, 9);
  m_older.insert_or_assign(4, 8);
  m_older.insert_or_assign(16, 65);
  std::map<int32_t, maybe_value> m_newer;
  m_newer.insert_or_assign(17, 4);
  m_newer.insert_or_assign(1, 83);
  m_newer.insert_or_assign(5, 13);
  m_newer.insert_or_assign(98, 107);
  f_older = l.write_to_file(m_older, "older_file");
  f_newer = l.write_to_file(m_newer, "newer_file");
  f_merged = l.merge(f_older, f_newer, "merged_file");
  std::map<int32_t, maybe_value> m_merged;
  l.read_to_map(f_merged, m_merged);

  for (auto const&x : m_merged) {
    std::cout << x.first << "\n";
  }
}
// test overwriting key