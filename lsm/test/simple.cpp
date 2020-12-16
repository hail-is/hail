#include <catch2/catch.hpp>
#include "lsm.h"

TEST_CASE("Simple LSM test", "") {
  LSM m{"db"};
  m.put(10, 7);
  m.put(63, 222);
  m.put(10, 5);
  REQUIRE( m.get(10) == 5 );
  REQUIRE( m.get(63) == 222 );
}
TEST_CASE("Merge 2 files", "") {
  LSM m{"db"};
  Level l{0, "db"};

  std::map<int32_t, maybe_value> m_older;
  m_older.insert_or_assign(10, maybe_value(7, 0));
  m_older.insert_or_assign(75, maybe_value(9, 0));
  m_older.insert_or_assign(4, maybe_value(8, 0));
  m_older.insert_or_assign(16, maybe_value(65, 0));
  std::map<int32_t, maybe_value> m_newer;
  m_newer.insert_or_assign(17, maybe_value(4, 0));
  m_newer.insert_or_assign(1, maybe_value(83, 0));
  m_newer.insert_or_assign(5, maybe_value(13, 0));
  m_newer.insert_or_assign(98, maybe_value(107, 0));
  File f_older = l.write_to_file(m_older, "older_file");
  File f_newer = l.write_to_file(m_newer, "newer_file");
  File f_merged = l.merge(f_older, f_newer, "merged_file");
  std::map<int32_t, maybe_value> m_merged;
  l.read_to_map(f_merged, m_merged);

  for (auto const&x : m_merged) {
    std::cout << x.first << "\n";
  }
}
TEST_CASE("Merge files overwrite key and delete key", "") {
  LSM m{"db"};
  Level l{0, "db"};

  std::map<int32_t, maybe_value> m_older;
  m_older.insert_or_assign(10, maybe_value(7, 0));
  m_older.insert_or_assign(75, maybe_value(9, 0));
  m_older.insert_or_assign(4, maybe_value(8, 0));
  m_older.insert_or_assign(16, maybe_value(65, 0));
  std::map<int32_t, maybe_value> m_newer;
  m_newer.insert_or_assign(17, maybe_value(4, 0));
  m_newer.insert_or_assign(1, maybe_value(83, 0));
  m_newer.insert_or_assign(10, maybe_value(13, 0));
  m_newer.insert_or_assign(4, maybe_value(0, 1));
  File f_older = l.write_to_file(m_older, "older_file");
  File f_newer = l.write_to_file(m_newer, "newer_file");
  File f_merged = l.merge(f_older, f_newer, "merged_file");
  std::map<int32_t, maybe_value> m_merged;
  l.read_to_map(f_merged, m_merged);

  for (auto const&x : m_merged) {
    std::cout << x.first << ":" << x.second.v << "\n";
  }
}
