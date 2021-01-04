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
  File f_merged = l.merge(f_older, f_newer);
  std::map<int32_t, maybe_value> m_merged;
  l.read_to_map(f_merged.filename, m_merged);

  REQUIRE( m_merged[10].v == 7 );
  REQUIRE( m_merged[75].v == 9 );
  REQUIRE( m_merged[4].v == 8 );
  REQUIRE( m_merged[16].v == 65 );
  REQUIRE( m_merged[17].v == 4 );
  REQUIRE( m_merged[1].v == 83 );
  REQUIRE( m_merged[5].v == 13 );
  REQUIRE( m_merged[98].v == 107 );
}
TEST_CASE("Merge files overwrite key and delete key", "") {
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
  File f_merged = l.merge(f_older, f_newer);
  std::map<int32_t, maybe_value> m_merged;
  l.read_to_map(f_merged.filename, m_merged);

  REQUIRE( m_merged[10].v == 13 );
  REQUIRE( m_merged[75].v == 9 );
  REQUIRE( m_merged[4].v ==  0 );
  REQUIRE( m_merged[16].v == 65 );
  REQUIRE( m_merged[17].v == 4 );
  REQUIRE( m_merged[1].v == 83 );
}
TEST_CASE("Merge two files overlapping keys with deletes and puts", "") {
  Level l{0, "db"};

  std::map<int32_t, maybe_value> m_older;
  m_older.insert_or_assign(10, maybe_value(7, 0));
  m_older.insert_or_assign(75, maybe_value(9, 0));
  m_older.insert_or_assign(4, maybe_value(8, 0));
  m_older.insert_or_assign(10, maybe_value(0, 1));
  std::map<int32_t, maybe_value> m_newer;
  m_newer.insert_or_assign(2, maybe_value(22, 0));
  m_newer.insert_or_assign(75, maybe_value(0, 1));
  m_newer.insert_or_assign(3, maybe_value(42, 0));
  m_newer.insert_or_assign(4, maybe_value(44, 0));
  File f_older = l.write_to_file(m_older, "older_file");
  File f_newer = l.write_to_file(m_newer, "newer_file");
  File f_merged = l.merge(f_older, f_newer);
  std::map<int32_t, maybe_value> m_merged;
  l.read_to_map(f_merged.filename, m_merged);

  REQUIRE( m_merged[10].v == 0 );
  REQUIRE( m_merged[10].is_deleted == 1 );
  REQUIRE( m_merged[75].v == 0 );
  REQUIRE( m_merged[75].is_deleted == 1 );
  REQUIRE( m_merged[4].v == 44 );
  REQUIRE( m_merged[3].v == 42 );
  REQUIRE( m_merged[2].v == 22 );

}
TEST_CASE("puts and deletes", "") {
  LSM m{"db"};

  m.put(10, 7);
  m.put(75, 9);
  m.put(4, 8);
  m.del(10);

  m.put(4, 44);
  m.put(3, 42);
  m.del(75);
  m.put(2, 22);

  REQUIRE( m.get(10) == std::nullopt );
  REQUIRE( m.get(4) == 44 );
  REQUIRE( m.get(3) == 42 );
  REQUIRE( m.get(75) == std::nullopt );
  REQUIRE( m.get(2) == 22 );
}
