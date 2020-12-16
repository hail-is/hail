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
