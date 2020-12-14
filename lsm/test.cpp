#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include "lsm.h"

TEST_CASE("Simple LSM test", "") {
  LSM m;
  m.put(10, 7);
  m.put(63, 222);
  m.put(10, 5);
  REQUIRE( m.get(10) == 5 );
  REQUIRE( m.get(63) == 222 );
}
