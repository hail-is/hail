#include "catch.hpp"
#include "hail/Region2.h"

namespace hail {

TEST_CASE("region pools allocate and manage regions/blocks") {
  RegionPool pool;
  REQUIRE(pool.num_free_regions() == 0);
  REQUIRE(pool.num_free_blocks() == 0);

  SECTION("regions can be requested from pool") {
    auto region = pool.get_region();

    SECTION("freeing requested region returns to pool") {
      region = nullptr;
      REQUIRE(pool.num_free_regions() == 1);
      REQUIRE(pool.num_free_blocks() == 0);
    }

    SECTION("blocks can be acquired for region") {
      region->allocate(4, 64*1024 - 3);

      REQUIRE(pool.num_free_regions() == 0);
      REQUIRE(pool.num_free_blocks() == 0);

      SECTION("blocks are not released until region is released") {
        region->allocate(4, 10);
        REQUIRE(pool.num_free_blocks() == 0);
        region = nullptr;
        REQUIRE(pool.num_free_regions() == 1);
        REQUIRE(pool.num_free_blocks() == 1);
      }

      SECTION("large chunks are not returned to block pool") {
        region->allocate(4, 5000);
        REQUIRE(pool.num_free_blocks() == 0);
        region = nullptr;
        REQUIRE(pool.num_free_regions() == 1);
        REQUIRE(pool.num_free_blocks() == 0);
      }
    }

    SECTION("referenced regions are not freed until referencing region is freed") {
      auto region2 = region->get_region();
      region2->add_reference_to(region);
      region = nullptr;
      REQUIRE(pool.num_free_regions() == 0);
      region2 = nullptr;
      REQUIRE(pool.num_free_regions() == 2);
    }
  }
}

}