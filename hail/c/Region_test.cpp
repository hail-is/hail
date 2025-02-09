#include "catch.hpp"
#include "hail/RegionPool.h"
#include <vector>
#include <ctime>
#include <cstring>
#include <iostream>

namespace hail {

TEST_CASE("region pools allocate and manage regions/blocks") {
  RegionPool pool;
  REQUIRE(pool.num_regions() == 0);
  REQUIRE(pool.num_free_regions() == 0);
  REQUIRE(pool.num_free_blocks() == 0);

  SECTION("regions can be requested from pool") {
    auto region = pool.get_region();
    CHECK(pool.num_regions() == 1);

    SECTION("freeing requested region returns to pool") {
      region = nullptr;
      CHECK(pool.num_regions() == 1);
      CHECK(pool.num_free_regions() == 1);
      CHECK(pool.num_free_blocks() == 1);
    }

    SECTION("blocks can be acquired for region") {
      region->allocate(4, 64*1024 - 3);

      CHECK(pool.num_regions() == 1);
      CHECK(pool.num_free_regions() == 0);
      CHECK(pool.num_free_blocks() == 0);

      SECTION("allocation at start of block allocates correctly") {
        auto off1 = region->allocate(1, 10);
        auto off2 = region->allocate(1, 10);
        CHECK(off2 - off1 == 10);
        region = nullptr;
      }

      SECTION("blocks are not released until region is released") {
        region->allocate(4, 10);
        CHECK(pool.num_free_blocks() == 0);
        region = nullptr;
        CHECK(pool.num_free_blocks() == 2);
      }

      SECTION("large chunks are not returned to block pool") {
        region->allocate(4, 5000);
        region = nullptr;
        CHECK(pool.num_free_blocks() == 1);
      }
      CHECK(pool.num_free_regions() == 1);
    }

    SECTION("referenced regions are not freed until referencing region is freed") {
      auto region2 = region->get_region();
      region2->add_reference_to(region);
      CHECK(pool.num_regions() == 2);
      region = nullptr;
      CHECK(pool.num_regions() == 2);
      CHECK(pool.num_free_regions() == 0);
      region2 = nullptr;
      CHECK(pool.num_regions() == 2);
      CHECK(pool.num_free_regions() == 2);
    }

    SECTION("copy assignment from one RegionPtr to another increments ref count") {
      auto region2 {region};
      region2 = nullptr;
      CHECK(pool.num_free_regions() == 0);
      region = nullptr;
      CHECK(pool.num_free_regions() == 1);
    }
  }
}

TEST_CASE("get_region() speed for allocating 10,000 regions in different batch sizes", "[!benchmark]") {
   RegionPool pool;
   REQUIRE(pool.num_free_regions() == 0);
   REQUIRE(pool.num_free_blocks() == 0);

   std::vector<RegionPtr> regions {};
   REQUIRE(regions.size() == 0);

   std::vector<int> batches { 5, 25, 100, 1000, 10000 };

   for(int size : batches) {
     SECTION(std::string("SIZE=") + std::to_string(size)) {
       for (int i = 0; i < 10000; i += size) {
         for (int j = 0; j < size; ++j) {
           regions.push_back(pool.get_region());
         }
         regions.clear();
       }
     }
   }
   REQUIRE(regions.size() == 0);

 }

TEST_CASE("block sizes work as expected") {
  RegionPool pool;
  REQUIRE(pool.num_free_regions() == 0);
  REQUIRE(pool.num_free_blocks() == 0);

  auto region1 = pool.get_region();
  REQUIRE(region1->get_block_size() == BLOCK_SIZE_1);
  region1 = nullptr;

  REQUIRE(pool.num_free_regions() == 1);
  REQUIRE(pool.num_free_blocks() == 1);

  auto region2 = pool.get_region(BLOCK_SIZE_2);
  REQUIRE(region2->get_block_size() == BLOCK_SIZE_2);

  REQUIRE(pool.num_free_regions() == 0);
  REQUIRE(pool.num_free_blocks() == 1);
  region2 = nullptr;
  REQUIRE(pool.num_free_regions() == 1);
  REQUIRE(pool.num_free_blocks() == 2);
}

}