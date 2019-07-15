#ifndef HAIL_REGIONPOOL_H
#define HAIL_REGIONPOOL_H 1

#include <memory>
#include <vector>
#include <utility>
#include "hail/NativeStatus.h"
#include "hail/NativeObj.h"
#include "hail/Region.h"

namespace hail {

class RegionPool {
  friend struct ScalaRegionPool;

  private:
    std::vector<std::unique_ptr<Region>> regions_{};
    Region::SharedPtr new_region(size_t block_size);

  public:
    std::vector<Region *> free_regions_{};
    std::vector<std::unique_ptr<char[]>> free_blocks_1_{};
    std::vector<std::unique_ptr<char[]>> free_blocks_2_{};
    std::vector<std::unique_ptr<char[]>> free_blocks_3_{};
    std::vector<std::unique_ptr<char[]>> free_blocks_4_{};
    RegionPool() = default;
    RegionPool(RegionPool &p) = delete;
    RegionPool(RegionPool &&p) = delete;
    std::unique_ptr<char[]> get_block(size_t size);
    Region::SharedPtr get_region(size_t block_size);

    std::vector<std::unique_ptr<char[]>> * get_block_pool(size_t size) {
      switch(size) {
        case BLOCK_SIZE_1: return &free_blocks_1_;
        case BLOCK_SIZE_2: return &free_blocks_2_;
        case BLOCK_SIZE_3: return &free_blocks_3_;
        case BLOCK_SIZE_4: return &free_blocks_4_;
        default: return &free_blocks_1_;
      }
    }

    std::unique_ptr<char[]> get_block() { return get_block(BLOCK_SIZE_1); }
    Region::SharedPtr get_region() { return get_region(BLOCK_SIZE_1); }

    //tracking methods:
    size_t num_regions() { return regions_.size(); }
    size_t num_free_regions() { return free_regions_.size(); }
    size_t num_free_blocks() {
      return free_blocks_1_.size() +
        free_blocks_2_.size() +
        free_blocks_3_.size() +
        free_blocks_4_.size(); }
};

struct ScalaRegionPool : public NativeObj {
  RegionPool pool_{};
  std::shared_ptr<ScalaRegion> get_region(size_t block_size) { return std::make_shared<ScalaRegion>(this, block_size); }
  void own(RegionPool &&pool);
  virtual const char* get_class_name() { return "RegionPool"; }
};

}

#endif