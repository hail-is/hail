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
    Region::SharedPtr new_region();

  public:
    std::vector<Region *> free_regions_{};
    std::vector<std::unique_ptr<char[]>> free_blocks_{};
    RegionPool() = default;
    RegionPool(RegionPool &p) = delete;
    RegionPool(RegionPool &&p) = delete;
    std::unique_ptr<char[]> get_block();
    Region::SharedPtr get_region();

    //tracking methods:
    size_t num_regions() { return regions_.size(); }
    size_t num_free_regions() { return free_regions_.size(); }
    size_t num_free_blocks() { return free_blocks_.size(); }
};

struct ScalaRegionPool : public NativeObj {
  RegionPool pool_{};
  std::shared_ptr<ScalaRegion> get_region() { return std::make_shared<ScalaRegion>(this); }
  void own(RegionPool &&pool);
  virtual const char* get_class_name() { return "RegionPool"; }
};

}

#endif