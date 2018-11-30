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
  friend class Region;
  friend class ScalaRegionPool;

  private:
    std::vector<std::unique_ptr<Region>> regions_{};
    std::vector<Region *> free_regions_{};
    std::vector<std::unique_ptr<char[]>> free_blocks_{};
    std::unique_ptr<char[]> get_block();
    Region::SharedPtr new_region();

  public:
    RegionPool() = default;
    RegionPool(RegionPool &p) = delete;
    RegionPool(RegionPool &&p) = delete;
    Region::SharedPtr get_region();

    //tracking methods:
    size_t num_regions() { return regions_.size(); }
    size_t num_free_regions() { return free_regions_.size(); }
    size_t num_free_blocks() { return free_blocks_.size(); }
};

class ScalaRegionPool : public NativeObj {
  friend class ScalaRegion;
  RegionPool pool_{};
  public:
    std::shared_ptr<ScalaRegion> get_region() { return std::make_shared<ScalaRegion>(this); }
    void own(RegionPool &&pool);


    virtual const char* get_class_name() { return "RegionPool"; }
};

}

#endif