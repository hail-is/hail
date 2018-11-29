#ifndef HAIL_REGION2_H
#define HAIL_REGION2_H 1

#include <memory>
#include <vector>
#include <utility>
#include "hail/Upcalls.h"
#include "hail/NativeStatus.h"

namespace hail {

class RegionPool {
  static constexpr ssize_t block_size = 64*1024;
  static constexpr ssize_t block_threshold = 4096;

  public:
    class Region {
      public:
        class SharedPtr {
          private:
            Region * region_;
            void clear();
          public:
            SharedPtr(Region * region) {
              if (region_ != nullptr) { ++(region_->references_); }
            }
            SharedPtr(const SharedPtr &ptr) : SharedPtr(ptr.region_) { }
            SharedPtr(SharedPtr &&ptr) : region_(nullptr) {
              region_ = ptr.region_;
              ptr.region_ = nullptr;
            }
            ~SharedPtr() { clear(); }

            void swap(RegionPtr &other) { std::swap(region_, other.region_); }
            RegionPtr& RegionPtr::operator=(RegionPtr other) {
              swap(other);
              return *this;
            }
            SharedPtr& operator=(std::nullptr_t) noexcept {
              clear();
              return *this;
            }
            inline Region * get() { return region_; }
            inline Region & operator*() { return *region_; }
            inline Region * operator->() { return region_; }
        };

      private:
        RegionPool * pool_;
        int references_ {0};
        size_t block_offset_;
        std::unique_ptr<char[]> current_block_;
        std::vector<std::unique_ptr<char[]>> used_blocks_{};
        std::vector<std::unique_ptr<char[]>> big_chunks_{};
        std::vector<SharedPtr> parents_{};
        char * allocate_new_block();
        char * allocate_big_chunk(size_t size);
      public:
        Region(RegionPool * pool);
        void clear();
        inline char * allocate(size_t alignment, size_t n) {
          size_t aligned_off = (block_offset_ + alignment - 1) & ~(alignment - 1);
          if (aligned_off + n <= block_size) {
            char* p = current_block_.get() + aligned_off;
            block_offset_ = aligned_off + n;
            return p;
          } else {
            return (n <= block_threshold) ? allocate_new_block() : allocate_big_chunk(n);
          }
        }
        SharedPtr get_region();
        void add_reference_to(SharedPtr region);
    };

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

using Region2 = RegionPool::Region;
using RegionPtr = Region2::SharedPtr;

}

#endif