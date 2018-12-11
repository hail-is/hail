#ifndef HAIL_REGION_H
#define HAIL_REGION_H 1

#include <memory>
#include <vector>
#include "hail/NativeStatus.h"
#include "hail/NativeObj.h"

namespace hail {

#define REGION_BLOCK_SIZE 64*1024
#define REGION_BLOCK_THRESHOLD 4*1024

struct ScalaRegionPool;
class RegionPool;

class Region {
  friend class RegionPool;
  friend struct ScalaRegionPool;
  public:
    class SharedPtr {
      friend class RegionPool;
      private:
        Region * region_;
        void clear();
        explicit SharedPtr(Region * region) : region_(region) {
          if (region_ != nullptr) { ++(region_->references_); }
        }
      public:
        SharedPtr(const SharedPtr &ptr) : SharedPtr(ptr.region_) { }
        SharedPtr(SharedPtr &&ptr) : region_(ptr.region_) { ptr.region_ = nullptr; }
        ~SharedPtr() { clear(); }

        void swap(SharedPtr &other) { std::swap(region_, other.region_); }
        SharedPtr& operator=(SharedPtr other) {
          swap(other);
          return *this;
        }
        SharedPtr& operator=(std::nullptr_t) noexcept {
          clear();
          return *this;
        }
        Region * get() { return region_; }
        Region & operator*() { return *region_; }
        Region * operator->() { return region_; }
    };

  private:
    RegionPool * pool_;
    int references_ {0};
    size_t block_offset_;
    std::unique_ptr<char[]> current_block_;
    std::vector<std::unique_ptr<char[]>> used_blocks_{};
    std::vector<std::unique_ptr<char[]>> big_chunks_{};
    std::vector<SharedPtr> parents_{};
    char * allocate_new_block(size_t n);
    char * allocate_big_chunk(size_t size);
    explicit Region(RegionPool * pool);
  public:
    Region(Region &pool) = delete;
    Region(Region &&pool) = delete;
    Region& operator=(Region pool) = delete;
    void clear();
    void align(size_t a) {
      block_offset_ = (block_offset_ + a-1) & ~(a-1);
    }

    char* allocate(size_t n) {
      if (block_offset_ + n <= REGION_BLOCK_SIZE) {
        char* p = (current_block_.get() + block_offset_);
        block_offset_ += n;
        return p;
      } else {
        return (n <= REGION_BLOCK_THRESHOLD) ? allocate_new_block(n) : allocate_big_chunk(n);
      }
    }

    char * allocate(size_t alignment, size_t n) {
      size_t aligned_off = (block_offset_ + alignment - 1) & ~(alignment - 1);
      if (aligned_off + n <= REGION_BLOCK_SIZE) {
        char* p = current_block_.get() + aligned_off;
        block_offset_ = aligned_off + n;
        return p;
      } else {
        return (n <= REGION_BLOCK_THRESHOLD) ? allocate_new_block(n) : allocate_big_chunk(n);
      }
    }
    SharedPtr get_region();
    void add_reference_to(SharedPtr region);
};

using RegionPtr = Region::SharedPtr;

class ScalaRegion : public NativeObj {
  public:
    RegionPtr region_;
    ScalaRegion(ScalaRegionPool * pool);
    void align(size_t alignment) { region_->align(alignment); }
    char * allocate(size_t alignment, size_t n) { return region_->allocate(alignment, n); }
    char * allocate(size_t n) { return region_->allocate(n); }

    virtual const char* get_class_name() { return "Region"; }
    virtual ~ScalaRegion() = default;
  };

}

#endif