#ifndef HAIL_REGION_H
#define HAIL_REGION_H 1

#include <memory>
#include <vector>
#include "hail/NativeStatus.h"
#include "hail/NativeObj.h"
#include "hail/Utils.h"

namespace hail {

#define BLOCK_SIZE_1 64*1024
#define BLOCK_SIZE_2 8*1024
#define BLOCK_SIZE_3 1024
#define BLOCK_SIZE_4 256

#define BLOCK_THRESHOLD 4 * 1024

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
        SharedPtr(std::nullptr_t) : region_(nullptr) { }
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
        bool operator==(std::nullptr_t) noexcept {
          return region_ == nullptr;
        }
        Region * get() { return region_; }
        Region & operator*() { return *region_; }
        Region * operator->() { return region_; }
    };

  private:
    RegionPool * pool_;
    int references_ {0};
    size_t block_size_;
    size_t block_threshold_;
    size_t block_offset_;
    std::unique_ptr<char[]> current_block_;
    std::vector<std::unique_ptr<char[]>> used_blocks_{};
    std::vector<std::unique_ptr<char[]>> big_chunks_{};
    std::vector<SharedPtr> parents_{};
    char * allocate_new_block(size_t n);
    char * allocate_big_chunk(size_t size);
    explicit Region(RegionPool * pool, size_t block_size);
  public:
    Region(Region &pool) = delete;
    Region(Region &&pool) = delete;
    Region& operator=(Region pool) = delete;
    void set_block_size(size_t block_size) {
      if (current_block_ == nullptr) {
        block_size_ = block_size;
        block_threshold_ = (block_size < BLOCK_THRESHOLD) ? block_size : BLOCK_THRESHOLD;
      } else {
        throw new FatalError("tried to set blocksize on non-empty region");
      }
    }
    void clear();
    void align(size_t a) {
      block_offset_ = (block_offset_ + a-1) & ~(a-1);
    }

    char* allocate(size_t n) {
      if (block_offset_ + n <= block_size_) {
        char* p = (current_block_.get() + block_offset_);
        block_offset_ += n;
        return p;
      } else {
        return (n <= block_threshold_) ? allocate_new_block(n) : allocate_big_chunk(n);
      }
    }

    char * allocate(size_t alignment, size_t n) {
      size_t aligned_off = (block_offset_ + alignment - 1) & ~(alignment - 1);
      if (aligned_off + n <= block_size_) {
        char* p = current_block_.get() + aligned_off;
        block_offset_ = aligned_off + n;
        return p;
      } else {
        return (n <= block_threshold_) ? allocate_new_block(n) : allocate_big_chunk(n);
      }
    }
    SharedPtr get_region(size_t block_size);
    SharedPtr get_region() { return get_region(block_size_); }
    void add_reference_to(SharedPtr region);

    size_t get_num_parents();
    void set_num_parents(int n);
    void set_parent_reference(SharedPtr region, int i);
    SharedPtr get_parent_reference(int i);
    SharedPtr new_parent_reference(int i, size_t block_size);
    void clear_parent_reference(int i);

    size_t get_block_size() { return block_size_; }
    int get_num_chunks() { return big_chunks_.size(); }
    int get_num_used_blocks() { return used_blocks_.size(); }
    int get_current_offset() { return block_offset_; }
    long get_block_address() { return reinterpret_cast<long>(current_block_.get()); }
};

using RegionPtr = Region::SharedPtr;

class ScalaRegion : public NativeObj {
  public:
    RegionPtr region_;
    ScalaRegion(ScalaRegionPool * pool, size_t block_size);
    ScalaRegion(std::nullptr_t);
    void align(size_t alignment) { region_->align(alignment); }
    char * allocate(size_t alignment, size_t n) { return region_->allocate(alignment, n); }
    char * allocate(size_t n) { return region_->allocate(n); }
    // should only be used for things like Compile and PackDecoder, where underlying region is not supposed to change.
    Region * get_wrapped_region() { return region_.get(); };

    virtual const char* get_class_name() { return "Region"; }
    virtual ~ScalaRegion() = default;
  };

}

#endif