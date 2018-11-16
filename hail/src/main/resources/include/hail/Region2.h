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
      private:
        RegionPool * pool_;
        ssize_t block_offset_;
        char * current_block_;
        std::vector<char *> used_blocks_;
        std::vector<char *> big_chunks_;
        std::vector<std::shared_ptr<Region>> parents_;
        char * allocate_new_block();
        char * allocate_big_chunk(ssize_t size);
      public:
        Region(RegionPool * pool);
        void clear();
        inline char * allocate(ssize_t alignment, ssize_t n) {
          ssize_t aligned_off = (block_offset_ + alignment - 1) & ~(alignment - 1);
          if (aligned_off + n <= block_size) {
            char* p = current_block_ + aligned_off;
            block_offset_ = aligned_off + n;
            return p;
          } else {
            return (n <= block_threshold) ? allocate_new_block() : allocate_big_chunk(n);
          }
        }
        std::shared_ptr<Region> get_region();
        void add_reference_to(std::shared_ptr<Region> region);
    };

  private:
    std::vector<Region *> free_regions_;
    std::vector<char *> free_blocks_;
    struct RegionDeleter {
      RegionPool * pool_;
      RegionDeleter(RegionPool * pool) : pool_(pool) { }
      void operator()(Region* p) const;
    };
    RegionDeleter del_;
    char * get_block();
    std::shared_ptr<Region> new_region();

  public:
    RegionPool();
    RegionPool(RegionPool &p) = delete;
    RegionPool(RegionPool &&p) = delete;
    ~RegionPool();
    std::shared_ptr<Region> get_region();

    //tracking methods:
    ssize_t num_free_regions() { return free_regions_.size(); }
    ssize_t num_free_blocks() { return free_blocks_.size(); }
};

using Region2 = RegionPool::Region;

template<typename Encoder>
class RegionPoolTracker {
  private:
    NativeStatus* st_;
    RegionPool * pool_;
    Encoder * encoder_;

  public:
    RegionPoolTracker(NativeStatus * st, RegionPool * pool, Encoder * enc) : st_(st), pool_(pool), encoder_(enc) { }
    void log_state() {
      encoder_->encode_byte(st_, 1);
      long * row = (long*) malloc((sizeof(long) * 2));
      *row = (long) pool_->num_free_regions();
      *(row + 1) = (long) pool_->num_free_blocks();
      encoder_->encode_row(st_, (char *) row);
      free(row);
    }

    void stop() {
      encoder_->encode_byte(st_, 0);
      encoder_->flush(st_);
    }
};

}

#endif