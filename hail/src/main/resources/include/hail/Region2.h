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

  struct Block {
    char * buf_;
    ssize_t size_;
    Block(ssize_t size) : buf_((char *) malloc(size)), size_(size) { }
    ~Block() { if (buf_) free(buf_); }
  };

  public:
    class Region {
      private:
        RegionPool * pool_;
        ssize_t block_offset_;
        std::shared_ptr<Block> current_block_;
        std::vector<std::shared_ptr<Block>> used_blocks_;
        std::vector<std::shared_ptr<Region>> parents_;
        char * allocate_new_block();
        char * allocate_big_chunk(ssize_t size);
      public:
        Region(RegionPool * pool);
        void clear();
        inline char * allocate(ssize_t alignment, ssize_t n) {
          ssize_t aligned_off = (block_offset_ + alignment - 1) & ~(alignment - 1);
          if (aligned_off + n <= block_size) {
            char* p = current_block_->buf_ + aligned_off;
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
    std::vector<Block *> free_blocks_;
    std::vector<Block *> free_sized_blocks_;
    struct RegionDeleter {
      RegionPool * pool_;
      RegionDeleter(RegionPool * pool) : pool_(pool) { }
      void operator()(Region* p) const;
    };
    struct BlockDeleter {
      RegionPool * pool_;
      BlockDeleter(RegionPool * pool) : pool_(pool) { }
      void operator()(Block* p) const;
    };
    struct SizedBlockDeleter {
      RegionPool * pool_;
      SizedBlockDeleter(RegionPool * pool) : pool_(pool) { }
      void operator()(Block* p) const;
    };
    RegionDeleter del_;
    BlockDeleter block_del_;
    SizedBlockDeleter sized_block_del_;
    std::shared_ptr<Block> new_block();
    std::shared_ptr<Block> new_sized_block(ssize_t size);
    std::shared_ptr<Region> new_region();
    std::shared_ptr<Block> get_block();
    std::shared_ptr<Block> get_sized_block(ssize_t size);

  public:
    RegionPool();
    RegionPool(RegionPool &p) = delete;
    RegionPool(RegionPool &&p) = delete;
    ~RegionPool();
    std::shared_ptr<Region> get_region();

    //tracking methods:
    ssize_t num_free_regions() { return free_regions_.size(); }
    ssize_t num_free_blocks() { return free_blocks_.size(); }
    ssize_t num_free_sized_blocks() { return free_sized_blocks_.size(); }
    void sized_block_sizes(ssize_t * size_array) {
      for (size_t i = 0; i < free_sized_blocks_.size(); ++i) {
        size_array[i] = free_sized_blocks_[i]->size_;
      }
    }
};

using Region2 = RegionPool::Region;

template<typename Encoder>
class RegionPoolTracker {
  private:
    RegionPool * pool_;
    UpcallEnv up_;
    Encoder * encoder_;

  public:
    RegionPoolTracker(RegionPool * pool, Encoder * enc) : pool_(pool), encoder_(enc) { }
    void write(NativeStatus* st) {
      encoder_->encode_byte(st, 1);
      ssize_t num_sized = pool_->num_free_sized_blocks();
      char * array = (char *) malloc(sizeof(long) + (num_sized * sizeof(long)));
      *((int *) array) = (int) num_sized;
      pool_->sized_block_sizes((ssize_t *) (array + sizeof(ssize_t)));
      long * row = (long*) malloc((sizeof(long) * 3));
      *row = (long) pool_->num_free_regions();
      *(row + 1) = (long) pool_->num_free_blocks();
      *(row + 2) = reinterpret_cast<long>(array);
      encoder_->encode_row(st, (char *) row);
      free(row);
    }

    void stop(NativeStatus* st) {
      encoder_->encode_byte(st, 0);
      encoder_->flush(st);
    }
};

}

#endif