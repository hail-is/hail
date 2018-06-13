#ifndef HAIL_REGION_H
#define HAIL_REGION_H 1

#include "hail/NativeObj.h"
#include "hail/NativePtr.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <memory>
#include <vector>

namespace hail {

class Region;
using RegionPtr = std::shared_ptr<Region>;

class Region : public NativeObj {
private:
  static constexpr ssize_t kChunkCap = 64*1024;
  static constexpr ssize_t kMaxSmall = 4*1024;
  static constexpr ssize_t kNumBigToKeep = 4;
  
  struct CharArray {
    char buf_[1];
  };
  
  class UniqueChunk {
   public:
    std::unique_ptr<CharArray> ptr_;
    ssize_t size_;
    
    UniqueChunk() : ptr_(), size_(0) { }
    UniqueChunk(ssize_t n) : ptr_(), size_(0) { alloc(n); }
   
    UniqueChunk(UniqueChunk&& b) = default;   
    UniqueChunk& operator=(UniqueChunk&& b) = default;
    
    char* get() const { return ptr_.get()->buf_; }
    
    void alloc(ssize_t n) {
      ptr_.reset((CharArray*)malloc(n));
      size_ = n;
    }
    
    bool empty() const { return !ptr_; }
  };
  
public:
  UniqueChunk chunk_;
  ssize_t pos_;
  std::vector<UniqueChunk> free_chunks_;
  std::vector<UniqueChunk> full_chunks_;
  std::vector<UniqueChunk> big_free_;
  std::vector<UniqueChunk> big_used_;
  std::vector<RegionPtr> required_regions_;

 public:  
  Region() :
    chunk_(),
    pos_(kChunkCap) {
  }
  
  // clear_but_keep_mem() will make the Region empty without free'ing chunks
  void clear_but_keep_mem();
  
 private:
  void new_chunk();
  
  // Restrict the choice of block sizes to improve re-use
  ssize_t choose_big_size(ssize_t n);
  
  char* allocate_big(ssize_t n);
 
 public: 
  inline void align(ssize_t a) {
    pos_ = (pos_ + a-1) & ~(a-1);
  }
  
  inline char* allocate(ssize_t a, ssize_t n) {
    ssize_t mask = (a-1);
    if (n <= kMaxSmall) {
      ssize_t apos = ((pos_ + mask) & ~mask);
      if (apos+n > kChunkCap) {
        new_chunk();
        apos = 0;
      }
      char* p = (chunk_.get() + apos);
      pos_ = (apos + n);
      return p;
    } else {
      return allocate_big((n+mask) & ~mask);
    }
  }
    
  inline char* allocate(ssize_t n) {
    if (n <= kMaxSmall) {
      ssize_t apos = pos_;
      if (apos+n > kChunkCap) {
        new_chunk();
        apos = 0;
      }
      char* p = (chunk_.get() + apos);
      pos_ = (apos + n);
      return p;
    } else {
      return allocate_big(n);
    }
  }
  
  virtual const char* get_class_name() { return "Region"; }
  
  virtual int64_t get_field_offset(int field_size, const char* s) {
    if (strcmp(s, "chunk_") == 0) return ((int64_t)&chunk_ - (int64_t)this);
    if (strcmp(s, "pos_") == 0) return ((int64_t)&chunk_ - (int64_t)this);
    return -1;
  }
};

} // end hail

#endif
