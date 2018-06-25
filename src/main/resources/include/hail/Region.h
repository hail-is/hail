#ifndef HAIL_REGION_H
#define HAIL_REGION_H 1

#include "hail/hail.h"
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
  
  class UniqueChunk {
   public:
    char* buf_;
    ssize_t size_;
    
    UniqueChunk(const UniqueChunk& b) = delete;
    
    // no-args constructor is needed only for resize()
    UniqueChunk() : buf_(nullptr), size_(0) { }
    
    UniqueChunk(ssize_t n) : buf_((char*)malloc(n)), size_(n) { }
   
    UniqueChunk(UniqueChunk&& b) : buf_(b.buf_), size_(b.size_) { b.buf_ = nullptr; }
    
    ~UniqueChunk() { if (buf_) free(buf_); }
     
    UniqueChunk& operator=(UniqueChunk&& b) {
      if (buf_) free(buf_);
      buf_ = b.buf_;
      size_ = b.size_;
      b.buf_ = nullptr;
      return *this;
    }
  };
  
public:
  char* buf_;
  ssize_t pos_;
  ssize_t chunks_used_;
  std::vector<UniqueChunk> chunks_;
  std::vector<UniqueChunk> big_free_;
  std::vector<UniqueChunk> big_used_;
  std::vector<RegionPtr> required_regions_;

 public:  
  Region() :
    buf_(nullptr),
    pos_(kChunkCap),
    chunks_used_(0) {
  }
  
  // clear_but_keep_mem() will make the Region empty without free'ing chunks
  void clear_but_keep_mem();
  
 private:
  char* new_chunk_alloc(ssize_t n);
  
  // Restrict the choice of block sizes to improve re-use
  ssize_t choose_big_size(ssize_t n);
  
  char* big_alloc(ssize_t n);
 
 public: 
  inline void align(ssize_t a) {
    pos_ = (pos_ + a-1) & ~(a-1);
  }
  
  inline char* allocate(ssize_t a, ssize_t n) {
    ssize_t mask = (a-1);
    ssize_t apos = ((pos_ + mask) & ~mask);
    if (apos+n <= kChunkCap) {
      char* p = (buf_ + apos);
      pos_ = (apos + n);
      return p;
    }
    return (n <= kMaxSmall) ? new_chunk_alloc(n) : big_alloc(n);
  }
    
  inline char* allocate(ssize_t n) {
    ssize_t apos = pos_;
    if (apos+n <= kChunkCap) {
      char* p = (buf_ + apos);
      pos_ = (apos + n);
      return p;
    }
    return (n <= kMaxSmall) ? new_chunk_alloc(n) : big_alloc(n);
  }
  
  virtual const char* get_class_name() { return "Region"; }
  
  virtual int64_t get_field_offset(int field_size, const char* s) {
    if (strcmp(s, "buf_") == 0) return ((char*)&buf_ - (char*)this);
    if (strcmp(s, "pos_") == 0) return ((char*)&pos_ - (char*)this);
    return -1;
  }
};

} // end hail

#endif
