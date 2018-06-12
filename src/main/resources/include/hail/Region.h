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
  static constexpr int64_t kChunkCap = 64*1024;
  static constexpr int64_t kMaxSmall = 4*1024;
  static constexpr size_t kNumBigToKeep = 4;
  
  struct BigAlloc {
    char* buf_;
    int64_t size_;
    
    inline BigAlloc() : buf_(nullptr), size_(0) { }

    inline BigAlloc(char* buf, int64_t size) : buf_(buf), size_(size) { }
    
    inline BigAlloc(const BigAlloc& b) : buf_(b.buf_), size_(b.size_) { }
    
    inline BigAlloc& operator=(const BigAlloc& b) {
      buf_ = b.buf_;
      size_ = b.size_;
      return *this;
    }
  };

public:
  char* chunk_;
  int64_t pos_;
  std::vector<char*> free_chunks_;
  std::vector<char*> full_chunks_;
  std::vector<BigAlloc> big_free_;
  std::vector<BigAlloc> big_used_;
  std::vector<RegionPtr> required_regions_;

public:  
  Region() :
    chunk_(nullptr),
    pos_(kChunkCap) {
  }
  
  virtual ~Region() {
    if (chunk_) free(chunk_);
    for (auto p : free_chunks_) free(p);
    for (auto p : full_chunks_) free(p);
    for (auto& b : big_free_) free(b.buf_);
    for (auto& b : big_used_) free(b.buf_);
  }
  
  // clear_but_keep_mem() will make the Region empty without free'ing chunks
  void clear_but_keep_mem() {
    pos_ = (chunk_ ? 0 : kChunkCap);
    for (auto p : full_chunks_) free_chunks_.push_back(p);
    full_chunks_.clear();
    // Move big_used_ to big_free_
    for (auto& used : big_used_) {
      big_free_.emplace_back(used.buf_, used.size_);
      used.buf_ = nullptr;
    }
    big_used_.clear();
    // Sort in descending order of size
    std::sort(
      big_free_.begin(),
      big_free_.end(),
      [](const BigAlloc& a, const BigAlloc& b)->bool { return (a.size_ > b.size_); }
    );
    if (big_free_.size() > kNumBigToKeep) {
      for (int idx = big_free_.size(); --idx >= (int)kNumBigToKeep;) {
        free(big_free_[idx].buf_);
      }
      big_free_.resize(kNumBigToKeep);
    }
  }
  
  void new_chunk() {
    if (chunk_) full_chunks_.push_back(chunk_);
    if (free_chunks_.empty()) {
      chunk_ = (char*)malloc(kChunkCap);
    } else {
      chunk_ = free_chunks_.back();
      free_chunks_.pop_back();
    }
    pos_ = 0;
  }
  
  // Restrict the choice of block sizes to improve re-use
  int64_t choose_big_size(int64_t n) {
    for (int64_t b = 1024;; b <<= 1) {
      if (n <= b) return b;
      // sqrt(2) is 1.414213, 181/128 is 1.414062
      int64_t bmid = (((181*b) >> 7) + 0x3f) & ~0x3f;
      if (n <= bmid) return bmid;
    }
  }
  
  char* allocate_big(int64_t n) {
    char* buf = nullptr;
    //
    n = ((n + 0x3f) & ~0x3f); // round up to multiple of 64byte cache line
    for (int idx = big_free_.size(); --idx >= 0;) {
      auto& b = big_free_[idx];
      if (n <= big_free_[idx].size_) {
        buf = b.buf_;
        b.buf_ = nullptr;
        big_used_.emplace_back(buf, b.size_);
        // Fast enough for small kNumBigToKeep
        big_free_.erase(big_free_.begin()+idx);
        return buf;
      }
    }
    n = choose_big_size(n);
    buf = (char*)malloc(n);
    big_used_.emplace_back(buf, n);
    return buf;
  }
  
  inline void align(int64_t a) {
    pos_ = (pos_ + a-1) & ~(a-1);
  }
  
  inline char* allocate(int64_t a, int64_t n) {
    int64_t mask = (a-1);
    if (n <= kMaxSmall) {
      int64_t apos = ((pos_ + mask) & ~mask);
      if (apos+n > kChunkCap) {
        new_chunk();
        apos = 0;
      }
      char* p = (chunk_ + apos);
      pos_ = (apos + n);
      return p;
    } else {
      return allocate_big((n+mask) & ~mask);
    }
  }
    
  inline char* allocate(int64_t n) {
    if (n <= kMaxSmall) {
      int64_t apos = pos_;
      if (apos+n > kChunkCap) {
        new_chunk();
        apos = 0;
      }
      char* p = (chunk_ + apos);
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
