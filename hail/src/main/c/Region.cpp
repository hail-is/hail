#include "hail/Region.h"
#include "hail/NativeObj.h"
#include "hail/NativePtr.h"
#include <jni.h>

namespace hail {

constexpr ssize_t Region::kChunkCap;
constexpr ssize_t Region::kMaxSmall;
constexpr ssize_t Region::kNumBigToKeep;

void Region::clear_but_keep_mem() {
  buf_ = nullptr;
  pos_ = kChunkCap;
  chunks_used_ = 0;
  // Move big_used_ to big_free_
  for (auto& used : big_used_) big_free_.emplace_back(std::move(used));
  big_used_.clear();
  // Sort in descending order of size
  std::sort(
    big_free_.begin(),
    big_free_.end(),
    [](const UniqueChunk& a, const UniqueChunk& b)->bool { return (a.size_ > b.size_); }
  );
  if (big_free_.size() > kNumBigToKeep) big_free_.resize(kNumBigToKeep);
}

char* Region::new_chunk_alloc(ssize_t n) {
  if (chunks_used_ >= ssize(chunks_)) {
    chunks_.emplace_back(kChunkCap);
  }
  auto& chunk = chunks_[chunks_used_++];
  buf_ = chunk.buf_;
  pos_ = n;
  return buf_;
}
  
// Restrict the choice of block sizes to improve re-use
ssize_t Region::choose_big_size(ssize_t n) {
  for (ssize_t b = 1024;; b <<= 1) {
    if (n <= b) return b;
    // sqrt(2) is 1.414213, 181/128 is 1.414062
    ssize_t bmid = (((181*b) >> 7) + 0x3f) & ~0x3f;
    if (n <= bmid) return bmid;
  }
}
  
char* Region::big_alloc(ssize_t n) {
  char* buf = nullptr;
  n = ((n + 0x3f) & ~0x3f); // round up to multiple of 64byte cache line
  for (ssize_t idx = big_free_.size(); --idx >= 0;) {
    auto& b = big_free_[idx];
    if (n <= big_free_[idx].size_) {
      buf = b.buf_;
      big_used_.emplace_back(std::move(b));
      // Fast enough for small kNumBigToKeep
      big_free_.erase(big_free_.begin()+idx);
      return buf;
    }
  }
  n = choose_big_size(n);
  big_used_.emplace_back(n);
  return big_used_.back().buf_;
}

#define REGIONMETHOD(rtype, scala_class, scala_method) \
  extern "C" __attribute__((visibility("default"))) \
    rtype Java_is_hail_annotations_##scala_class##_##scala_method

REGIONMETHOD(void, Region, nativeCtor)(
  JNIEnv* env,
  jobject thisJ
) {
  NativeObjPtr ptr = std::make_shared<Region>();
  init_NativePtr(env, thisJ, &ptr);
}

REGIONMETHOD(void, Region, clearButKeepMem)(
  JNIEnv* env,
  jobject thisJ
) {
  auto r = static_cast<Region*>(get_from_NativePtr(env, thisJ));
  r->clear_but_keep_mem();
}

REGIONMETHOD(void, Region, nativeAlign)(
  JNIEnv* env,
  jobject thisJ,
  jlong a
) {
  auto r = static_cast<Region*>(get_from_NativePtr(env, thisJ));
  r->align(a);
}

REGIONMETHOD(jlong, Region, nativeAlignAllocate)(
  JNIEnv* env,
  jobject thisJ,
  jlong a,
  jlong n
) {
  auto r = static_cast<Region*>(get_from_NativePtr(env, thisJ));
  return reinterpret_cast<jlong>(r->allocate(a, n));
}

REGIONMETHOD(jlong, Region, nativeAllocate)(
  JNIEnv* env,
  jobject thisJ,
  jlong n
) {
  auto r = static_cast<Region*>(get_from_NativePtr(env, thisJ));
  return reinterpret_cast<jlong>(r->allocate(n));
}

} // end hail
