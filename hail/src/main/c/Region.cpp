#include "hail/Region.h"
#include "hail/Upcalls.h"
#include <memory>
#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>

namespace hail {

void RegionPtr::clear() {
  if (region_ != nullptr) {
    --(region_->references_);
    if (region_->references_ == 0) {
      region_->clear();
      region_->pool_->free_regions_.push_back(region_);
    }
    region_ = nullptr;
  }
}

Region::Region(RegionPool * pool) :
pool_(pool),
block_offset_(0),
current_block_(pool->get_block()) { }

char * Region::allocate_new_block(size_t n) {
  used_blocks_.push_back(std::move(current_block_));
  current_block_ = pool_->get_block();
  block_offset_ = n;
  return current_block_.get();
}

char * Region::allocate_big_chunk(size_t n) {
  big_chunks_.push_back(std::make_unique<char[]>(n));
  return big_chunks_.back().get();
}

void Region::clear() {
  block_offset_ = 0;
  std::move(std::begin(used_blocks_), std::end(used_blocks_), std::back_inserter(pool_->free_blocks_));
  used_blocks_.clear();
  big_chunks_.clear();
  parents_.clear();
}

RegionPtr Region::get_region() {
  return pool_->get_region();
}

void Region::add_reference_to(RegionPtr region) {
  parents_.push_back(std::move(region));
}

std::unique_ptr<char[]> RegionPool::get_block() {
  if (free_blocks_.empty()) {
    return std::make_unique<char[]>(block_size);
  }
  std::unique_ptr<char[]> block = std::move(free_blocks_.back());
  free_blocks_.pop_back();
  return block;
}

RegionPtr RegionPool::new_region() {
  regions_.emplace_back(new Region(this));
  return RegionPtr(regions_.back().get());
}

RegionPtr RegionPool::get_region() {
  if (free_regions_.empty()) {
    return new_region();
  }
  Region * region = std::move(free_regions_.back());
  free_regions_.pop_back();
  return RegionPtr(region);
}

void ScalaRegionPool::own(RegionPool &&pool) {
  for (auto &region : pool.regions_) {
    if (region->references_ != 0) {
      region->pool_ = &this->pool_;
      this->pool_.regions_.push_back(std::move(region));
    }
  }
}

void ScalaRegionPool::Region::clear() {
  auto r2 = region_->get_region();
  region_ = nullptr;
  region_ = std::move(r2);
}

#define REGIONMETHOD(rtype, scala_class, scala_method) \
  extern "C" __attribute__((visibility("default"))) \
    rtype Java_is_hail_annotations_##scala_class##_##scala_method

REGIONMETHOD(void, RegionPool, nativeCtor)(
  JNIEnv* env,
  jobject thisJ
) {
  NativeObjPtr ptr = std::make_shared<ScalaRegionPool>();
  init_NativePtr(env, thisJ, &ptr);
}

REGIONMETHOD(void, Region, nativeCtor)(
  JNIEnv* env,
  jobject thisJ,
  jobject poolJ
) {
  auto pool = static_cast<ScalaRegionPool*>(get_from_NativePtr(env, poolJ));
  NativeObjPtr ptr = std::make_shared<ScalaRegionPool::Region>(pool);
  init_NativePtr(env, thisJ, &ptr);
}

REGIONMETHOD(void, Region, clearButKeepMem)(
  JNIEnv* env,
  jobject thisJ
) {
  auto r = static_cast<ScalaRegionPool::Region*>(get_from_NativePtr(env, thisJ));
  r->clear();
}

REGIONMETHOD(void, Region, nativeAlign)(
  JNIEnv* env,
  jobject thisJ,
  jlong a
) {
  auto r = static_cast<ScalaRegionPool::Region*>(get_from_NativePtr(env, thisJ));
  r->align(a);
}

REGIONMETHOD(jlong, Region, nativeAlignAllocate)(
  JNIEnv* env,
  jobject thisJ,
  jlong a,
  jlong n
) {
  auto r = static_cast<ScalaRegionPool::Region*>(get_from_NativePtr(env, thisJ));
  return reinterpret_cast<jlong>(r->allocate((size_t)a, (size_t)n));
}

REGIONMETHOD(jlong, Region, nativeAllocate)(
  JNIEnv* env,
  jobject thisJ,
  jlong n
) {
  auto r = static_cast<ScalaRegionPool::Region*>(get_from_NativePtr(env, thisJ));
  return reinterpret_cast<jlong>(r->allocate((size_t)n));
}

}