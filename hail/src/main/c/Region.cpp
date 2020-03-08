#include "hail/RegionPool.h"
#include "hail/NativePtr.h"
#include "hail/Upcalls.h"
#include "hail/Utils.h"
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

Region::Region(RegionPool * pool, size_t block_size) :
pool_(pool),
block_size_(block_size),
block_threshold_((block_size < BLOCK_THRESHOLD) ? block_size : BLOCK_THRESHOLD),
block_offset_(0),
current_block_(pool->get_block(block_size)) { }

char * Region::allocate_new_block(size_t n) {
  used_blocks_.push_back(std::move(current_block_));
  current_block_ = pool_->get_block(block_size_);
  block_offset_ = n;
  return current_block_.get();
}

char * Region::allocate_big_chunk(size_t n) {
  big_chunks_.push_back(std::unique_ptr<char[]>(new char[n]));
  return big_chunks_.back().get();
}

void Region::clear() {
  block_offset_ = 0;
  std::move(std::begin(used_blocks_), std::end(used_blocks_), std::back_inserter(*pool_->get_block_pool(block_size_)));
  used_blocks_.clear();
  big_chunks_.clear();
  parents_.clear();
  pool_->get_block_pool(block_size_)->push_back(std::move(current_block_));
  current_block_ = nullptr;
}

RegionPtr Region::get_region(size_t block_size) {
  return pool_->get_region(block_size);
}

void Region::add_reference_to(RegionPtr region) {
  parents_.push_back(std::move(region));
}

size_t Region::get_num_parents() {
  return parents_.size();
}

void Region::set_num_parents(int n) {
  parents_.resize(n, nullptr);
}

void Region::set_parent_reference(RegionPtr region, int i) {
  parents_[i] = region;
}

RegionPtr Region::get_parent_reference(int i) { return parents_[i]; }

RegionPtr Region::new_parent_reference(int i, size_t block_size) {
  auto r = get_region(block_size);
  parents_[i] = r;
  return r;
}

void Region::clear_parent_reference(int i) {
  parents_[i] = nullptr;
}
std::unique_ptr<char[]> RegionPool::get_block(size_t size) {
  auto free_blocks = get_block_pool(size);
  if (free_blocks->empty()) {
    return std::unique_ptr<char[]>(new char[size]);
  }
  std::unique_ptr<char[]> block = std::move(free_blocks->back());
  free_blocks->pop_back();
  return block;
}

RegionPtr RegionPool::new_region(size_t block_size) {
  regions_.emplace_back(new Region(this, block_size));
  return RegionPtr(regions_.back().get());
}

RegionPtr RegionPool::get_region(size_t block_size) {
  if (free_regions_.empty()) {
    return new_region(block_size);
  }
  Region * region = free_regions_.back();
  region->set_block_size(block_size);
  region->current_block_ = get_block(block_size);
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

ScalaRegion::ScalaRegion(ScalaRegionPool * pool, size_t block_size) :
region_(pool->pool_.get_region(block_size)) { }

ScalaRegion::ScalaRegion(std::nullptr_t) :
region_(nullptr) { }

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

REGIONMETHOD(jint, RegionPool, numRegions)(
  JNIEnv* env,
  jobject thisJ
) {
  auto pool = static_cast<ScalaRegionPool*>(get_from_NativePtr(env, thisJ));
  return (jint) pool->pool_.num_regions();
}

REGIONMETHOD(jint, RegionPool, numFreeRegions)(
  JNIEnv* env,
  jobject thisJ
) {
  auto pool = static_cast<ScalaRegionPool*>(get_from_NativePtr(env, thisJ));
  return (jint) pool->pool_.num_free_regions();
}

REGIONMETHOD(jint, RegionPool, numFreeBlocks)(
  JNIEnv* env,
  jobject thisJ
) {
    auto pool = static_cast<ScalaRegionPool*>(get_from_NativePtr(env, thisJ));
    return (jint) pool->pool_.num_free_blocks();
}

REGIONMETHOD(void, Region, nativeCtor)(
  JNIEnv* env,
  jobject thisJ,
  jobject poolJ,
  jint blockSizeJ
) {
  auto pool = static_cast<ScalaRegionPool*>(get_from_NativePtr(env, poolJ));
  size_t block_size = (size_t) blockSizeJ;
  NativeObjPtr ptr = std::make_shared<ScalaRegion>(pool, block_size);
  init_NativePtr(env, thisJ, &ptr);
}

REGIONMETHOD(void, Region, clearButKeepMem)(
  JNIEnv*,
  jobject,
  jlong addr
) {
  auto r = reinterpret_cast<ScalaRegion*>(addr);
  r->region_ = r->region_->get_region(r->region_->get_block_size());
}

REGIONMETHOD(void, Region, nativeAlign)(
  JNIEnv*,
  jobject,
  jlong addr,
  jlong a
) {
  auto r = reinterpret_cast<ScalaRegion*>(addr);
  r->region_->align(a);
}

REGIONMETHOD(jlong, Region, nativeAlignAllocate)(
  JNIEnv*,
  jobject,
  jlong addr,
  jlong a,
  jlong n
) {
  auto r = reinterpret_cast<ScalaRegion*>(addr);
  return reinterpret_cast<jlong>(r->region_->allocate((size_t)a, (size_t)n));
}

REGIONMETHOD(jlong, Region, nativeAllocate)(
  JNIEnv*,
  jobject,
  jlong addr,
  jlong n
) {
  auto r = reinterpret_cast<ScalaRegion*>(addr);
  return reinterpret_cast<jlong>(r->region_->allocate((size_t)n));
}

REGIONMETHOD(void, Region, nativeReference)(
  JNIEnv*,
  jobject,
  jlong thisAddr,
  jlong otherAddr
) {
  auto r = reinterpret_cast<ScalaRegion*>(thisAddr);
  auto r2 = reinterpret_cast<ScalaRegion*>(otherAddr);
  r->region_->add_reference_to(r2->region_);
}

REGIONMETHOD(void, Region, nativeGetNewRegion)(
  JNIEnv*,
  jobject,
  jlong addr,
  jlong addrPool,
  jint blockSizeJ
) {
  auto r = reinterpret_cast<ScalaRegion*>(addr);
  auto pool = reinterpret_cast<ScalaRegionPool*>(addrPool);
  r->region_ = pool->pool_.get_region((size_t) blockSizeJ);
}

REGIONMETHOD(jint, Region, nativeGetNumParents)(
  JNIEnv*,
  jobject,
  jlong addr
) {
  auto r = reinterpret_cast<ScalaRegion*>(addr);
  return (jint) r->region_->get_num_parents();
}

REGIONMETHOD(void, Region, nativeSetNumParents)(
  JNIEnv*,
  jobject,
  jlong addr,
  jint i
) {
  auto r = reinterpret_cast<ScalaRegion*>(addr);
  r->region_->set_num_parents((int) i);
}

REGIONMETHOD(void, Region, nativeSetParentReference)(
  JNIEnv*,
  jobject,
  jlong addr1,
  jlong addr2,
  jint i
) {
  auto r = reinterpret_cast<ScalaRegion*>(addr1);
  auto r2 = reinterpret_cast<ScalaRegion*>(addr2);
  r->region_->set_parent_reference(r2->region_, (int) i);
}

REGIONMETHOD(void, Region, nativeGetParentReferenceInto)(
  JNIEnv*,
  jobject,
  jlong addr1,
  jlong addr2,
  jint i,
  jint blockSizeJ
) {
  auto r = reinterpret_cast<ScalaRegion*>(addr1);
  auto r2 = reinterpret_cast<ScalaRegion*>(addr2);
  auto block_size = (size_t) blockSizeJ;
  r2->region_ = r->region_->get_parent_reference((int) i);
  if (r2->region_.get() == nullptr) {
    r2->region_ = r->region_->new_parent_reference((int) i, block_size);
  } else {
    if (r2->region_->get_block_size() != block_size) {
      throw new FatalError("blocksizes are wrong!");
    }
  }
}

REGIONMETHOD(void, Region, nativeClearParentReference)(
  JNIEnv*,
  jobject,
  jlong addr,
  jint i
) {
  auto r = reinterpret_cast<ScalaRegion*>(addr);
  r->region_->clear_parent_reference((int) i);
}

REGIONMETHOD(jint, Region, nativeGetBlockSize)(
  JNIEnv* env,
  jobject thisJ
) {
  auto r = static_cast<ScalaRegion*>(get_from_NativePtr(env, thisJ));
  return (jint) r->region_->get_block_size();
}

REGIONMETHOD(jint, Region, nativeGetNumChunks)(
  JNIEnv* env,
  jobject thisJ
) {
  auto r = static_cast<ScalaRegion*>(get_from_NativePtr(env, thisJ));
  return (jint) r->region_->get_num_chunks();
}

REGIONMETHOD(jint, Region, nativeGetNumUsedBlocks)(
  JNIEnv* env,
  jobject thisJ
) {
  auto r = static_cast<ScalaRegion*>(get_from_NativePtr(env, thisJ));
  return (jint) r->region_->get_num_used_blocks();
}


REGIONMETHOD(jint, Region, nativeGetCurrentOffset)(
  JNIEnv* env,
  jobject thisJ
) {
  auto r = static_cast<ScalaRegion*>(get_from_NativePtr(env, thisJ));
  return (jint) r->region_->get_current_offset();
}

REGIONMETHOD(jlong, Region, nativeGetBlockAddress)(
  JNIEnv* env,
  jobject thisJ
) {
  auto r = static_cast<ScalaRegion*>(get_from_NativePtr(env, thisJ));
  return (jlong) r->region_->get_block_address();
}

REGIONMETHOD(void, Region, setNull)(
  JNIEnv*,
  jobject,
  jlong addr
) {
  auto r = reinterpret_cast<ScalaRegion*>(addr);
  r->region_ = nullptr;
}

}